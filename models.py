# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Models for TrAISformer.
    https://arxiv.org/abs/2109.03958

The code is built upon:
    https://github.com/karpathy/minGPT
"""

import math
import logging
import pdb


import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.max_seqlen, config.max_seqlen))
                                     .view(1, 1, config.max_seqlen, config.max_seqlen))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TrAISformer(nn.Module):
    """Transformer for AIS trajectories."""

    def __init__(self, config, partition_model = None):
        super().__init__()

        self.lat_size = config.lat_size
        self.lon_size = config.lon_size
        self.sog_size = config.sog_size
        self.cog_size = config.cog_size
        self.full_size = config.full_size
        self.n_lat_embd = config.n_lat_embd
        self.n_lon_embd = config.n_lon_embd
        self.n_sog_embd = config.n_sog_embd
        self.n_cog_embd = config.n_cog_embd
        self.register_buffer(
            "att_sizes", 
            torch.tensor([config.lat_size, config.lon_size, config.sog_size, config.cog_size]))
        self.register_buffer(
            "emb_sizes", 
            torch.tensor([config.n_lat_embd, config.n_lon_embd, config.n_sog_embd, config.n_cog_embd]))
        
        if hasattr(config,"partition_mode"):
            self.partition_mode = config.partition_mode
        else:
            self.partition_mode = "uniform"
        self.partition_model = partition_model
        
        if hasattr(config,"blur"):
            self.blur = config.blur
            self.blur_learnable = config.blur_learnable
            self.blur_loss_w = config.blur_loss_w
            self.blur_n = config.blur_n
            if self.blur:
                self.blur_module = nn.Conv1d(1, 1, 3, padding = 1, padding_mode = 'replicate', groups=1, bias=False)
                if not self.blur_learnable:
                    for params in self.blur_module.parameters():
                        params.requires_grad = False
                        params.fill_(1/3)
            else:
                self.blur_module = None
                
        
        if hasattr(config,"lat_min"): # the ROI is provided.
            self.lat_min = config.lat_min
            self.lat_max = config.lat_max
            self.lon_min = config.lon_min
            self.lon_max = config.lon_max
            self.lat_range = config.lat_max-config.lat_min
            self.lon_range = config.lon_max-config.lon_min
            self.sog_range = 30.
            
        if hasattr(config,"mode"): # mode: "pos" or "velo".
            # "pos": predict directly the next positions.
            # "velo": predict the velocities, use them to 
            # calculate the next positions.
            self.mode = config.mode
        else:
            self.mode = "pos"
    

        # Passing from the 4-D space to a high-dimentional space
        self.lat_emb = nn.Embedding(self.lat_size, config.n_lat_embd)
        self.lon_emb = nn.Embedding(self.lon_size, config.n_lon_embd)
        self.sog_emb = nn.Embedding(self.sog_size, config.n_sog_embd)
        self.cog_emb = nn.Embedding(self.cog_size, config.n_cog_embd)
            
            
        self.pos_emb = nn.Parameter(torch.zeros(1, config.max_seqlen, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        self.use_map_prior = getattr(config, "use_map_prior", False)
        self.map_emb_w = float(getattr(config, "map_emb_w", 0.1))
        self.map_prior_channels = int(getattr(config, "map_prior_channels", 4))
        if self.use_map_prior:
            self.map_encoder = nn.Sequential(
                nn.Linear(self.map_prior_channels, config.n_embd),
                nn.GELU(),
                nn.Dropout(getattr(config, "map_emb_pdrop", 0.0)),
                nn.Linear(config.n_embd, config.n_embd),
                nn.LayerNorm(config.n_embd),
            )
        else:
            self.map_encoder = None

        self.use_turn_intent_head = getattr(config, "use_turn_intent_head", False)
        self.turn_intent_loss_w = float(getattr(config, "turn_intent_loss_w", 0.0))
        self.turn_straight_threshold_deg = float(getattr(config, "turn_straight_threshold_deg", 10.0))
        self.turn_sharp_threshold_deg = float(getattr(config, "turn_sharp_threshold_deg", 35.0))
        self.turn_ignore_first = getattr(config, "turn_ignore_first", True)
        if self.use_turn_intent_head:
            self.turn_head = nn.Linear(config.n_embd, 5)
        else:
            self.turn_head = None

        self.use_qwen_semantic_encoder = getattr(config, "use_qwen_semantic_encoder", False)
        self.qwen_hidden_size = int(getattr(config, "qwen_hidden_size", 2560))
        self.qwen_fusion = getattr(config, "qwen_fusion", "both")
        self.qwen_emb_w = float(getattr(config, "qwen_emb_w", 0.05))
        self.qwen_bias_w = float(getattr(config, "qwen_bias_w", 0.05))
        self.qwen_logit_bias_radius = float(getattr(config, "qwen_logit_bias_radius", getattr(config, "r_vicinity", 40)))
        if self.use_qwen_semantic_encoder:
            self.qwen_projector = nn.Sequential(
                nn.LayerNorm(self.qwen_hidden_size),
                nn.Linear(self.qwen_hidden_size, config.n_embd),
                nn.GELU(),
                nn.Dropout(getattr(config, "qwen_emb_pdrop", 0.0)),
                nn.Linear(config.n_embd, config.n_embd),
                nn.LayerNorm(config.n_embd),
            )
            self.qwen_gate = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd),
                nn.Sigmoid(),
            )
            self.captain_selector = nn.Sequential(
                nn.LayerNorm(self.qwen_hidden_size),
                nn.Linear(self.qwen_hidden_size, max(64, config.n_embd // 4)),
                nn.GELU(),
                nn.Linear(max(64, config.n_embd // 4), 5),
                nn.Tanh(),
            )
        else:
            self.qwen_projector = None
            self.qwen_gate = None
            self.captain_selector = None

        self.use_grid_residual = getattr(config, "use_grid_residual", False)
        self.grid_residual_loss_w = float(getattr(config, "grid_residual_loss_w", 0.0))
        self.grid_residual_max_abs_cell = float(getattr(config, "grid_residual_max_abs_cell", 0.5))
        if self.use_grid_residual:
            self.grid_residual_head = nn.Sequential(
                nn.LayerNorm(config.n_embd),
                nn.Linear(config.n_embd, max(64, config.n_embd // 2)),
                nn.GELU(),
                nn.Dropout(getattr(config, "grid_residual_pdrop", 0.0)),
                nn.Linear(max(64, config.n_embd // 2), 2),
            )
        else:
            self.grid_residual_head = None
        
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        
        
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        if self.mode in ("mlp_pos","mlp"):
            self.head = nn.Linear(config.n_embd, config.n_embd, bias=False)
        else:
            self.head = nn.Linear(config.n_embd, self.full_size, bias=False) # Classification head
            
        self.max_seqlen = config.max_seqlen
        self.apply(self._init_weights)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_max_seqlen(self):
        return self.max_seqlen

    def register_map_prior(self, features, direction_probs=None):
        """Register map prior arrays as non-persistent buffers."""
        features = torch.as_tensor(features, dtype=torch.float32)
        if features.ndim != 3:
            raise ValueError("map prior features must have shape [H, W, C]")
        if features.shape[-1] != self.map_prior_channels:
            raise ValueError(
                f"map prior has {features.shape[-1]} channels, "
                f"but model expects {self.map_prior_channels}"
            )
        self.register_buffer("map_prior_features", features, persistent=False)
        if direction_probs is not None:
            self.register_buffer(
                "map_direction_probs",
                torch.as_tensor(direction_probs, dtype=torch.float32),
                persistent=False,
            )

    def query_map_features(self, x_real):
        if not self.use_map_prior:
            return None
        if not hasattr(self, "map_prior_features"):
            raise RuntimeError("use_map_prior=True but no map prior has been registered")

        features = self.map_prior_features.to(device=x_real.device, dtype=x_real.dtype)
        H, W, _ = features.shape
        lat_idx = torch.clamp((x_real[..., 0] * H).long(), 0, H - 1)
        lon_idx = torch.clamp((x_real[..., 1] * W).long(), 0, W - 1)
        return features[lat_idx, lon_idx, :]

    def _bearing_deg_torch(self, lat1, lon1, lat2, lon2):
        lat1 = lat1 * math.pi / 180.0
        lat2 = lat2 * math.pi / 180.0
        dlon = (lon2 - lon1) * math.pi / 180.0
        y = torch.sin(dlon) * torch.cos(lat2)
        x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(dlon)
        return torch.remainder(torch.atan2(y, x) * 180.0 / math.pi + 360.0, 360.0)

    def _signed_circular_diff_deg_torch(self, a, b):
        return torch.remainder(a - b + 180.0, 360.0) - 180.0

    def make_turn_labels(self, x, input_len, masks=None):
        """Create turn intent labels for input positions.

        Bearing is measured clockwise from north, so positive heading deltas
        are right turns and negative heading deltas are left turns.
        """
        batchsize, full_len, _ = x.shape
        labels = torch.zeros((batchsize, input_len), dtype=torch.long, device=x.device)
        turn_mask = torch.zeros((batchsize, input_len), dtype=torch.bool, device=x.device)
        if full_len < 3 or input_len <= 1:
            return labels, turn_mask

        lat = self.lat_min + x[..., 0] * self.lat_range
        lon = self.lon_min + x[..., 1] * self.lon_range
        headings = self._bearing_deg_torch(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
        seg_valid = torch.abs(x[:, 1:, :2] - x[:, :-1, :2]).sum(dim=-1) > 1e-10

        center_count = min(input_len - 1, headings.size(1) - 1)
        if center_count <= 0:
            return labels, turn_mask

        delta = self._signed_circular_diff_deg_torch(
            headings[:, 1:1 + center_count],
            headings[:, :center_count],
        )
        center_labels = torch.zeros_like(delta, dtype=torch.long)
        straight = self.turn_straight_threshold_deg
        sharp = self.turn_sharp_threshold_deg

        center_labels[(delta < -straight) & (delta > -sharp)] = 1
        center_labels[(delta > straight) & (delta < sharp)] = 2
        center_labels[delta <= -sharp] = 3
        center_labels[delta >= sharp] = 4

        center_valid = seg_valid[:, 1:1 + center_count] & seg_valid[:, :center_count]
        if masks is not None:
            mask_bool = masks > 0
            center_valid = (
                center_valid
                & mask_bool[:, 1:1 + center_count]
                & mask_bool[:, :center_count]
            )
        if self.turn_ignore_first:
            turn_mask[:, 0] = False

        labels[:, 1:1 + center_count] = center_labels
        turn_mask[:, 1:1 + center_count] = center_valid
        return labels, turn_mask

    def _prepare_qwen(self, qwen_vec, qwen_mask=None):
        if not self.use_qwen_semantic_encoder:
            return None, None
        if qwen_vec is None:
            return None, None
        qwen_vec = qwen_vec.to(dtype=self.pos_emb.dtype, device=self.pos_emb.device)
        if qwen_mask is None:
            qwen_mask = torch.ones(qwen_vec.size(0), dtype=qwen_vec.dtype, device=qwen_vec.device)
        else:
            qwen_mask = qwen_mask.to(dtype=qwen_vec.dtype, device=qwen_vec.device)
        qwen_mask = qwen_mask.view(-1, 1)
        return qwen_vec, qwen_mask

    def _qwen_embedding(self, qwen_vec, qwen_mask):
        if qwen_vec is None or self.qwen_projector is None:
            return None
        qwen_emb = self.qwen_projector(qwen_vec) * qwen_mask
        if self.qwen_fusion in ("gated_add", "both"):
            qwen_emb = self.qwen_gate(qwen_emb) * qwen_emb
        return qwen_emb

    def _apply_qwen_logit_bias(self, logits, inputs_real, qwen_vec, qwen_mask):
        if (
            qwen_vec is None
            or self.captain_selector is None
            or self.qwen_bias_w <= 0
            or self.qwen_fusion not in ("logit_bias", "both")
            or logits.size(-1) != self.full_size
        ):
            return logits

        batchsize, seqlen, _ = inputs_real.shape
        direction_scores = self.captain_selector(qwen_vec) * qwen_mask
        direction_deltas = torch.tensor(
            [0.0, -25.0, 25.0, -70.0, 70.0],
            dtype=inputs_real.dtype,
            device=inputs_real.device,
        )

        heading = torch.remainder(inputs_real[..., 3] * 360.0, 360.0)
        bearings = torch.remainder(heading.unsqueeze(-1) + direction_deltas.view(1, 1, 5), 360.0)
        bearings_rad = bearings * math.pi / 180.0
        desired_lat = torch.cos(bearings_rad)
        desired_lon = torch.sin(bearings_rad)

        radius = max(self.qwen_logit_bias_radius, 1.0)
        lat_values = torch.arange(self.lat_size, dtype=inputs_real.dtype, device=inputs_real.device)
        lon_values = torch.arange(self.lon_size, dtype=inputs_real.dtype, device=inputs_real.device)
        cur_lat = inputs_real[..., 0] * (self.lat_size - 1)
        cur_lon = inputs_real[..., 1] * (self.lon_size - 1)
        lat_delta = torch.clamp((lat_values.view(1, 1, -1) - cur_lat.unsqueeze(-1)) / radius, -1.0, 1.0)
        lon_delta = torch.clamp((lon_values.view(1, 1, -1) - cur_lon.unsqueeze(-1)) / radius, -1.0, 1.0)

        score = direction_scores.view(batchsize, 1, 5)
        lat_bias = torch.sum(score.unsqueeze(-1) * desired_lat.unsqueeze(-1) * lat_delta.unsqueeze(2), dim=2)
        lon_bias = torch.sum(score.unsqueeze(-1) * desired_lon.unsqueeze(-1) * lon_delta.unsqueeze(2), dim=2)

        lat_logits, lon_logits, sog_logits, cog_logits = torch.split(
            logits,
            (self.lat_size, self.lon_size, self.sog_size, self.cog_size),
            dim=-1,
        )
        lat_logits = lat_logits + self.qwen_bias_w * lat_bias
        lon_logits = lon_logits + self.qwen_bias_w * lon_bias
        return torch.cat((lat_logits, lon_logits, sog_logits, cog_logits), dim=-1)

    def predict_grid_residual(self, fea):
        if self.grid_residual_head is None:
            return None
        residual = torch.tanh(self.grid_residual_head(fea))
        return self.grid_residual_max_abs_cell * residual

    def make_grid_residual_targets(self, targets_uniform, targets_real):
        """Return normalized sub-grid offsets for lat/lon targets.

        Each offset is measured in cell units relative to the target cell center,
        so values are roughly in [-0.5, 0.5].
        """
        lat_target = targets_real[..., 0] * float(self.lat_size) - (targets_uniform[..., 0].float() + 0.5)
        lon_target = targets_real[..., 1] * float(self.lon_size) - (targets_uniform[..., 1].float() + 0.5)
        target = torch.stack((lat_target, lon_target), dim=-1)
        return torch.clamp(
            target,
            -self.grid_residual_max_abs_cell,
            self.grid_residual_max_abs_cell,
        )

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer
   
    
    def to_indexes(self, x, mode="uniform"):
        """Convert tokens to indexes.
        
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated 
                to [0,1).
            model: currenly only supports "uniform".
        
        Returns:
            idxs: a Tensor (dtype: Long) of indexes.
        """
        bs, seqlen, data_dim = x.shape
        if mode == "uniform":
            idxs = (x*self.att_sizes).long()
            return idxs, idxs
        elif mode in ("freq", "freq_uniform"):
            
            idxs = (x*self.att_sizes).long()
            idxs_uniform = idxs.clone()
            discrete_lats, discrete_lons, lat_ids, lon_ids = self.partition_model(x[:,:,:2])
#             pdb.set_trace()
            idxs[:,:,0] = torch.round(lat_ids.reshape((bs,seqlen))).long()
            idxs[:,:,1] = torch.round(lon_ids.reshape((bs,seqlen))).long()                               
            return idxs, idxs_uniform
    
    
    def forward(
            self,
            x,
            masks=None,
            with_targets=False,
            return_loss_tuple=False,
            qwen_vec=None,
            qwen_mask=None,
            return_residual=False):
        """
        Args:
            x: a Tensor of size (batchsize, seqlen, 4). x has been truncated 
                to [0,1).
            masks: a Tensor of the same size of x. masks[idx] = 0. if 
                x[idx] is a padding.
            with_targets: if True, inputs = x[:,:-1,:], targets = x[:,1:,:], 
                otherwise inputs = x.
        Returns: 
            logits, loss
        """
        
        if self.mode in ("mlp_pos","mlp",):
            idxs, idxs_uniform = x, x # use the real-values of x.
        else:            
            # Convert to indexes
            idxs, idxs_uniform = self.to_indexes(x, mode=self.partition_mode)
        
        if with_targets:
            inputs = idxs[:,:-1,:].contiguous()
            targets = idxs[:,1:,:].contiguous()
            targets_uniform = idxs_uniform[:,1:,:].contiguous()
            inputs_real = x[:,:-1,:].contiguous()
            targets_real = x[:,1:,:].contiguous()
        else:
            inputs_real = x
            inputs = idxs
            targets = None
            
        batchsize, seqlen, _ = inputs.size()
        assert seqlen <= self.max_seqlen, "Cannot forward, model block size is exhausted."

        qwen_vec, qwen_mask = self._prepare_qwen(qwen_vec, qwen_mask)

        # forward the GPT model
        lat_embeddings = self.lat_emb(inputs[:,:,0]) # (bs, seqlen, lat_size)
        lon_embeddings = self.lon_emb(inputs[:,:,1]) 
        sog_embeddings = self.sog_emb(inputs[:,:,2]) 
        cog_embeddings = self.cog_emb(inputs[:,:,3])      
        token_embeddings = torch.cat((lat_embeddings, lon_embeddings, sog_embeddings, cog_embeddings),dim=-1)
            
        position_embeddings = self.pos_emb[:, :seqlen, :] # each position maps to a (learnable) vector (1, seqlen, n_embd)
        embeddings = token_embeddings + position_embeddings
        if self.use_map_prior:
            map_features = self.query_map_features(inputs_real)
            map_embeddings = self.map_encoder(map_features)
            embeddings = embeddings + self.map_emb_w * map_embeddings
        qwen_emb = self._qwen_embedding(qwen_vec, qwen_mask)
        if qwen_emb is not None and self.qwen_fusion in ("add", "gated_add", "both"):
            embeddings = embeddings + self.qwen_emb_w * qwen_emb.unsqueeze(1)
        fea = self.drop(embeddings)
        fea = self.blocks(fea)
        fea = self.ln_f(fea) # (bs, seqlen, n_embd)
        logits = self.head(fea) # (bs, seqlen, full_size) or (bs, seqlen, n_embd)
        logits = self._apply_qwen_logit_bias(logits, inputs_real, qwen_vec, qwen_mask)
        grid_residual = self.predict_grid_residual(fea)
        
        lat_logits, lon_logits, sog_logits, cog_logits =\
            torch.split(logits, (self.lat_size, self.lon_size, self.sog_size, self.cog_size), dim=-1)
        
        # Calculate the loss
        loss = None
        loss_tuple = None
        if targets is not None:

            sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_size), 
                                       targets[:,:,2].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_size), 
                                       targets[:,:,3].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_size), 
                                       targets[:,:,0].view(-1), 
                                       reduction="none").view(batchsize,seqlen)
            lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_size), 
                                       targets[:,:,1].view(-1), 
                                       reduction="none").view(batchsize,seqlen)                     

            if self.blur:
                lat_probs = F.softmax(lat_logits, dim=-1) 
                lon_probs = F.softmax(lon_logits, dim=-1)
                sog_probs = F.softmax(sog_logits, dim=-1)
                cog_probs = F.softmax(cog_logits, dim=-1)

                for _ in range(self.blur_n):
                    blurred_lat_probs = self.blur_module(lat_probs.reshape(-1,1,self.lat_size)).reshape(lat_probs.shape)
                    blurred_lon_probs = self.blur_module(lon_probs.reshape(-1,1,self.lon_size)).reshape(lon_probs.shape)
                    blurred_sog_probs = self.blur_module(sog_probs.reshape(-1,1,self.sog_size)).reshape(sog_probs.shape)
                    blurred_cog_probs = self.blur_module(cog_probs.reshape(-1,1,self.cog_size)).reshape(cog_probs.shape)

                    blurred_lat_loss = F.nll_loss(blurred_lat_probs.view(-1, self.lat_size),
                                                  targets[:,:,0].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_lon_loss = F.nll_loss(blurred_lon_probs.view(-1, self.lon_size),
                                                  targets[:,:,1].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_sog_loss = F.nll_loss(blurred_sog_probs.view(-1, self.sog_size),
                                                  targets[:,:,2].view(-1),
                                                  reduction="none").view(batchsize,seqlen)
                    blurred_cog_loss = F.nll_loss(blurred_cog_probs.view(-1, self.cog_size),
                                                  targets[:,:,3].view(-1),
                                                  reduction="none").view(batchsize,seqlen)

                    lat_loss += self.blur_loss_w*blurred_lat_loss
                    lon_loss += self.blur_loss_w*blurred_lon_loss
                    sog_loss += self.blur_loss_w*blurred_sog_loss
                    cog_loss += self.blur_loss_w*blurred_cog_loss

                    lat_probs = blurred_lat_probs
                    lon_probs = blurred_lon_probs
                    sog_probs = blurred_sog_probs
                    cog_probs = blurred_cog_probs
                    

            loss_tuple = (lat_loss, lon_loss, sog_loss, cog_loss)
            loss = sum(loss_tuple)
        
            if masks is not None:
                loss = (loss*masks).sum(dim=1)/masks.sum(dim=1)
        
            loss = loss.mean()

            if (
                self.use_grid_residual
                and self.grid_residual_loss_w > 0
                and grid_residual is not None
            ):
                residual_targets = self.make_grid_residual_targets(targets_uniform, targets_real)
                residual_loss = F.smooth_l1_loss(
                    grid_residual,
                    residual_targets,
                    reduction="none",
                ).mean(dim=-1)
                if masks is not None:
                    residual_loss = (residual_loss * masks).sum() / masks.sum().clamp_min(1.0)
                else:
                    residual_loss = residual_loss.mean()
                loss = loss + self.grid_residual_loss_w * residual_loss

            if self.use_turn_intent_head and self.turn_intent_loss_w > 0:
                turn_logits = self.turn_head(fea)
                turn_labels, turn_mask = self.make_turn_labels(x, seqlen, masks=masks)
                turn_loss = F.cross_entropy(
                    turn_logits.reshape(-1, 5),
                    turn_labels.reshape(-1),
                    reduction="none",
                ).view(batchsize, seqlen)
                turn_weights = turn_mask.float()
                turn_loss = (turn_loss * turn_weights).sum() / turn_weights.sum().clamp_min(1.0)
                loss = loss + self.turn_intent_loss_w * turn_loss
        
        if return_loss_tuple:
            if return_residual:
                return logits, loss, loss_tuple, grid_residual
            return logits, loss, loss_tuple
        else:
            if return_residual:
                return logits, loss, grid_residual
            return logits, loss
        

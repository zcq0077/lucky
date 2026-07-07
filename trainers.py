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

"""Boilerplate for training a neural network.

References:
    https://github.com/karpathy/minGPT
"""

import os
import math
import logging

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader
from torch.nn import functional as F
import utils

logger = logging.getLogger(__name__)


def _dynamic_qwen_complex_mask(seqs, prior, turn_threshold, branch_threshold, entropy_threshold):
    """Return which rollout samples are currently in complex map cells."""
    batchsize = seqs.size(0)
    if prior is None or "features" not in prior:
        return torch.zeros(batchsize, dtype=torch.bool, device=seqs.device)

    features = np.asarray(prior["features"])
    if features.ndim != 3 or features.shape[-1] < 4:
        return torch.zeros(batchsize, dtype=torch.bool, device=seqs.device)

    h, w, _ = features.shape
    points = seqs[:, -1, :2].detach().cpu().numpy()
    lat_idx = np.clip((points[:, 0] * h).astype(np.int64), 0, h - 1)
    lon_idx = np.clip((points[:, 1] * w).astype(np.int64), 0, w - 1)
    cell_features = features[lat_idx, lon_idx]

    complex_np = (
        (cell_features[:, 1] >= turn_threshold)
        | (cell_features[:, 2] >= branch_threshold)
        | (cell_features[:, 3] >= entropy_threshold)
    )
    return torch.as_tensor(complex_np, dtype=torch.bool, device=seqs.device)


def _dynamic_qwen_rollout_gate_mask(
        seqs,
        prior,
        config,
        initial_len,
        turn_threshold,
        branch_threshold,
        entropy_threshold,
        gate_min_pred_points,
        gate_min_complex_points,
        gate_min_turn_points,
        gate_turn_angle_deg):
    """Decide which rollout samples are complex enough to deserve Qwen refresh."""
    batchsize = seqs.size(0)
    initial_len = int(initial_len)
    pred_len = max(0, seqs.size(1) - initial_len)
    if pred_len < int(gate_min_pred_points):
        return torch.zeros(batchsize, dtype=torch.bool, device=seqs.device)

    seq_np = seqs.detach().cpu().numpy()
    complex_count = np.zeros((batchsize,), dtype=np.int64)
    if prior is not None and "features" in prior:
        features = np.asarray(prior["features"])
        if features.ndim == 3 and features.shape[-1] >= 4 and pred_len > 0:
            h, w, _ = features.shape
            points = seq_np[:, initial_len:, :2]
            lat_idx = np.clip((points[..., 0] * h).astype(np.int64), 0, h - 1)
            lon_idx = np.clip((points[..., 1] * w).astype(np.int64), 0, w - 1)
            cell_features = features[lat_idx, lon_idx]
            complex_cells = (
                (cell_features[..., 1] >= turn_threshold)
                | (cell_features[..., 2] >= branch_threshold)
                | (cell_features[..., 3] >= entropy_threshold)
            )
            complex_count = complex_cells.sum(axis=1)

    turn_count = np.zeros((batchsize,), dtype=np.int64)
    start = max(0, initial_len - 1)
    traj = seq_np[:, start:, :2]
    if traj.shape[1] >= 3:
        if all(hasattr(config, name) for name in ("lat_min", "lat_max", "lon_min", "lon_max")):
            import map_prior

            lat = config.lat_min + traj[..., 0] * (config.lat_max - config.lat_min)
            lon = config.lon_min + traj[..., 1] * (config.lon_max - config.lon_min)
            headings = map_prior.bearing_deg(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
            diffs = np.abs(map_prior.circular_signed_diff_deg(headings[:, 1:], headings[:, :-1]))
        else:
            delta = np.diff(traj, axis=1)
            headings = (np.degrees(np.arctan2(delta[..., 1], delta[..., 0])) + 360.0) % 360.0
            diffs = np.abs((headings[:, 1:] - headings[:, :-1] + 180.0) % 360.0 - 180.0)
        turn_count = (diffs >= float(gate_turn_angle_deg)).sum(axis=1)

    gate_np = (
        (complex_count >= int(gate_min_complex_points))
        | (turn_count >= int(gate_min_turn_points))
    )
    return torch.as_tensor(gate_np, dtype=torch.bool, device=seqs.device)


def _dynamic_qwen_latest_turn_mask(
        seqs,
        config,
        initial_len,
        gate_min_pred_points,
        event_turn_angle_deg,
        event_window):
    """Return samples whose latest predicted segment contains a sharp turn."""
    batchsize = seqs.size(0)
    initial_len = int(initial_len)
    pred_len = max(0, seqs.size(1) - initial_len)
    if pred_len < int(gate_min_pred_points):
        return torch.zeros(batchsize, dtype=torch.bool, device=seqs.device)

    seq_np = seqs.detach().cpu().numpy()
    start = max(0, initial_len - 1)
    traj = seq_np[:, start:, :2]
    if traj.shape[1] < 3:
        return torch.zeros(batchsize, dtype=torch.bool, device=seqs.device)

    if all(hasattr(config, name) for name in ("lat_min", "lat_max", "lon_min", "lon_max")):
        import map_prior

        lat = config.lat_min + traj[..., 0] * (config.lat_max - config.lat_min)
        lon = config.lon_min + traj[..., 1] * (config.lon_max - config.lon_min)
        headings = map_prior.bearing_deg(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
        diffs = np.abs(map_prior.circular_signed_diff_deg(headings[:, 1:], headings[:, :-1]))
    else:
        delta = np.diff(traj, axis=1)
        headings = (np.degrees(np.arctan2(delta[..., 1], delta[..., 0])) + 360.0) % 360.0
        diffs = np.abs((headings[:, 1:] - headings[:, :-1] + 180.0) % 360.0 - 180.0)

    if diffs.shape[1] == 0:
        return torch.zeros(batchsize, dtype=torch.bool, device=seqs.device)

    window = max(1, int(event_window))
    recent_diffs = diffs[:, -window:]
    event_np = (recent_diffs >= float(event_turn_angle_deg)).any(axis=1)
    return torch.as_tensor(event_np, dtype=torch.bool, device=seqs.device)


def _maybe_update_dynamic_qwen(
        model,
        seqs,
        pred_step,
        updates_done,
        qwen_vec,
        qwen_mask,
        qwen_dynamic_encoder,
        qwen_dynamic_config,
        qwen_dynamic_prior,
        qwen_dynamic_update_interval,
        qwen_dynamic_min_pred_step,
        qwen_dynamic_max_updates,
        qwen_dynamic_turn_threshold,
        qwen_dynamic_branch_threshold,
        qwen_dynamic_entropy_threshold,
        qwen_dynamic_batch_size,
        qwen_dynamic_max_length,
        qwen_dynamic_show_progress,
        qwen_dynamic_complexity_gate,
        qwen_dynamic_gate_interval_only,
        qwen_dynamic_initial_len,
        qwen_dynamic_gate_min_pred_points,
        qwen_dynamic_gate_min_complex_points,
        qwen_dynamic_gate_min_turn_points,
        qwen_dynamic_gate_turn_angle_deg,
        qwen_dynamic_total_steps,
        qwen_dynamic_skip_last_steps,
        qwen_dynamic_event_trigger,
        qwen_dynamic_event_turn_angle_deg,
        qwen_dynamic_event_window):
    """Optionally refresh Qwen vectors from the current generated rollout."""
    if qwen_dynamic_encoder is None:
        return qwen_vec, qwen_mask, updates_done
    if not getattr(model, "use_qwen_semantic_encoder", False):
        return qwen_vec, qwen_mask, updates_done
    if qwen_dynamic_config is None:
        return qwen_vec, qwen_mask, updates_done
    if pred_step < qwen_dynamic_min_pred_step:
        return qwen_vec, qwen_mask, updates_done
    if (
        qwen_dynamic_total_steps > 0
        and qwen_dynamic_skip_last_steps > 0
        and (qwen_dynamic_total_steps - pred_step) < qwen_dynamic_skip_last_steps
    ):
        return qwen_vec, qwen_mask, updates_done

    batchsize = seqs.size(0)
    if not torch.is_tensor(updates_done):
        updates_done = torch.full(
            (batchsize,),
            int(updates_done),
            dtype=torch.long,
            device=seqs.device,
        )
    else:
        updates_done = updates_done.to(device=seqs.device, dtype=torch.long).view(-1)
    if qwen_dynamic_max_updates >= 0 and bool((updates_done >= qwen_dynamic_max_updates).all().item()):
        return qwen_vec, qwen_mask, updates_done

    interval_trigger = (
        qwen_dynamic_update_interval > 0
        and pred_step % qwen_dynamic_update_interval == 0
    )
    current_complex_mask = _dynamic_qwen_complex_mask(
        seqs,
        qwen_dynamic_prior,
        qwen_dynamic_turn_threshold,
        qwen_dynamic_branch_threshold,
        qwen_dynamic_entropy_threshold,
    )
    if bool(qwen_dynamic_complexity_gate):
        if bool(qwen_dynamic_gate_interval_only) and not interval_trigger:
            return qwen_vec, qwen_mask, updates_done
        rollout_gate_mask = _dynamic_qwen_rollout_gate_mask(
            seqs,
            qwen_dynamic_prior,
            qwen_dynamic_config,
            qwen_dynamic_initial_len,
            qwen_dynamic_turn_threshold,
            qwen_dynamic_branch_threshold,
            qwen_dynamic_entropy_threshold,
            qwen_dynamic_gate_min_pred_points,
            qwen_dynamic_gate_min_complex_points,
            qwen_dynamic_gate_min_turn_points,
            qwen_dynamic_gate_turn_angle_deg,
        )
        if bool(qwen_dynamic_event_trigger):
            latest_turn_mask = _dynamic_qwen_latest_turn_mask(
                seqs,
                qwen_dynamic_config,
                qwen_dynamic_initial_len,
                qwen_dynamic_gate_min_pred_points,
                qwen_dynamic_event_turn_angle_deg,
                qwen_dynamic_event_window,
            )
            event_mask = latest_turn_mask | current_complex_mask
            interval_mask = rollout_gate_mask if interval_trigger else torch.zeros_like(rollout_gate_mask)
            update_mask = (event_mask & rollout_gate_mask) | interval_mask
        else:
            update_mask = rollout_gate_mask if interval_trigger else (current_complex_mask & rollout_gate_mask)
    elif interval_trigger:
        update_mask = torch.ones(batchsize, dtype=torch.bool, device=seqs.device)
    else:
        update_mask = current_complex_mask
    if qwen_dynamic_max_updates >= 0:
        update_mask = update_mask & (updates_done < qwen_dynamic_max_updates)
    if not bool(update_mask.any().item()):
        return qwen_vec, qwen_mask, updates_done

    update_idxs = torch.nonzero(update_mask, as_tuple=False).view(-1)
    seqs_for_qwen = seqs[update_idxs].detach().cpu().numpy()
    qwen_np = qwen_dynamic_encoder.encode_sequences(
        seqs_for_qwen,
        qwen_dynamic_config,
        prior=qwen_dynamic_prior,
        batch_size=max(1, int(qwen_dynamic_batch_size)),
        max_length=int(qwen_dynamic_max_length),
        show_progress=bool(qwen_dynamic_show_progress),
        desc=f"Qwen dynamic step {pred_step} (n={int(update_idxs.numel())}/{batchsize})",
    )
    qwen_new = torch.as_tensor(qwen_np, dtype=torch.float32, device=seqs.device)

    if qwen_vec is None:
        qwen_vec = torch.zeros(
            batchsize,
            qwen_new.size(-1),
            dtype=qwen_new.dtype,
            device=seqs.device,
        )
    else:
        qwen_vec = qwen_vec.to(device=seqs.device).clone()
    qwen_vec[update_idxs] = qwen_new

    if qwen_mask is None:
        qwen_mask = torch.zeros(batchsize, dtype=qwen_new.dtype, device=seqs.device)
    else:
        qwen_mask = qwen_mask.to(device=seqs.device, dtype=qwen_new.dtype).view(-1).clone()
    qwen_mask[update_idxs] = 1.0

    updates_done = updates_done.clone()
    updates_done[update_idxs] += 1
    return qwen_vec, qwen_mask, updates_done


@torch.no_grad()
def sample(model,
           seqs,
           steps,
           temperature=1.0,
           sample=False,
           sample_mode="pos_vicinity",
           r_vicinity=20,
           top_k=None,
           qwen_vec=None,
           qwen_mask=None,
           qwen_dynamic_encoder=None,
           qwen_dynamic_config=None,
           qwen_dynamic_prior=None,
           qwen_dynamic_update_interval=6,
           qwen_dynamic_min_pred_step=3,
           qwen_dynamic_max_updates=4,
           qwen_dynamic_turn_threshold=0.35,
           qwen_dynamic_branch_threshold=0.25,
           qwen_dynamic_entropy_threshold=0.75,
           qwen_dynamic_batch_size=1,
           qwen_dynamic_max_length=1024,
           qwen_dynamic_show_progress=False,
           qwen_dynamic_complexity_gate=False,
           qwen_dynamic_gate_interval_only=False,
           qwen_dynamic_gate_min_pred_points=4,
           qwen_dynamic_gate_min_complex_points=2,
           qwen_dynamic_gate_min_turn_points=2,
           qwen_dynamic_gate_turn_angle_deg=18.0,
           qwen_dynamic_skip_last_steps=0,
           qwen_dynamic_event_trigger=False,
           qwen_dynamic_event_turn_angle_deg=30.0,
           qwen_dynamic_event_window=1,
           return_refined=False):
    """
    Take a conditoning sequence of AIS observations seq and predict the next observation,
    feed the predictions back into the model each time. 
    """
    max_seqlen = model.get_max_seqlen()
    model.eval()
    dynamic_updates = torch.zeros(seqs.size(0), dtype=torch.long, device=seqs.device)
    dynamic_initial_len = seqs.size(1)
    refined_seqs = seqs.clone()
    for k in range(steps):
        seqs_cond = seqs if seqs.size(1) <= max_seqlen else seqs[:, -max_seqlen:]  # crop context if needed

        # logits.shape: (batch_size, seq_len, data_size)
        use_refined = bool(return_refined and getattr(model, "use_grid_residual", False))
        if use_refined:
            logits, _, grid_residual = model(
                seqs_cond,
                qwen_vec=qwen_vec,
                qwen_mask=qwen_mask,
                return_residual=True,
            )
        else:
            logits, _ = model(seqs_cond, qwen_vec=qwen_vec, qwen_mask=qwen_mask)
            grid_residual = None
        d2inf_pred = torch.zeros((logits.shape[0], 4)).to(seqs.device) + 0.5

        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature  # (batch_size, data_size)

        lat_logits, lon_logits, sog_logits, cog_logits = \
            torch.split(logits, (model.lat_size, model.lon_size, model.sog_size, model.cog_size), dim=-1)

        # optionally crop probabilities to only the top k options
        if sample_mode in ("pos_vicinity",):
            idxs, idxs_uniform = model.to_indexes(seqs_cond[:, -1:, :])
            lat_idxs, lon_idxs = idxs_uniform[:, 0, 0:1], idxs_uniform[:, 0, 1:2]
            lat_logits = utils.top_k_nearest_idx(lat_logits, lat_idxs, r_vicinity)
            lon_logits = utils.top_k_nearest_idx(lon_logits, lon_idxs, r_vicinity)

        if top_k is not None:
            lat_logits = utils.top_k_logits(lat_logits, top_k)
            lon_logits = utils.top_k_logits(lon_logits, top_k)
            sog_logits = utils.top_k_logits(sog_logits, top_k)
            cog_logits = utils.top_k_logits(cog_logits, top_k)

        # apply softmax to convert to probabilities
        lat_probs = F.softmax(lat_logits, dim=-1)
        lon_probs = F.softmax(lon_logits, dim=-1)
        sog_probs = F.softmax(sog_logits, dim=-1)
        cog_probs = F.softmax(cog_logits, dim=-1)

        # sample from the distribution or take the most likely
        if sample:
            lat_ix = torch.multinomial(lat_probs, num_samples=1)  # (batch_size, 1)
            lon_ix = torch.multinomial(lon_probs, num_samples=1)
            sog_ix = torch.multinomial(sog_probs, num_samples=1)
            cog_ix = torch.multinomial(cog_probs, num_samples=1)
        else:
            _, lat_ix = torch.topk(lat_probs, k=1, dim=-1)
            _, lon_ix = torch.topk(lon_probs, k=1, dim=-1)
            _, sog_ix = torch.topk(sog_probs, k=1, dim=-1)
            _, cog_ix = torch.topk(cog_probs, k=1, dim=-1)

        ix = torch.cat((lat_ix, lon_ix, sog_ix, cog_ix), dim=-1)
        # convert to x (range: [0,1))
        x_sample = (ix.float() + d2inf_pred) / model.att_sizes
        x_refined = x_sample.clone()
        if use_refined and grid_residual is not None:
            residual = grid_residual[:, -1, :]
            x_refined[:, 0] = torch.clamp(
                (lat_ix.view(-1).float() + 0.5 + residual[:, 0]) / float(model.lat_size),
                0.0,
                0.9999,
            )
            x_refined[:, 1] = torch.clamp(
                (lon_ix.view(-1).float() + 0.5 + residual[:, 1]) / float(model.lon_size),
                0.0,
                0.9999,
            )

        # append to the sequence and continue
        seqs = torch.cat((seqs, x_sample.unsqueeze(1)), dim=1)
        refined_seqs = torch.cat((refined_seqs, x_refined.unsqueeze(1)), dim=1)
        qwen_update_seqs = refined_seqs if use_refined else seqs
        qwen_vec, qwen_mask, dynamic_updates = _maybe_update_dynamic_qwen(
            model=model,
            seqs=qwen_update_seqs,
            pred_step=k + 1,
            updates_done=dynamic_updates,
            qwen_vec=qwen_vec,
            qwen_mask=qwen_mask,
            qwen_dynamic_encoder=qwen_dynamic_encoder,
            qwen_dynamic_config=qwen_dynamic_config,
            qwen_dynamic_prior=qwen_dynamic_prior,
            qwen_dynamic_update_interval=int(qwen_dynamic_update_interval),
            qwen_dynamic_min_pred_step=int(qwen_dynamic_min_pred_step),
            qwen_dynamic_max_updates=int(qwen_dynamic_max_updates),
            qwen_dynamic_turn_threshold=float(qwen_dynamic_turn_threshold),
            qwen_dynamic_branch_threshold=float(qwen_dynamic_branch_threshold),
            qwen_dynamic_entropy_threshold=float(qwen_dynamic_entropy_threshold),
            qwen_dynamic_batch_size=int(qwen_dynamic_batch_size),
            qwen_dynamic_max_length=int(qwen_dynamic_max_length),
            qwen_dynamic_show_progress=bool(qwen_dynamic_show_progress),
            qwen_dynamic_complexity_gate=bool(qwen_dynamic_complexity_gate),
            qwen_dynamic_gate_interval_only=bool(qwen_dynamic_gate_interval_only),
            qwen_dynamic_initial_len=dynamic_initial_len,
            qwen_dynamic_gate_min_pred_points=int(qwen_dynamic_gate_min_pred_points),
            qwen_dynamic_gate_min_complex_points=int(qwen_dynamic_gate_min_complex_points),
            qwen_dynamic_gate_min_turn_points=int(qwen_dynamic_gate_min_turn_points),
            qwen_dynamic_gate_turn_angle_deg=float(qwen_dynamic_gate_turn_angle_deg),
            qwen_dynamic_total_steps=int(steps),
            qwen_dynamic_skip_last_steps=int(qwen_dynamic_skip_last_steps),
            qwen_dynamic_event_trigger=bool(qwen_dynamic_event_trigger),
            qwen_dynamic_event_turn_angle_deg=float(qwen_dynamic_event_turn_angle_deg),
            qwen_dynamic_event_window=int(qwen_dynamic_event_window),
        )

    return refined_seqs if bool(return_refined) else seqs


def unpack_batch(batch):
    if len(batch) == 5:
        seqs, masks, seqlens, mmsis, time_starts = batch
        return seqs, masks, seqlens, mmsis, time_starts, None, None
    if len(batch) == 7:
        seqs, masks, seqlens, mmsis, time_starts, qwen_vecs, qwen_masks = batch
        return seqs, masks, seqlens, mmsis, time_starts, qwen_vecs, qwen_masks
    raise ValueError(f"Unexpected batch size: {len(batch)}")


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader
    early_stopping = False
    early_stop_patience = 10
    early_stop_min_delta = 0.0
    early_stop_min_epochs = 0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:

    def __init__(self, model, train_dataset, test_dataset, config, savedir=None, device=torch.device("cpu"), aisdls={},
                 INIT_SEQLEN=0):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        self.savedir = savedir

        self.device = device
        self.model = model.to(device)
        self.aisdls = aisdls
        self.INIT_SEQLEN = INIT_SEQLEN

    def save_checkpoint(self, best_epoch):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Best epoch: {best_epoch:03d}, saving model to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config, aisdls, INIT_SEQLEN, = self.model, self.config, self.aisdls, self.INIT_SEQLEN
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)
        tb_writer = getattr(config, "tb_writer", None)
        use_tb = getattr(config, "tb_log", False) and tb_writer is not None
        if model.mode in ("gridcont_gridsin", "gridcont_gridsigmoid", "gridcont2_gridsigmoid",):
            return_loss_tuple = True
        else:
            return_loss_tuple = False

        def run_epoch(split, epoch=0):
            is_train = split == 'Training'
            model.train(is_train)
            data = self.train_dataset if is_train else self.test_dataset
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=config.batch_size,
                                num_workers=config.num_workers)

            losses = []
            n_batches = len(loader)
            pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)
            d_loss, d_reg_loss, d_n = 0, 0, 0
            for it, batch in pbar:
                seqs, masks, seqlens, mmsis, time_starts, qwen_vecs, qwen_masks = unpack_batch(batch)

                # place data on the correct device
                seqs = seqs.to(self.device)
                masks = masks[:, 1:].to(self.device)
                if qwen_vecs is not None:
                    qwen_vecs = qwen_vecs.to(self.device)
                    qwen_masks = qwen_masks.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    if return_loss_tuple:
                        logits, loss, loss_tuple = model(seqs,
                                                         masks=masks,
                                                         with_targets=True,
                                                         return_loss_tuple=return_loss_tuple,
                                                         qwen_vec=qwen_vecs,
                                                         qwen_mask=qwen_masks)
                    else:
                        logits, loss = model(
                            seqs,
                            masks=masks,
                            with_targets=True,
                            qwen_vec=qwen_vecs,
                            qwen_mask=qwen_masks,
                        )
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                d_loss += loss.item() * seqs.shape[0]
                if return_loss_tuple:
                    reg_loss = loss_tuple[-1]
                    reg_loss = reg_loss.mean()
                    d_reg_loss += reg_loss.item() * seqs.shape[0]
                d_n += seqs.shape[0]
                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                                seqs >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(f"epoch {epoch + 1} iter {it}: loss {loss.item():.5f}. lr {lr:e}")

                    # tb logging
                    if use_tb:
                        tb_writer.add_scalar("loss",
                                             loss.item(),
                                             epoch * n_batches + it)
                        tb_writer.add_scalar("lr",
                                             lr,
                                             epoch * n_batches + it)

                        for name, params in model.head.named_parameters():
                            tb_writer.add_histogram(f"head.{name}", params, epoch * n_batches + it)
                            tb_writer.add_histogram(f"head.{name}.grad", params.grad, epoch * n_batches + it)
                        if model.mode in ("gridcont_real",):
                            for name, params in model.res_pred.named_parameters():
                                tb_writer.add_histogram(f"res_pred.{name}", params, epoch * n_batches + it)
                                tb_writer.add_histogram(f"res_pred.{name}.grad", params.grad, epoch * n_batches + it)

            if is_train:
                if return_loss_tuple:
                    logging.info(
                        f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, {d_reg_loss / d_n:.5f}, lr {lr:e}.")
                else:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}, lr {lr:e}.")
            else:
                if return_loss_tuple:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}.")
                else:
                    logging.info(f"{split}, epoch {epoch + 1}, loss {d_loss / d_n:.5f}.")

            if not is_train:
                test_loss = float(np.mean(losses))
                #                 logging.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float('inf')
        self.tokens = 0  # counter used for learning rate decay
        best_epoch = 0
        epochs_without_improve = 0
        early_stopping = getattr(config, "early_stopping", False)
        early_stop_patience = int(getattr(config, "early_stop_patience", 10))
        early_stop_min_delta = float(getattr(config, "early_stop_min_delta", 0.0))
        early_stop_min_epochs = int(getattr(config, "early_stop_min_epochs", 0))

        for epoch in range(config.max_epochs):

            run_epoch('Training', epoch=epoch)
            if self.test_dataset is not None:
                test_loss = run_epoch('Valid', epoch=epoch)

            # supports early stopping based on the validation loss, or just save always if no validation set is provided
            good_model = self.test_dataset is None or test_loss < best_loss - early_stop_min_delta
            if good_model:
                if self.test_dataset is not None:
                    best_loss = test_loss
                best_epoch = epoch
                epochs_without_improve = 0
                if self.config.ckpt_path is not None:
                    self.save_checkpoint(best_epoch + 1)
            elif self.test_dataset is not None:
                epochs_without_improve += 1
                logging.info(
                    f"Early stopping monitor: {epochs_without_improve}/"
                    f"{early_stop_patience} epochs without improvement. "
                    f"Best epoch {best_epoch + 1:03d}, best valid loss {best_loss:.5f}."
                )

            if (
                early_stopping
                and self.test_dataset is not None
                and epoch + 1 >= early_stop_min_epochs
                and epochs_without_improve >= early_stop_patience
            ):
                logging.info(
                    f"Early stopping at epoch {epoch + 1:03d}. "
                    f"Best epoch {best_epoch + 1:03d}, best valid loss {best_loss:.5f}."
                )
                break

            ## SAMPLE AND PLOT
            # ==========================================================================================
            # ==========================================================================================
            raw_model = model.module if hasattr(self.model, "module") else model
            seqs, masks, seqlens, mmsis, time_starts, qwen_vecs, qwen_masks = unpack_batch(next(iter(aisdls["test"])))
            n_plots = 7
            init_seqlen = INIT_SEQLEN
            seqs_init = seqs[:n_plots, :init_seqlen, :].to(self.device)
            if qwen_vecs is not None:
                qwen_vecs = qwen_vecs[:n_plots].to(self.device)
                qwen_masks = qwen_masks[:n_plots].to(self.device)
            preds = sample(raw_model,
                           seqs_init,
                           96 - init_seqlen,
                           temperature=1.0,
                           sample=True,
                           sample_mode=self.config.sample_mode,
                           r_vicinity=self.config.r_vicinity,
                           top_k=self.config.top_k,
                           qwen_vec=qwen_vecs,
                           qwen_mask=qwen_masks,
                           return_refined=getattr(self.config, "grid_residual_use_for_eval", False))

            img_path = os.path.join(self.savedir, f'epoch_{epoch + 1:03d}.jpg')
            plt.figure(figsize=(9, 6), dpi=150)
            cmap = plt.cm.get_cmap("jet")
            preds_np = preds.detach().cpu().numpy()
            inputs_np = seqs.detach().cpu().numpy()
            for idx in range(n_plots):
                c = cmap(float(idx) / (n_plots))
                try:
                    seqlen = seqlens[idx].item()
                except:
                    continue
                plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], color=c)
                plt.plot(inputs_np[idx][:init_seqlen, 1], inputs_np[idx][:init_seqlen, 0], "o", markersize=3, color=c)
                plt.plot(inputs_np[idx][:seqlen, 1], inputs_np[idx][:seqlen, 0], linestyle="-.", color=c)
                plt.plot(preds_np[idx][init_seqlen:, 1], preds_np[idx][init_seqlen:, 0], "x", markersize=4, color=c)
            plt.xlim([-0.05, 1.05])
            plt.ylim([-0.05, 1.05])
            plt.savefig(img_path, dpi=150)
            plt.close()

        # Final state
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        #         logging.info("saving %s", self.config.ckpt_path)
        logging.info(f"Last epoch: {epoch:03d}, saving model to {self.config.ckpt_path}")
        save_path = self.config.ckpt_path.replace("model.pt", f"model_{epoch + 1:03d}.pt")
        torch.save(raw_model.state_dict(), save_path)

# coding=utf-8
"""Lightweight student model distilled from multi-stage Qwen navigation vectors."""

import os

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def query_map_features_np(seqs, prior, max_points, map_channels):
    """Return per-point map features for padded normalized trajectories."""
    batch, length, _ = seqs.shape
    out = np.zeros((batch, length, map_channels), dtype=np.float32)
    if prior is None or "features" not in prior:
        return out

    features = np.asarray(prior["features"], dtype=np.float32)
    if features.ndim != 3:
        return out
    h, w, channels = features.shape
    use_channels = min(map_channels, channels)
    points = np.clip(seqs[..., :2], 0.0, 0.999999)
    lat_idx = np.clip((points[..., 0] * h).astype(np.int64), 0, h - 1)
    lon_idx = np.clip((points[..., 1] * w).astype(np.int64), 0, w - 1)
    out[..., :use_channels] = features[lat_idx, lon_idx, :use_channels]
    return out[:, -max_points:, :]


def pad_sequence_prefix(seq, max_points):
    """Keep the latest max_points and return padded sequence plus mask."""
    seq = np.asarray(seq[:, :4], dtype=np.float32)
    seq = np.clip(seq, 0.0, 0.999999)
    clipped = seq[-max_points:]
    out = np.zeros((max_points, 4), dtype=np.float32)
    mask = np.zeros((max_points,), dtype=np.float32)
    out[-len(clipped):] = clipped
    mask[-len(clipped):] = 1.0
    return out, mask


def turn_label_from_prefix(full_traj, stage_len, config, future_window=3):
    """Create a coarse turn label at a teacher stage from true future motion."""
    import map_prior

    traj = np.asarray(full_traj[:, :4], dtype=np.float64)
    stage_len = int(stage_len)
    if stage_len < 2 or stage_len >= len(traj):
        return 0, 0.0

    prev_idx = max(0, stage_len - 2)
    cur_idx = stage_len - 1
    fut_idx = min(len(traj) - 1, cur_idx + max(1, int(future_window)))
    if fut_idx <= cur_idx:
        return 0, 0.0

    lat = config.lat_min + traj[:, 0] * (config.lat_max - config.lat_min)
    lon = config.lon_min + traj[:, 1] * (config.lon_max - config.lon_min)
    in_bearing = map_prior.bearing_deg(lat[prev_idx], lon[prev_idx], lat[cur_idx], lon[cur_idx])
    out_bearing = map_prior.bearing_deg(lat[cur_idx], lon[cur_idx], lat[fut_idx], lon[fut_idx])
    delta = float(map_prior.circular_signed_diff_deg(out_bearing, in_bearing))

    straight = float(getattr(config, "turn_straight_threshold_deg", 10.0))
    sharp = float(getattr(config, "turn_sharp_threshold_deg", 35.0))
    label = 0
    if -sharp < delta < -straight:
        label = 1
    elif straight < delta < sharp:
        label = 2
    elif delta <= -sharp:
        label = 3
    elif delta >= sharp:
        label = 4
    return label, delta


def complexity_label_from_prefix(prefix, prior, config):
    """Use map prior at current position as a simple complex-route target."""
    if prior is None or "features" not in prior or len(prefix) == 0:
        return 0.0
    features = np.asarray(prior["features"], dtype=np.float32)
    if features.ndim != 3 or features.shape[-1] < 4:
        return 0.0
    h, w, _ = features.shape
    point = np.clip(np.asarray(prefix[-1, :2], dtype=np.float32), 0.0, 0.999999)
    lat_idx = int(np.clip(point[0] * h, 0, h - 1))
    lon_idx = int(np.clip(point[1] * w, 0, w - 1))
    cell = features[lat_idx, lon_idx]
    return float(
        (cell[1] >= float(getattr(config, "qwen_dynamic_turn_threshold", 0.55)))
        or (cell[2] >= float(getattr(config, "qwen_dynamic_branch_threshold", 0.45)))
        or (cell[3] >= float(getattr(config, "qwen_dynamic_entropy_threshold", 0.90)))
    )


class CaptainStudent(nn.Module):
    """Small sequence encoder that predicts Qwen-like navigation semantics."""

    def __init__(
            self,
            input_dim=8,
            hidden_dim=256,
            output_dim=2560,
            num_layers=1,
            dropout=0.1):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)

        self.point_encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )
        self.gru = nn.GRU(
            self.hidden_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )
        self.semantic_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.intent_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 5),
        )
        self.complexity_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, seqs, masks=None, map_features=None):
        if map_features is not None:
            x = torch.cat((seqs, map_features), dim=-1)
        else:
            x = seqs
        encoded = self.point_encoder(x)
        if masks is None:
            lengths = torch.full(
                (seqs.size(0),),
                seqs.size(1),
                dtype=torch.long,
                device=seqs.device,
            )
        else:
            lengths = masks.sum(dim=1).clamp_min(1).long()

        out, _ = self.gru(encoded)
        gather_idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, out.size(-1))
        final = out.gather(1, gather_idx).squeeze(1)
        return {
            "student_vec": self.semantic_head(final),
            "intent_logits": self.intent_head(final),
            "complexity_logits": self.complexity_head(final).squeeze(-1),
        }


def student_distillation_loss(
        outputs,
        teacher_vec,
        turn_label=None,
        complexity_label=None,
        mask=None,
        cos_w=1.0,
        mse_w=0.05,
        intent_w=0.2,
        complexity_w=0.05):
    student_vec = outputs["student_vec"]
    teacher_vec = teacher_vec.to(student_vec.device, dtype=student_vec.dtype)
    weights = torch.ones(student_vec.size(0), dtype=student_vec.dtype, device=student_vec.device)
    if mask is not None:
        weights = mask.to(student_vec.device, dtype=student_vec.dtype).view(-1)
    denom = weights.sum().clamp_min(1.0)

    cos = 1.0 - F.cosine_similarity(student_vec, teacher_vec, dim=-1)
    loss = float(cos_w) * (cos * weights).sum() / denom
    if mse_w > 0:
        mse = F.mse_loss(
            F.normalize(student_vec, dim=-1),
            F.normalize(teacher_vec, dim=-1),
            reduction="none",
        ).mean(dim=-1)
        loss = loss + float(mse_w) * (mse * weights).sum() / denom

    loss_parts = {"cos": float((cos * weights).sum().detach().cpu() / denom.detach().cpu())}
    if turn_label is not None and intent_w > 0:
        turn_label = turn_label.to(student_vec.device, dtype=torch.long)
        intent_loss = F.cross_entropy(outputs["intent_logits"], turn_label, reduction="none")
        intent_loss = (intent_loss * weights).sum() / denom
        loss = loss + float(intent_w) * intent_loss
        loss_parts["intent"] = float(intent_loss.detach().cpu())

    if complexity_label is not None and complexity_w > 0:
        complexity_label = complexity_label.to(student_vec.device, dtype=student_vec.dtype)
        complexity_loss = F.binary_cross_entropy_with_logits(
            outputs["complexity_logits"],
            complexity_label,
            reduction="none",
        )
        complexity_loss = (complexity_loss * weights).sum() / denom
        loss = loss + float(complexity_w) * complexity_loss
        loss_parts["complexity"] = float(complexity_loss.detach().cpu())

    return loss, loss_parts


def save_student_checkpoint(path, model, metadata):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "metadata": dict(metadata),
    }
    torch.save(payload, path)


def load_student_checkpoint(path, device="cpu"):
    payload = torch.load(path, map_location=device)
    meta = dict(payload.get("metadata", {}))
    model = CaptainStudent(
        input_dim=int(meta["input_dim"]),
        hidden_dim=int(meta["hidden_dim"]),
        output_dim=int(meta["output_dim"]),
        num_layers=int(meta.get("num_layers", 1)),
        dropout=float(meta.get("dropout", 0.1)),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, meta


class CaptainStudentDynamicEncoder:
    """Adapter with the same encode_sequences interface as QwenSemanticEncoder."""

    def __init__(self, checkpoint_path, config, prior=None, device=None):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Captain Student checkpoint not found: {checkpoint_path}\n"
                "Train it first with train_captain_student.py."
            )
        device = device or getattr(config, "device", torch.device("cpu"))
        self.device = torch.device(device)
        self.model, self.meta = load_student_checkpoint(checkpoint_path, self.device)
        self.max_points = int(self.meta.get("max_points", getattr(config, "captain_student_max_points", 42)))
        self.use_map_features = bool(self.meta.get("use_map_features", True))
        self.map_channels = int(self.meta.get("map_channels", getattr(config, "map_prior_channels", 4)))
        self.prior = prior

    @property
    def hidden_size(self):
        return int(self.meta["output_dim"])

    def encode_sequences(
            self,
            seqs,
            config,
            prior=None,
            batch_size=32,
            max_length=1024,
            show_progress=False,
            desc="Captain Student dynamic"):
        del max_length, show_progress, desc
        prior = prior if prior is not None else self.prior
        seqs = np.asarray(seqs, dtype=np.float32)
        padded, masks = [], []
        for seq in seqs:
            seq_pad, seq_mask = pad_sequence_prefix(seq, self.max_points)
            padded.append(seq_pad)
            masks.append(seq_mask)
        padded = np.stack(padded, axis=0)
        masks = np.stack(masks, axis=0)
        if self.use_map_features:
            map_features = query_map_features_np(padded, prior, self.max_points, self.map_channels)
        else:
            map_features = None

        vectors = []
        bs = max(1, int(batch_size))
        with torch.no_grad():
            for start in range(0, len(padded), bs):
                seq_batch = torch.as_tensor(padded[start:start + bs], dtype=torch.float32, device=self.device)
                mask_batch = torch.as_tensor(masks[start:start + bs], dtype=torch.float32, device=self.device)
                map_batch = None
                if map_features is not None:
                    map_batch = torch.as_tensor(
                        map_features[start:start + bs],
                        dtype=torch.float32,
                        device=self.device,
                    )
                out = self.model(seq_batch, mask_batch, map_batch)
                vectors.append(out["student_vec"].detach().float().cpu().numpy())
        return np.concatenate(vectors, axis=0).astype(np.float32)

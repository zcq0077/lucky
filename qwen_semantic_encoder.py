# coding=utf-8
"""Qwen semantic embeddings for AIS trajectory conditioning.

This module does not ask Qwen to generate JSON labels or future coordinates.
It builds a history-only prompt, runs a frozen local Qwen model, and returns
hidden-state vectors that can condition TrAISformer.
"""

import hashlib
import json
import math

import numpy as np
from tqdm import tqdm

import map_prior


def _safe_float(x, default=0.0):
    try:
        value = float(x)
        if math.isfinite(value):
            return value
    except (TypeError, ValueError):
        pass
    return default


def _normalized_to_degrees(traj, config):
    lat = config.lat_min + traj[..., 0] * (config.lat_max - config.lat_min)
    lon = config.lon_min + traj[..., 1] * (config.lon_max - config.lon_min)
    return lat, lon


def _bearing_series(traj, config):
    if len(traj) < 2:
        return np.zeros((0,), dtype=np.float64)
    lat, lon = _normalized_to_degrees(traj[:, :2], config)
    dpos = np.abs(np.diff(traj[:, :2], axis=0)).sum(axis=1)
    headings = map_prior.bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return headings[dpos > 1e-10]


def _mean_signed_turn(headings):
    if len(headings) < 2:
        return 0.0
    diffs = map_prior.circular_signed_diff_deg(headings[1:], headings[:-1])
    return float(np.mean(diffs))


def _grid_index(point, prior):
    h, w = prior["features"].shape[:2]
    lat_idx = int(np.clip(point[0] * h, 0, h - 1))
    lon_idx = int(np.clip(point[1] * w, 0, w - 1))
    return lat_idx, lon_idx


def query_map_context(point, prior):
    lat_idx, lon_idx = _grid_index(point, prior)
    features = prior["features"][lat_idx, lon_idx]
    names = [
        "density_norm",
        "turn_rate",
        "branch_score",
        "direction_entropy",
        "obstacle_proximity",
        "obstacle_turn_score",
    ]
    feature_dict = {
        name: round(float(features[i]), 4)
        for i, name in enumerate(names[: len(features)])
    }

    top_dirs = []
    direction_probs = prior["direction_probs"][lat_idx, lon_idx]
    if direction_probs.sum() > 0:
        n_bins = len(direction_probs)
        top_idxs = np.argsort(direction_probs)[-3:][::-1]
        for idx in top_idxs:
            prob = float(direction_probs[idx])
            if prob <= 0:
                continue
            top_dirs.append(
                {
                    "bearing_deg": round(((idx + 0.5) * 360.0 / n_bins) % 360.0, 1),
                    "prob": round(prob, 4),
                }
            )

    return {
        "grid_index": [lat_idx, lon_idx],
        "features": feature_dict,
        "dominant_directions": top_dirs,
    }


def build_history_summary(vessel, traj_idx, config, prior):
    """Build a compact history-only summary for Qwen semantic encoding."""
    max_points = int(getattr(config, "qwen_prompt_max_points", config.init_seqlen))
    hist_len = min(max_points, config.init_seqlen, len(vessel["traj"]))
    traj = np.asarray(vessel["traj"][:hist_len, :4], dtype=np.float64)
    if len(traj) < 2:
        raise ValueError("Trajectory is too short for Qwen semantic encoding.")

    current = traj[-1]
    lat, lon = _normalized_to_degrees(traj[:, :2], config)
    headings = _bearing_series(traj, config)
    recent_headings = headings[-5:] if len(headings) > 0 else headings
    recent_steps = np.diff(traj[-6:, :2], axis=0) if len(traj) >= 6 else np.diff(traj[:, :2], axis=0)
    mean_step_norm = float(np.linalg.norm(recent_steps, axis=1).mean()) if len(recent_steps) else 0.0
    speed_values = traj[-6:, 2] if len(traj) >= 6 else traj[:, 2]
    speed_trend = float(speed_values[-1] - speed_values[0]) if len(speed_values) > 1 else 0.0

    return {
        "trajectory_id": int(traj_idx),
        "mmsi": int(vessel["mmsi"]),
        "history_points": int(len(traj)),
        "current_position": {
            "lat": round(float(lat[-1]), 6),
            "lon": round(float(lon[-1]), 6),
        },
        "current_heading_deg": round(float(headings[-1]) if len(headings) else 0.0, 2),
        "recent_mean_heading_deg": round(float(np.mean(recent_headings)) if len(recent_headings) else 0.0, 2),
        "recent_mean_turn_deg": round(_mean_signed_turn(recent_headings), 2),
        "recent_mean_step_norm": round(mean_step_norm, 6),
        "speed_trend_norm": round(speed_trend, 5),
        "current_sog_norm": round(_safe_float(current[2]), 4),
        "current_cog_deg_from_data": round(_safe_float(current[3]) * 360.0, 2),
        "map_context": query_map_context(current[:2], prior),
    }


def build_sequence_summary(seq, config, prior=None, trajectory_id=-1, mmsi=0):
    """Build a compact summary from the current rollout sequence.

    This is used by dynamic inference. The sequence may contain the real
    history followed by points that the model has already generated.
    """
    max_points = int(getattr(config, "qwen_prompt_max_points", config.init_seqlen))
    traj = np.asarray(seq[-max_points:, :4], dtype=np.float64)
    if len(traj) < 2:
        raise ValueError("Sequence is too short for Qwen semantic encoding.")

    traj[:, :2] = np.clip(traj[:, :2], 0.0, 0.999999)
    current = traj[-1]
    lat, lon = _normalized_to_degrees(traj[:, :2], config)
    headings = _bearing_series(traj, config)
    recent_headings = headings[-5:] if len(headings) > 0 else headings
    recent_steps = np.diff(traj[-6:, :2], axis=0) if len(traj) >= 6 else np.diff(traj[:, :2], axis=0)
    mean_step_norm = float(np.linalg.norm(recent_steps, axis=1).mean()) if len(recent_steps) else 0.0
    speed_values = traj[-6:, 2] if len(traj) >= 6 else traj[:, 2]
    speed_trend = float(speed_values[-1] - speed_values[0]) if len(speed_values) > 1 else 0.0
    map_context = query_map_context(current[:2], prior) if prior is not None else None

    return {
        "trajectory_id": int(trajectory_id),
        "mmsi": int(mmsi),
        "history_points": int(len(traj)),
        "source": "dynamic_rollout",
        "current_position": {
            "lat": round(float(lat[-1]), 6),
            "lon": round(float(lon[-1]), 6),
        },
        "current_heading_deg": round(float(headings[-1]) if len(headings) else 0.0, 2),
        "recent_mean_heading_deg": round(float(np.mean(recent_headings)) if len(recent_headings) else 0.0, 2),
        "recent_mean_turn_deg": round(_mean_signed_turn(recent_headings), 2),
        "recent_mean_step_norm": round(mean_step_norm, 6),
        "speed_trend_norm": round(speed_trend, 5),
        "current_sog_norm": round(_safe_float(current[2]), 4),
        "current_cog_deg_from_data": round(_safe_float(current[3]) * 360.0, 2),
        "map_context": map_context,
    }


def build_semantic_prompt(summary):
    """Build a deterministic prompt. It must contain history only."""
    payload = {
        "role": "semantic_encoder",
        "instruction": (
            "Encode the navigation situation for AIS trajectory prediction. "
            "Use only historical motion and map statistics. Do not predict "
            "future coordinates and do not assume unseen vessels or charts."
        ),
        "navigation_priors": [
            "Ships usually preserve course and speed continuity.",
            "High turn_rate suggests a turning area.",
            "High branch_score or direction_entropy suggests route choice complexity.",
            "Dense routes and dominant directions may indicate common traffic flow.",
        ],
        "history_summary": summary,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


class QwenSemanticEncoder:
    """Frozen local Qwen wrapper that returns pooled hidden states."""

    def __init__(self, model_path, device="auto", freeze=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device_arg = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )

        load_kwargs = {
            "trust_remote_code": True,
            "local_files_only": True,
            "torch_dtype": "auto",
        }
        if device == "auto":
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map="auto",
                    **load_kwargs,
                )
            except Exception:
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
                fallback = "cuda" if torch.cuda.is_available() else "cpu"
                self.model.to(fallback)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            self.model.to(device)

        self.model.eval()
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    @property
    def hidden_size(self):
        return int(getattr(self.model.config, "hidden_size"))

    def encode_prompts(self, prompts, batch_size=4, max_length=1024, desc="Qwen encoding", show_progress=True):
        vectors = []
        ranges = range(0, len(prompts), batch_size)
        for start in tqdm(
            ranges,
            total=math.ceil(len(prompts) / batch_size),
            desc=desc,
            disable=not show_progress,
            leave=False,
            dynamic_ncols=True,
            position=1,
        ):
            chunk = prompts[start:start + batch_size]
            device = next(self.model.parameters()).device
            inputs = self.tokenizer(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            ).to(device)
            with self.torch.no_grad():
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
            hidden = outputs.hidden_states[-1]
            last_idx = inputs["attention_mask"].sum(dim=1) - 1
            pooled = hidden[self.torch.arange(hidden.size(0), device=hidden.device), last_idx]
            vectors.append(pooled.detach().float().cpu().numpy())
        return np.concatenate(vectors, axis=0).astype(np.float32)

    def encode_sequences(
            self,
            seqs,
            config,
            prior=None,
            batch_size=1,
            max_length=1024,
            show_progress=False,
            desc="Qwen dynamic"):
        prompts = [
            build_semantic_prompt(build_sequence_summary(seq, config, prior, trajectory_id=i))
            for i, seq in enumerate(seqs)
        ]
        return self.encode_prompts(
            prompts,
            batch_size=batch_size,
            max_length=max_length,
            desc=desc,
            show_progress=show_progress,
        )

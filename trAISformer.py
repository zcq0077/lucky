#!/usr/bin/env python
# coding: utf-8
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

"""Pytorch implementation of TrAISformer---A generative transformer for
AIS trajectory prediction

https://arxiv.org/abs/2109.03958

"""
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import os
import sys
import pickle
import json
import csv
from datetime import datetime
from tqdm import tqdm
import math
import logging
import pdb

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

import models, trainers, datasets, utils
from config_trAISformer import Config

cf = Config()
TB_LOG = cf.tb_log
if TB_LOG:
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter()
    cf.tb_writer = tb
else:
    cf.tb_writer = None

# make deterministic
utils.set_seed(42)
torch.pi = torch.acos(torch.zeros(1)).item() * 2


def _json_safe_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe_value(v) for k, v in value.items()}
    return str(value)


def _public_config_dict(config):
    items = {}
    for name in dir(config):
        if name.startswith("_") or name in ("tb_writer",):
            continue
        value = getattr(config, name)
        if callable(value):
            continue
        items[name] = _json_safe_value(value)
    return items


def save_experiment_snapshot(config, metrics=None):
    """Save a readable config/metric snapshot for experiment comparison."""
    os.makedirs(config.savedir, exist_ok=True)
    config_items = _public_config_dict(config)
    payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "savedir": config.savedir,
        "filename": getattr(config, "filename", ""),
        "metrics": metrics or {},
        "config": config_items,
    }

    json_path = os.path.join(config.savedir, "config_snapshot.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    key_names = [
        "retrain",
        "use_map_prior",
        "map_emb_w",
        "use_turn_intent_head",
        "turn_intent_loss_w",
        "use_qwen_semantic_encoder",
        "qwen_model_path",
        "qwen_cache_dir",
        "qwen_hidden_size",
        "qwen_emb_w",
        "qwen_bias_w",
        "qwen_emb_pdrop",
        "qwen_dynamic_inference",
        "qwen_dynamic_inference_pred_samples",
        "qwen_dynamic_visualize",
        "qwen_dynamic_visualize_pred_samples",
        "eval_n_samples",
        "use_grid_residual",
        "grid_residual_loss_w",
        "grid_residual_max_abs_cell",
        "grid_residual_use_for_eval",
        "learning_rate",
        "batch_size",
        "max_epochs",
        "early_stopping",
        "early_stop_patience",
    ]
    md_path = os.path.join(config.savedir, "config_snapshot.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Experiment Snapshot\n\n")
        f.write(f"- created_at: {payload['created_at']}\n")
        f.write(f"- savedir: `{config.savedir}`\n")
        f.write(f"- filename: `{getattr(config, 'filename', '')}`\n\n")
        f.write("## Key Settings\n\n")
        for name in key_names:
            if name in config_items:
                f.write(f"- {name}: `{config_items[name]}`\n")
        if metrics:
            f.write("\n## Test Metrics\n\n")
            for name, value in metrics.items():
                f.write(f"- {name}: `{value}`\n")
        f.write("\n## Full Config\n\n")
        for name in sorted(config_items):
            f.write(f"- {name}: `{config_items[name]}`\n")


def append_experiment_index(config, metrics):
    """Append one row to results/_experiment_index.csv for quick comparison."""
    os.makedirs("./results", exist_ok=True)
    index_path = os.path.join("./results", "_experiment_index.csv")
    fieldnames = [
        "created_at",
        "filename",
        "savedir",
        "err_1h_km",
        "err_2h_km",
        "err_3h_km",
        "err_4h_km",
        "mean_err_km",
        "map_emb_w",
        "turn_intent_loss_w",
        "qwen_emb_w",
        "qwen_bias_w",
        "qwen_emb_pdrop",
        "qwen_hidden_size",
        "qwen_dynamic_inference",
        "qwen_dynamic_inference_pred_samples",
        "qwen_dynamic_visualize",
        "qwen_dynamic_visualize_pred_samples",
        "eval_n_samples",
        "use_grid_residual",
        "grid_residual_loss_w",
        "grid_residual_max_abs_cell",
        "grid_residual_use_for_eval",
        "learning_rate",
        "batch_size",
        "max_epochs",
        "early_stopping",
    ]
    row = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": getattr(config, "filename", ""),
        "savedir": config.savedir,
    }
    for name in fieldnames:
        if name in row:
            continue
        if name in metrics:
            row[name] = metrics[name]
        else:
            row[name] = _json_safe_value(getattr(config, name, ""))

    write_header = not os.path.exists(index_path)
    with open(index_path, "a", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"======= Updated experiment index: {index_path}")


def _to_lon_lat_degrees(seqs, model):
    """Convert normalized [lat, lon, ...] arrays to lon/lat degrees."""
    lat = model.lat_min + seqs[..., 0] * model.lat_range
    lon = model.lon_min + seqs[..., 1] * model.lon_range
    return lon, lat


def _mean_future_haversine_km(pred, truth, model, init_seqlen, true_len):
    """Mean future distance between one sampled prediction and truth."""
    start = init_seqlen
    end = min(true_len, pred.shape[0], truth.shape[0])
    if end <= start:
        return float("inf")

    pred_lat = model.lat_min + pred[start:end, 0] * model.lat_range
    pred_lon = model.lon_min + pred[start:end, 1] * model.lon_range
    true_lat = model.lat_min + truth[start:end, 0] * model.lat_range
    true_lon = model.lon_min + truth[start:end, 1] * model.lon_range

    pred_lat = np.radians(pred_lat)
    pred_lon = np.radians(pred_lon)
    true_lat = np.radians(true_lat)
    true_lon = np.radians(true_lon)

    dlat = pred_lat - true_lat
    dlon = pred_lon - true_lon
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(true_lat) * np.cos(pred_lat) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(np.mean(6371.0 * c))


def load_qwen_cache(cf, phase, n_items):
    cache_path = os.path.join(
        cf.qwen_cache_dir,
        f"{cf.dataset_name}_{phase}_qwen_vecs.npz",
    )
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Qwen semantic cache not found: {cache_path}\n"
            "Generate it first, for example:\n"
            "python generate_qwen_embedding_cache.py --phase all --device cuda --overwrite"
        )
    cache = np.load(cache_path, allow_pickle=True)
    vectors = cache["qwen_vecs"].astype(np.float32)
    masks = cache["qwen_mask"].astype(np.float32)
    if len(vectors) != n_items:
        raise ValueError(
            f"Qwen cache length mismatch for {phase}: cache has {len(vectors)}, "
            f"dataset has {n_items}. Regenerate the cache after data/config changes."
        )
    print(f"======= Loaded Qwen semantic cache for {phase}: {cache_path} {vectors.shape}")
    return vectors, masks


def load_qwen_dynamic_encoder(cf, prior=None):
    """Load dynamic semantic encoder only when dynamic inference is explicitly requested."""
    wants_dynamic = (
        bool(getattr(cf, "qwen_dynamic_inference", False))
        or bool(getattr(cf, "qwen_dynamic_visualize", False))
    )
    if not wants_dynamic:
        return None
    if not getattr(cf, "use_qwen_semantic_encoder", False):
        print("======= Qwen dynamic requested but use_qwen_semantic_encoder=False; skip dynamic Qwen.")
        return None
    if prior is None:
        print("======= Qwen dynamic will run without map context; interval updates only.")

    if bool(getattr(cf, "use_captain_student_dynamic", False)):
        from captain_student import CaptainStudentDynamicEncoder

        ckpt_path = getattr(cf, "captain_student_ckpt_path", "./captain_student/captain_student.pt")
        print(f"======= Loading Captain Student dynamic encoder from: {ckpt_path} ({cf.device})")
        encoder = CaptainStudentDynamicEncoder(
            ckpt_path,
            config=cf,
            prior=prior,
            device=cf.device,
        )
        expected_hidden = int(getattr(cf, "qwen_hidden_size", encoder.hidden_size))
        if encoder.hidden_size != expected_hidden:
            raise ValueError(
                f"Captain Student output_dim={encoder.hidden_size}, "
                f"but qwen_hidden_size={expected_hidden}. "
                "Regenerate teacher/static cache and retrain the student with the same Qwen model."
            )
        return encoder

    from qwen_semantic_encoder import QwenSemanticEncoder

    device = getattr(cf, "qwen_dynamic_device", "cuda")
    print(f"======= Loading Qwen dynamic encoder from: {cf.qwen_model_path} ({device})")
    return QwenSemanticEncoder(
        cf.qwen_model_path,
        device=device,
        freeze=True,
    )


def _qwen_dynamic_value(cf, scope, name, default):
    scoped_name = f"qwen_dynamic_{scope}_{name}" if scope else None
    if scoped_name and hasattr(cf, scoped_name):
        return getattr(cf, scoped_name)
    return getattr(cf, f"qwen_dynamic_{name}", default)


def qwen_dynamic_kwargs(cf, encoder, prior, enabled, scope=None):
    if not enabled or encoder is None:
        return {}
    scope = scope or ""
    return {
        "qwen_dynamic_encoder": encoder,
        "qwen_dynamic_config": cf,
        "qwen_dynamic_prior": prior,
        "qwen_dynamic_update_interval": _qwen_dynamic_value(cf, scope, "update_interval", 6),
        "qwen_dynamic_min_pred_step": _qwen_dynamic_value(cf, scope, "min_pred_step", 3),
        "qwen_dynamic_max_updates": _qwen_dynamic_value(cf, scope, "max_updates", 4),
        "qwen_dynamic_turn_threshold": _qwen_dynamic_value(cf, scope, "turn_threshold", 0.35),
        "qwen_dynamic_branch_threshold": _qwen_dynamic_value(cf, scope, "branch_threshold", 0.25),
        "qwen_dynamic_entropy_threshold": _qwen_dynamic_value(cf, scope, "entropy_threshold", 0.75),
        "qwen_dynamic_batch_size": _qwen_dynamic_value(cf, scope, "batch_size", 1),
        "qwen_dynamic_max_length": _qwen_dynamic_value(cf, scope, "max_length", 1024),
        "qwen_dynamic_show_progress": _qwen_dynamic_value(cf, scope, "show_progress", False),
        "qwen_dynamic_complexity_gate": _qwen_dynamic_value(cf, scope, "complexity_gate", False),
        "qwen_dynamic_gate_interval_only": _qwen_dynamic_value(cf, scope, "gate_interval_only", False),
        "qwen_dynamic_gate_min_pred_points": _qwen_dynamic_value(cf, scope, "gate_min_pred_points", 4),
        "qwen_dynamic_gate_min_complex_points": _qwen_dynamic_value(cf, scope, "gate_min_complex_points", 2),
        "qwen_dynamic_gate_min_turn_points": _qwen_dynamic_value(cf, scope, "gate_min_turn_points", 2),
        "qwen_dynamic_gate_turn_angle_deg": _qwen_dynamic_value(cf, scope, "gate_turn_angle_deg", 18.0),
        "qwen_dynamic_skip_last_steps": _qwen_dynamic_value(cf, scope, "skip_last_steps", 0),
        "qwen_dynamic_event_trigger": _qwen_dynamic_value(cf, scope, "event_trigger", False),
        "qwen_dynamic_event_turn_angle_deg": _qwen_dynamic_value(cf, scope, "event_turn_angle_deg", 30.0),
        "qwen_dynamic_event_window": _qwen_dynamic_value(cf, scope, "event_window", 1),
    }


def plot_test_trajectory_examples(
        model,
        test_dataset,
        cf,
        init_seqlen,
        max_seqlen,
        savedir,
        qwen_dynamic_encoder=None,
        qwen_dynamic_prior=None):
    """Plot history, ground truth, and sampled predictions for test trajectories."""
    if not getattr(cf, "test_visualize", True):
        return

    n_examples = int(getattr(cf, "test_visualize_n", 10))
    n_pred_samples = int(getattr(cf, "test_visualize_pred_samples", getattr(cf, "n_samples", 16)))
    dynamic_visualize = bool(getattr(cf, "qwen_dynamic_visualize", False)) and qwen_dynamic_encoder is not None
    if dynamic_visualize:
        n_pred_samples = int(getattr(cf, "qwen_dynamic_visualize_pred_samples", n_pred_samples))
    rng = np.random.default_rng(int(getattr(cf, "test_visualize_seed", 42)))

    seqlens_all = np.array([
        min(len(v["traj"]), test_dataset.max_seqlen)
        for v in test_dataset.l_data
    ])
    eligible = np.flatnonzero(seqlens_all > init_seqlen + 1)
    if eligible.size == 0:
        print("======= No valid test trajectories for visualization.")
        return

    n_examples = min(n_examples, eligible.size)
    selected = rng.choice(eligible, size=n_examples, replace=False)

    seqs, masks, seqlens, mmsis = [], [], [], []
    qwen_vecs, qwen_masks = [], []
    for idx in selected:
        item = test_dataset[int(idx)]
        seq, mask, seqlen, mmsi, _, qwen_vec, qwen_mask = trainers.unpack_batch(item)
        seqs.append(seq)
        masks.append(mask)
        seqlens.append(int(seqlen.item()))
        mmsis.append(int(mmsi.item()))
        if qwen_vec is not None:
            qwen_vecs.append(qwen_vec)
            qwen_masks.append(qwen_mask)

    seqs = torch.stack(seqs, dim=0)
    seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
    if qwen_vecs:
        qwen_vecs = torch.stack(qwen_vecs, dim=0).to(cf.device)
        qwen_masks = torch.stack(qwen_masks, dim=0).to(cf.device)
    else:
        qwen_vecs = None
        qwen_masks = None
    steps = max_seqlen - init_seqlen

    pred_samples = []
    model.eval()
    dynamic_kwargs = qwen_dynamic_kwargs(
        cf,
        qwen_dynamic_encoder,
        qwen_dynamic_prior,
        dynamic_visualize,
        scope="visualize",
    )
    if dynamic_visualize:
        print(
            "======= Dynamic Qwen visualization: "
            f"{n_examples} trajectories x {n_pred_samples} samples. "
            "This is inference-time trajectory generation, not image post-processing."
        )
    with torch.no_grad():
        sample_iter = range(max(1, n_pred_samples))
        if dynamic_visualize:
            sample_iter = tqdm(sample_iter, desc="Dynamic Qwen visual samples")
        for _ in sample_iter:
            preds = trainers.sample(
                model,
                seqs_init,
                steps,
                temperature=1.0,
                sample=True,
                sample_mode=cf.sample_mode,
                r_vicinity=cf.r_vicinity,
                top_k=cf.top_k,
                qwen_vec=qwen_vecs,
                qwen_mask=qwen_masks,
                return_refined=getattr(cf, "grid_residual_use_for_eval", False),
                **dynamic_kwargs,
            )
            pred_samples.append(preds.detach().cpu().numpy())

    seqs_np = seqs.detach().cpu().numpy()
    ncols = 5
    nrows = int(math.ceil(n_examples / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), dpi=160)
    axes = np.asarray(axes).reshape(-1)

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= n_examples:
            ax.axis("off")
            continue

        true_len = min(seqlens[ax_idx], max_seqlen)
        hist_end = min(init_seqlen, true_len)
        seq = seqs_np[ax_idx]
        lon_true, lat_true = _to_lon_lat_degrees(seq, model)

        ax.plot(
            lon_true[:hist_end],
            lat_true[:hist_end],
            color="tab:blue",
            linewidth=2.0,
            marker="o",
            markersize=2.5,
            label="History" if ax_idx == 0 else None,
        )
        if true_len > hist_end:
            truth_start = max(hist_end - 1, 0)
            ax.plot(
                lon_true[truth_start:true_len],
                lat_true[truth_start:true_len],
                color="tab:green",
                linewidth=2.0,
                marker=".",
                markersize=3.0,
                label="Ground truth" if ax_idx == 0 else None,
            )

        sample_errors = [
            _mean_future_haversine_km(
                pred_np[ax_idx],
                seq,
                model,
                init_seqlen,
                true_len,
            )
            for pred_np in pred_samples
        ]
        best_sample_idx = int(np.argmin(sample_errors))

        pred_start = max(init_seqlen - 1, 0)
        for sample_idx, pred_np in enumerate(pred_samples):
            pred = pred_np[ax_idx]
            lon_pred, lat_pred = _to_lon_lat_degrees(pred, model)
            is_best = sample_idx == best_sample_idx
            ax.plot(
                lon_pred[pred_start:max_seqlen],
                lat_pred[pred_start:max_seqlen],
                color="tab:red",
                linewidth=2.4 if is_best else 1.0,
                alpha=0.95 if is_best else 0.18,
                label="Best prediction" if ax_idx == 0 and is_best else None,
            )

        ax.set_title(f"Test #{int(selected[ax_idx])}  MMSI {mmsis[ax_idx]}", fontsize=9)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, alpha=0.25)
        ax.set_aspect("equal", adjustable="datalim")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 1))

    out_path = os.path.join(savedir, "test_trajectory_examples.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"======= Saved test trajectory visualization: {out_path}")


if __name__ == "__main__":

    device = cf.device
    init_seqlen = cf.init_seqlen

    ## Logging
    # ===============================
    if not os.path.isdir(cf.savedir):
        os.makedirs(cf.savedir)
        print('======= Create directory to store trained models: ' + cf.savedir)
    else:
        print('======= Directory to store trained models: ' + cf.savedir)
    utils.new_log(cf.savedir, "log")
    save_experiment_snapshot(cf)
    print(f"======= Saved experiment config snapshot: {os.path.join(cf.savedir, 'config_snapshot.md')}")

    ## Data
    # ===============================
    moving_threshold = 0.05
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
    qwen_cache = {}
    if getattr(cf, "use_qwen_semantic_encoder", False) and not getattr(cf, "qwen_use_cache", True):
        raise NotImplementedError(
            "True online Qwen encoding is not enabled in the lightweight path. "
            "Use qwen_use_cache=True and run generate_qwen_embedding_cache.py first."
        )
    for phase, filename in zip(("train", "valid", "test"), l_pkl_filenames):
        datapath = os.path.join(cf.datadir, filename)
        print(f"Loading {datapath}...")
        with open(datapath, "rb") as f:
            l_pred_errors = pickle.load(f)
        for V in l_pred_errors:
            try:
                moving_idx = np.where(V["traj"][:, 2] > moving_threshold)[0][0]
            except:
                moving_idx = len(V["traj"]) - 1  # This track will be removed
            V["traj"] = V["traj"][moving_idx:, :]
        Data[phase] = [x for x in l_pred_errors if not np.isnan(x["traj"]).any() and len(x["traj"]) > cf.min_seqlen]
        print(len(l_pred_errors), len(Data[phase]))
        print(f"Length: {len(Data[phase])}")
        if getattr(cf, "use_qwen_semantic_encoder", False):
            qwen_vectors, qwen_masks = load_qwen_cache(cf, phase, len(Data[phase]))
            qwen_cache[phase] = (qwen_vectors, qwen_masks)
            cf.qwen_hidden_size = int(qwen_vectors.shape[1])
        else:
            qwen_cache[phase] = (None, None)
        print("Creating pytorch dataset...")
        # Latter in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
        # max_seqlen = cf.max_seqlen + 1.
        qwen_vectors, qwen_masks = qwen_cache[phase]
        if cf.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = datasets.AISDataset_grad(Data[phase],
                                                          max_seqlen=cf.max_seqlen + 1,
                                                          qwen_vectors=qwen_vectors,
                                                          qwen_mask=qwen_masks,
                                                          device=cf.device)
        else:
            aisdatasets[phase] = datasets.AISDataset(Data[phase],
                                                     max_seqlen=cf.max_seqlen + 1,
                                                     qwen_vectors=qwen_vectors,
                                                     qwen_mask=qwen_masks,
                                                     device=cf.device)
        if phase == "test":
            shuffle = False
        else:
            shuffle = True
        aisdls[phase] = DataLoader(aisdatasets[phase],
                                   batch_size=cf.batch_size,
                                   shuffle=shuffle)
    cf.final_tokens = 2 * len(aisdatasets["train"]) * cf.max_seqlen

    ## Model
    # ===============================
    model = models.TrAISformer(cf, partition_model=None)
    prior = None
    if getattr(cf, "use_map_prior", False):
        import map_prior

        prior = map_prior.build_map_prior(Data["train"], cf, savedir=cf.savedir)
        print(
            "======= Built map prior: "
            f"{prior['stats']['n_tracks']} tracks, "
            f"{prior['stats']['n_points']} points, "
            f"{prior['stats']['valid_cells']} valid cells, "
            f"{prior['stats']['feature_channels']} channels"
        )
        model.register_map_prior(prior["features"], prior["direction_probs"])

    ## Trainer
    # ===============================
    trainer = trainers.Trainer(
        model, aisdatasets["train"], aisdatasets["valid"], cf, savedir=cf.savedir, device=cf.device, aisdls=aisdls, INIT_SEQLEN=init_seqlen)

    ## Training
    # ===============================
    if cf.retrain:
        trainer.train()

    ## Evaluation
    # ===============================
    # Load the best model
    model.load_state_dict(torch.load(cf.ckpt_path))

    v_ranges = torch.tensor([2, 3, 0, 0]).to(cf.device)
    v_roi_min = torch.tensor([model.lat_min, -7, 0, 0]).to(cf.device)
    max_seqlen = init_seqlen + 6 * 4

    model.eval()
    qwen_dynamic_encoder = None
    if bool(getattr(cf, "qwen_dynamic_inference", False)):
        qwen_dynamic_encoder = load_qwen_dynamic_encoder(cf, prior)
    dynamic_eval_enabled = bool(getattr(cf, "qwen_dynamic_inference", False))
    dynamic_eval_kwargs = qwen_dynamic_kwargs(
        cf,
        qwen_dynamic_encoder,
        prior,
        dynamic_eval_enabled,
        scope="inference",
    )
    eval_static_samples = 0
    eval_dynamic_samples = int(getattr(cf, "n_samples", 16))
    if dynamic_eval_enabled:
        eval_static_samples = int(getattr(cf, "qwen_dynamic_inference_static_samples", 0))
        eval_dynamic_samples = int(getattr(cf, "qwen_dynamic_inference_pred_samples", eval_dynamic_samples))
        eval_n_samples = eval_static_samples + eval_dynamic_samples
        if eval_static_samples > 0:
            sample_text = f"mixed best-of-{eval_n_samples} ({eval_static_samples} static + {eval_dynamic_samples} dynamic-capable)"
        else:
            sample_text = f"conditional dynamic best-of-{eval_n_samples}"
        print(
            "======= Dynamic Qwen evaluation uses "
            f"{sample_text}, "
            f"max_updates={getattr(cf, 'qwen_dynamic_inference_max_updates', getattr(cf, 'qwen_dynamic_max_updates', 4))}, "
            f"update_interval={getattr(cf, 'qwen_dynamic_inference_update_interval', getattr(cf, 'qwen_dynamic_update_interval', 6))}, "
            f"qwen_batch={getattr(cf, 'qwen_dynamic_inference_batch_size', getattr(cf, 'qwen_dynamic_batch_size', 1))}, "
            f"complexity_gate={getattr(cf, 'qwen_dynamic_inference_complexity_gate', getattr(cf, 'qwen_dynamic_complexity_gate', False))}, "
            f"interval_only={getattr(cf, 'qwen_dynamic_inference_gate_interval_only', getattr(cf, 'qwen_dynamic_gate_interval_only', False))}, "
            f"event_trigger={getattr(cf, 'qwen_dynamic_inference_event_trigger', getattr(cf, 'qwen_dynamic_event_trigger', False))}, "
            f"skip_last={getattr(cf, 'qwen_dynamic_inference_skip_last_steps', getattr(cf, 'qwen_dynamic_skip_last_steps', 0))}, "
            f"student_dynamic={getattr(cf, 'use_captain_student_dynamic', False)}, "
            f"turn_gate={getattr(cf, 'qwen_dynamic_gate_min_turn_points', 2)}x"
            f"{getattr(cf, 'qwen_dynamic_gate_turn_angle_deg', 18.0)}deg."
        )
    else:
        eval_n_samples = eval_dynamic_samples
    cf.eval_n_samples = eval_n_samples
    l_min_errors, l_mean_errors, l_masks = [], [], []
    pbar = tqdm(
        enumerate(aisdls["test"]),
        total=len(aisdls["test"]),
        desc="Test batches",
        dynamic_ncols=True,
        leave=True,
        position=0,
    )
    with torch.no_grad():
        for it, batch in pbar:
            seqs, masks, seqlens, mmsis, time_starts, qwen_vecs, qwen_masks = trainers.unpack_batch(batch)
            seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
            masks = masks[:, :max_seqlen].to(cf.device)
            if qwen_vecs is not None:
                qwen_vecs = qwen_vecs.to(cf.device)
                qwen_masks = qwen_masks.to(cf.device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - cf.init_seqlen, eval_n_samples)).to(cf.device)
            sample_col = 0
            for _ in range(eval_static_samples):
                pbar.set_postfix_str(f"sample {sample_col + 1}/{eval_n_samples}")
                preds = trainers.sample(model,
                                        seqs_init,
                                        max_seqlen - init_seqlen,
                                        temperature=1.0,
                                        sample=True,
                                        sample_mode=cf.sample_mode,
                                        r_vicinity=cf.r_vicinity,
                                        top_k=cf.top_k,
                                        qwen_vec=qwen_vecs,
                                        qwen_mask=qwen_masks,
                                        return_refined=getattr(cf, "grid_residual_use_for_eval", False))
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, sample_col] = d[:, cf.init_seqlen:]
                sample_col += 1
            for _ in range(eval_dynamic_samples):
                pbar.set_postfix_str(f"sample {sample_col + 1}/{eval_n_samples}")
                preds = trainers.sample(model,
                                        seqs_init,
                                        max_seqlen - init_seqlen,
                                        temperature=1.0,
                                        sample=True,
                                        sample_mode=cf.sample_mode,
                                        r_vicinity=cf.r_vicinity,
                                        top_k=cf.top_k,
                                        qwen_vec=qwen_vecs,
                                        qwen_mask=qwen_masks,
                                        return_refined=getattr(cf, "grid_residual_use_for_eval", False),
                                        **dynamic_eval_kwargs)
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, sample_col] = d[:, cf.init_seqlen:]
                sample_col += 1
            # Accumulation through batches
            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, cf.init_seqlen:])

    l_min = [x.values for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()
    def _error_at(index):
        index = min(int(index), len(pred_errors) - 1)
        return round(float(pred_errors[index]), 6)

    test_metrics = {
        "err_1h_km": _error_at(6),
        "err_2h_km": _error_at(12),
        "err_3h_km": _error_at(18),
        "err_4h_km": _error_at(23),
        "mean_err_km": round(float(np.nanmean(pred_errors)), 6),
        "eval_n_samples": int(eval_n_samples),
    }
    save_experiment_snapshot(cf, metrics=test_metrics)
    append_experiment_index(cf, test_metrics)

    ## Plot
    # ===============================
    plt.figure(figsize=(9, 6), dpi=150)
    v_times = np.arange(len(pred_errors)) / 6
    plt.plot(v_times, pred_errors)

    timestep = 6
    plt.plot(1, pred_errors[timestep], "o")
    plt.plot([1, 1], [0, pred_errors[timestep]], "r")
    plt.plot([0, 1], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(1.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 12
    plt.plot(2, pred_errors[timestep], "o")
    plt.plot([2, 2], [0, pred_errors[timestep]], "r")
    plt.plot([0, 2], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(2.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)

    timestep = 18
    plt.plot(3, pred_errors[timestep], "o")
    plt.plot([3, 3], [0, pred_errors[timestep]], "r")
    plt.plot([0, 3], [pred_errors[timestep], pred_errors[timestep]], "r")
    plt.text(3.12, pred_errors[timestep] - 0.5, "{:.4f}".format(pred_errors[timestep]), fontsize=10)
    plt.xlabel("Time (hours)")
    plt.ylabel("Prediction errors (km)")
    plt.xlim([0, 12])
    plt.ylim([0, 20])
    # plt.ylim([0,pred_errors.max()+0.5])
    plt.savefig(cf.savedir + "prediction_error.png")
    plt.close()

    ## Test trajectory visualization
    # ===============================
    if qwen_dynamic_encoder is None and bool(getattr(cf, "qwen_dynamic_visualize", False)):
        qwen_dynamic_encoder = load_qwen_dynamic_encoder(cf, prior)
    plot_test_trajectory_examples(
        model,
        aisdatasets["test"],
        cf,
        init_seqlen,
        max_seqlen,
        cf.savedir,
        qwen_dynamic_encoder=qwen_dynamic_encoder,
        qwen_dynamic_prior=prior,
    )

    # Yeah, done!!!

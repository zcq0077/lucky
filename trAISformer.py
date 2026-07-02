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


def plot_test_trajectory_examples(model, test_dataset, cf, init_seqlen, max_seqlen, savedir):
    """Plot history, ground truth, and sampled predictions for test trajectories."""
    if not getattr(cf, "test_visualize", True):
        return

    n_examples = int(getattr(cf, "test_visualize_n", 10))
    n_pred_samples = int(getattr(cf, "test_visualize_pred_samples", 4))
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
    for idx in selected:
        seq, mask, seqlen, mmsi, _ = test_dataset[int(idx)]
        seqs.append(seq)
        masks.append(mask)
        seqlens.append(int(seqlen.item()))
        mmsis.append(int(mmsi.item()))

    seqs = torch.stack(seqs, dim=0)
    seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
    steps = max_seqlen - init_seqlen

    pred_samples = []
    model.eval()
    with torch.no_grad():
        for _ in range(max(1, n_pred_samples)):
            preds = trainers.sample(
                model,
                seqs_init,
                steps,
                temperature=1.0,
                sample=True,
                sample_mode=cf.sample_mode,
                r_vicinity=cf.r_vicinity,
                top_k=cf.top_k,
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

    ## Data
    # ===============================
    moving_threshold = 0.05
    l_pkl_filenames = [cf.trainset_name, cf.validset_name, cf.testset_name]
    Data, aisdatasets, aisdls = {}, {}, {}
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
        print("Creating pytorch dataset...")
        # Latter in this scipt, we will use inputs = x[:-1], targets = x[1:], hence
        # max_seqlen = cf.max_seqlen + 1.
        if cf.mode in ("pos_grad", "grad"):
            aisdatasets[phase] = datasets.AISDataset_grad(Data[phase],
                                                          max_seqlen=cf.max_seqlen + 1,
                                                          device=cf.device)
        else:
            aisdatasets[phase] = datasets.AISDataset(Data[phase],
                                                     max_seqlen=cf.max_seqlen + 1,
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
    l_min_errors, l_mean_errors, l_masks = [], [], []
    pbar = tqdm(enumerate(aisdls["test"]), total=len(aisdls["test"]))
    with torch.no_grad():
        for it, (seqs, masks, seqlens, mmsis, time_starts) in pbar:
            seqs_init = seqs[:, :init_seqlen, :].to(cf.device)
            masks = masks[:, :max_seqlen].to(cf.device)
            batchsize = seqs.shape[0]
            error_ens = torch.zeros((batchsize, max_seqlen - cf.init_seqlen, cf.n_samples)).to(cf.device)
            for i_sample in range(cf.n_samples):
                preds = trainers.sample(model,
                                        seqs_init,
                                        max_seqlen - init_seqlen,
                                        temperature=1.0,
                                        sample=True,
                                        sample_mode=cf.sample_mode,
                                        r_vicinity=cf.r_vicinity,
                                        top_k=cf.top_k)
                inputs = seqs[:, :max_seqlen, :].to(cf.device)
                input_coords = (inputs * v_ranges + v_roi_min) * torch.pi / 180
                pred_coords = (preds * v_ranges + v_roi_min) * torch.pi / 180
                d = utils.haversine(input_coords, pred_coords) * masks
                error_ens[:, :, i_sample] = d[:, cf.init_seqlen:]
            # Accumulation through batches
            l_min_errors.append(error_ens.min(dim=-1))
            l_mean_errors.append(error_ens.mean(dim=-1))
            l_masks.append(masks[:, cf.init_seqlen:])

    l_min = [x.values for x in l_min_errors]
    m_masks = torch.cat(l_masks, dim=0)
    min_errors = torch.cat(l_min, dim=0) * m_masks
    pred_errors = min_errors.sum(dim=0) / m_masks.sum(dim=0)
    pred_errors = pred_errors.detach().cpu().numpy()

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
    plot_test_trajectory_examples(
        model,
        aisdatasets["test"],
        cf,
        init_seqlen,
        max_seqlen,
        cf.savedir,
    )

    # Yeah, done!!!

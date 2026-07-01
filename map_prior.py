# coding=utf-8
"""Build map-conditioned route priors for TrAISformer.

The prior is estimated from training trajectories only. It summarizes local
traffic density, turning frequency, multi-modal outgoing directions, and
direction entropy on a normalized latitude/longitude grid.
"""

import math
import os

import numpy as np


def normalized_to_degrees(x, config):
    """Convert normalized [lat, lon] coordinates to geographic degrees."""
    lat = config.lat_min + x[..., 0] * (config.lat_max - config.lat_min)
    lon = config.lon_min + x[..., 1] * (config.lon_max - config.lon_min)
    return lat, lon


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point 1 to point 2 in degrees, clockwise from north."""
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def circular_abs_diff_deg(a, b):
    """Smallest absolute angular difference in degrees."""
    return np.abs((a - b + 180.0) % 360.0 - 180.0)


def circular_signed_diff_deg(a, b):
    """Signed angular difference a - b in degrees within [-180, 180)."""
    return (a - b + 180.0) % 360.0 - 180.0


def _grid_index(x, h, w):
    lat_idx = np.clip((x[..., 0] * h).astype(np.int64), 0, h - 1)
    lon_idx = np.clip((x[..., 1] * w).astype(np.int64), 0, w - 1)
    return lat_idx, lon_idx


def _spatial_smooth(arr, n_iter=1):
    """Cheap 3x3 smoothing over the first two axes."""
    out = arr.astype(np.float64, copy=False)
    for _ in range(max(0, int(n_iter))):
        pad_width = [(1, 1), (1, 1)] + [(0, 0)] * (out.ndim - 2)
        padded = np.pad(out, pad_width, mode="edge")
        out = (
            padded[:-2, :-2]
            + padded[:-2, 1:-1]
            + padded[:-2, 2:]
            + padded[1:-1, :-2]
            + padded[1:-1, 1:-1]
            + padded[1:-1, 2:]
            + padded[2:, :-2]
            + padded[2:, 1:-1]
            + padded[2:, 2:]
        ) / 9.0
    return out


def circular_smooth_hist(hist, n_iter=1):
    """Smooth direction histograms along the circular direction axis."""
    out = hist.astype(np.float64, copy=False)
    for _ in range(max(0, int(n_iter))):
        out = (np.roll(out, 1, axis=-1) + out + np.roll(out, -1, axis=-1)) / 3.0
    return out


def _circular_bin_distance(a, b, n_bins):
    d = abs(a - b)
    return min(d, n_bins - d)


def count_direction_peaks(prob, threshold, min_separation_bins):
    """Count separated local maxima in a circular direction distribution."""
    n_bins = prob.shape[-1]
    peak_mask = (
        (prob >= threshold)
        & (prob > np.roll(prob, 1))
        & (prob >= np.roll(prob, -1))
    )
    candidates = np.flatnonzero(peak_mask)
    if candidates.size == 0:
        return 0

    candidates = sorted(candidates.tolist(), key=lambda idx: prob[idx], reverse=True)
    selected = []
    for idx in candidates:
        if all(_circular_bin_distance(idx, prev, n_bins) >= min_separation_bins for prev in selected):
            selected.append(idx)
    return len(selected)


def _dilate_mask(mask, radius):
    """Binary dilation with a square neighborhood."""
    radius = int(radius)
    if radius <= 0:
        return mask.astype(np.float32)
    out = mask.astype(bool)
    for _ in range(radius):
        padded = np.pad(out, ((1, 1), (1, 1)), mode="edge")
        out = (
            padded[:-2, :-2]
            | padded[:-2, 1:-1]
            | padded[:-2, 2:]
            | padded[1:-1, :-2]
            | padded[1:-1, 1:-1]
            | padded[1:-1, 2:]
            | padded[2:, :-2]
            | padded[2:, 1:-1]
            | padded[2:, 2:]
        )
    return out.astype(np.float32)


def build_map_prior(l_data, config, savedir=None):
    """Build a multi-channel map prior from a list of AIS trajectory dicts."""
    h = int(getattr(config, "map_prior_lat_size", 120))
    w = int(getattr(config, "map_prior_lon_size", 120))
    k = int(getattr(config, "map_direction_bins", 36))
    min_count = int(getattr(config, "map_min_count", 10))
    smooth_iter = int(getattr(config, "map_smooth_iter", 1))
    direction_smooth_iter = int(getattr(config, "map_direction_smooth_iter", 1))
    turn_threshold = float(getattr(config, "map_turn_angle_threshold_deg", 25.0))
    peak_threshold = float(getattr(config, "map_branch_peak_prob_threshold", 0.12))
    min_sep_deg = float(getattr(config, "map_branch_min_separation_deg", 35.0))
    use_obstacle = bool(getattr(config, "map_use_obstacle_features", False))
    max_points = int(getattr(config, "max_seqlen", 120)) + 1

    visit_count = np.zeros((h, w), dtype=np.int64)
    turn_count = np.zeros((h, w), dtype=np.int64)
    turn_base_count = np.zeros((h, w), dtype=np.int64)
    direction_hist = np.zeros((h, w, k), dtype=np.int64)

    n_tracks = 0
    n_points = 0
    n_turn_events = 0
    eps = 1e-10

    for vessel in l_data:
        traj = np.asarray(vessel["traj"][:max_points, :4], dtype=np.float64)
        if traj.shape[0] < 2:
            continue
        traj[:, :2] = np.clip(traj[:, :2], 0.0, 0.999999)
        n_tracks += 1
        n_points += traj.shape[0]

        lat_idx, lon_idx = _grid_index(traj[:, :2], h, w)
        np.add.at(visit_count, (lat_idx, lon_idx), 1)

        lat, lon = normalized_to_degrees(traj[:, :2], config)
        dpos = np.abs(np.diff(traj[:, :2], axis=0)).sum(axis=1)
        seg_valid = dpos > eps

        if seg_valid.any():
            headings = bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])
            bins = np.floor(headings / (360.0 / k)).astype(np.int64) % k
            valid_idx = np.flatnonzero(seg_valid)
            np.add.at(
                direction_hist,
                (lat_idx[valid_idx], lon_idx[valid_idx], bins[valid_idx]),
                1,
            )

        if traj.shape[0] >= 3:
            center_valid = seg_valid[:-1] & seg_valid[1:]
            if center_valid.any():
                headings = bearing_deg(lat[:-1], lon[:-1], lat[1:], lon[1:])
                turn_angles = circular_abs_diff_deg(headings[1:], headings[:-1])
                center_idx = np.flatnonzero(center_valid) + 1
                np.add.at(turn_base_count, (lat_idx[center_idx], lon_idx[center_idx]), 1)
                turn_mask = center_valid & (turn_angles >= turn_threshold)
                if turn_mask.any():
                    turn_idx = np.flatnonzero(turn_mask) + 1
                    np.add.at(turn_count, (lat_idx[turn_idx], lon_idx[turn_idx]), 1)
                    n_turn_events += int(turn_mask.sum())

    density_smooth = _spatial_smooth(visit_count, smooth_iter)
    density_log = np.log1p(density_smooth)
    density_norm = density_log / max(float(density_log.max()), 1.0)

    turn_num = _spatial_smooth(turn_count, smooth_iter)
    turn_den = _spatial_smooth(turn_base_count, smooth_iter)
    turn_rate = np.divide(turn_num, turn_den, out=np.zeros_like(turn_num), where=turn_den > 0)

    hist_smooth = _spatial_smooth(direction_hist, smooth_iter)
    hist_smooth = circular_smooth_hist(hist_smooth, direction_smooth_iter)
    hist_sum = hist_smooth.sum(axis=-1, keepdims=True)
    direction_probs = np.divide(
        hist_smooth,
        hist_sum,
        out=np.zeros_like(hist_smooth, dtype=np.float64),
        where=hist_sum > 0,
    )

    direction_entropy = -np.sum(direction_probs * np.log(direction_probs + 1e-12), axis=-1)
    direction_entropy /= math.log(k)

    min_sep_bins = max(1, int(math.ceil(min_sep_deg / (360.0 / k))))
    branch_peak_count = np.zeros((h, w), dtype=np.int64)
    valid_count = visit_count >= min_count
    for i, j in np.argwhere(valid_count):
        branch_peak_count[i, j] = count_direction_peaks(
            direction_probs[i, j], peak_threshold, min_sep_bins
        )
    branch_score = np.clip((branch_peak_count - 1) / 2.0, 0.0, 1.0)

    turn_rate[~valid_count] = 0.0
    branch_score[~valid_count] = 0.0
    direction_entropy[~valid_count] = 0.0
    direction_probs[~valid_count] = 0.0

    feature_list = [density_norm, turn_rate, branch_score, direction_entropy]
    obstacle_proximity = np.zeros((h, w), dtype=np.float64)
    obstacle_turn_score = np.zeros((h, w), dtype=np.float64)
    if use_obstacle:
        positive_density = density_norm[visit_count > 0]
        if positive_density.size > 0:
            q = float(getattr(config, "map_obstacle_density_quantile", 0.10))
            density_threshold = np.quantile(positive_density, q)
            low_density = density_norm <= density_threshold
            radius = int(getattr(config, "map_obstacle_proximity_radius", 5))
            obstacle_proximity = _dilate_mask(low_density, radius)
            obstacle_turn_score = obstacle_proximity * turn_rate
        feature_list.extend([obstacle_proximity, obstacle_turn_score])

    features = np.stack(feature_list, axis=-1).astype(np.float32)
    direction_probs = direction_probs.astype(np.float32)

    prior = {
        "features": features,
        "direction_probs": direction_probs,
        "visit_count": visit_count,
        "turn_rate": turn_rate.astype(np.float32),
        "branch_score": branch_score.astype(np.float32),
        "direction_entropy": direction_entropy.astype(np.float32),
        "density_norm": density_norm.astype(np.float32),
        "obstacle_proximity": obstacle_proximity.astype(np.float32),
        "obstacle_turn_score": obstacle_turn_score.astype(np.float32),
        "turn_count": turn_count,
        "turn_base_count": turn_base_count,
        "branch_peak_count": branch_peak_count,
        "direction_hist": direction_hist,
        "stats": {
            "n_tracks": n_tracks,
            "n_points": n_points,
            "n_turn_events": n_turn_events,
            "valid_cells": int(valid_count.sum()),
            "nonzero_cells": int((visit_count > 0).sum()),
            "feature_channels": int(features.shape[-1]),
        },
    }

    if savedir is not None:
        os.makedirs(savedir, exist_ok=True)
        np.savez_compressed(
            os.path.join(savedir, "map_prior.npz"),
            features=prior["features"],
            direction_probs=prior["direction_probs"],
            visit_count=prior["visit_count"],
            turn_rate=prior["turn_rate"],
            branch_score=prior["branch_score"],
            direction_entropy=prior["direction_entropy"],
            density_norm=prior["density_norm"],
            obstacle_proximity=prior["obstacle_proximity"],
            obstacle_turn_score=prior["obstacle_turn_score"],
            turn_count=prior["turn_count"],
            turn_base_count=prior["turn_base_count"],
            branch_peak_count=prior["branch_peak_count"],
            direction_hist=prior["direction_hist"],
        )
        if bool(getattr(config, "map_prior_plot", True)):
            plot_map_prior(prior, config, savedir)

    return prior


def plot_map_prior(prior, config, savedir):
    """Save a compact diagnostic image for the prior components."""
    import matplotlib.pyplot as plt

    components = [
        ("density", prior["density_norm"]),
        ("turn_rate", prior["turn_rate"]),
        ("branch", prior["branch_score"]),
        ("entropy", prior["direction_entropy"]),
    ]
    if prior["features"].shape[-1] >= 6:
        components.extend(
            [
                ("obstacle", prior["obstacle_proximity"]),
                ("obstacle_turn", prior["obstacle_turn_score"]),
            ]
        )

    n = len(components)
    ncols = 3 if n > 4 else 2
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), dpi=150)
    axes = np.asarray(axes).reshape(-1)

    for ax, (title, arr) in zip(axes, components):
        im = ax.imshow(arr, origin="lower", cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for ax in axes[len(components):]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(savedir, "map_prior_components.png"))
    plt.close(fig)

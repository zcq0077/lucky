# coding=utf-8
"""Generate cached Qwen semantic vectors for TrAISformer.

The cache stores hidden embeddings, not JSON labels. Prompts contain only
history and map statistics, so the same cache can be used during training and
test evaluation without leaking future trajectory points.
"""

import argparse
import os
import pickle

import numpy as np

import map_prior
from config_trAISformer import Config
from qwen_semantic_encoder import (
    QwenSemanticEncoder,
    build_history_summary,
    build_semantic_prompt,
    prompt_hash,
)


def load_phase_data(config, phase, moving_threshold=0.05):
    filenames = {
        "train": config.trainset_name,
        "valid": config.validset_name,
        "test": config.testset_name,
    }
    datapath = os.path.join(config.datadir, filenames[phase])
    with open(datapath, "rb") as f:
        data = pickle.load(f)

    for vessel in data:
        try:
            moving_idx = np.where(vessel["traj"][:, 2] > moving_threshold)[0][0]
        except Exception:
            moving_idx = len(vessel["traj"]) - 1
        vessel["traj"] = vessel["traj"][moving_idx:, :]

    return [
        x
        for x in data
        if not np.isnan(x["traj"]).any() and len(x["traj"]) > config.min_seqlen
    ]


def phase_cache_path(config, phase):
    return os.path.join(
        config.qwen_cache_dir,
        f"{config.dataset_name}_{phase}_qwen_vecs.npz",
    )


def parse_args():
    config = Config()
    parser = argparse.ArgumentParser(description="Generate Qwen semantic embedding cache.")
    parser.add_argument("--phase", choices=["train", "valid", "test", "all"], default="all")
    parser.add_argument("--model-path", default=config.qwen_model_path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=config.qwen_cache_batch_size)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    os.makedirs(config.qwen_cache_dir, exist_ok=True)

    print("======= Loading AIS data")
    all_data = {
        "train": load_phase_data(config, "train"),
        "valid": load_phase_data(config, "valid"),
        "test": load_phase_data(config, "test"),
    }

    print("======= Building map prior from train split")
    old_plot = getattr(config, "map_prior_plot", True)
    config.map_prior_plot = False
    prior = map_prior.build_map_prior(all_data["train"], config, savedir=None)
    config.map_prior_plot = old_plot

    phases = ["train", "valid", "test"] if args.phase == "all" else [args.phase]
    encoder = None
    if not args.dry_run:
        print(f"======= Loading Qwen model from: {args.model_path}")
        encoder = QwenSemanticEncoder(
            args.model_path,
            device=args.device,
            freeze=bool(getattr(config, "qwen_freeze", True)),
        )
        print(f"======= Qwen hidden size: {encoder.hidden_size}")

    for phase in phases:
        out_path = phase_cache_path(config, phase)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"======= Skip existing cache: {out_path}")
            continue

        data = all_data[phase]
        n_total = len(data)
        n_use = n_total if args.max_samples <= 0 else min(args.max_samples, n_total)
        prompts, hashes = [], []
        mmsis, time_starts, trajectory_ids = [], [], []

        print(f"======= Building {phase} prompts: {n_use}/{n_total}")
        for idx, vessel in enumerate(data[:n_use]):
            summary = build_history_summary(vessel, idx, config, prior)
            prompt = build_semantic_prompt(summary)
            prompts.append(prompt)
            hashes.append(prompt_hash(prompt))
            mmsis.append(int(vessel["mmsi"]))
            time_starts.append(int(vessel["traj"][0, 4]))
            trajectory_ids.append(int(idx))

        if args.dry_run:
            print("======= First dry-run prompt")
            print(prompts[0][:4000] if prompts else "NO_PROMPT")
            vectors = np.zeros((n_use, int(getattr(config, "qwen_hidden_size", 2560))), dtype=np.float32)
        else:
            print(f"======= Encoding {len(prompts)} {phase} prompts with Qwen")
            vectors = encoder.encode_prompts(
                prompts,
                batch_size=max(1, int(args.batch_size)),
                desc=f"Qwen {phase}",
            )

        if n_use < n_total:
            padded = np.zeros((n_total, vectors.shape[1]), dtype=np.float32)
            mask = np.zeros((n_total,), dtype=np.float32)
            padded[:n_use] = vectors
            mask[:n_use] = 1.0
            vectors = padded
        else:
            mask = np.ones((n_total,), dtype=np.float32)

        np.savez_compressed(
            out_path,
            qwen_vecs=vectors.astype(np.float32),
            qwen_mask=mask.astype(np.float32),
            trajectory_ids=np.asarray(trajectory_ids, dtype=np.int64),
            mmsis=np.asarray(mmsis, dtype=np.int64),
            time_starts=np.asarray(time_starts, dtype=np.int64),
            prompt_hashes=np.asarray(hashes),
            phase=np.asarray(phase),
            model_path=np.asarray(args.model_path),
        )
        print(f"======= Saved {phase} cache: {out_path}  shape={vectors.shape}")


if __name__ == "__main__":
    main()

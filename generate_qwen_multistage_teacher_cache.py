# coding=utf-8
"""Generate multi-stage Qwen teacher vectors for Captain Student distillation."""

import argparse
import os
import pickle

import numpy as np
from tqdm import tqdm

import map_prior
from captain_student import complexity_label_from_prefix, turn_label_from_prefix
from config_trAISformer import Config
from qwen_semantic_encoder import (
    QwenSemanticEncoder,
    build_semantic_prompt,
    build_sequence_summary,
    prompt_hash,
)


def parse_stage_lengths(value):
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


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


def teacher_cache_path(config, phase):
    return os.path.join(
        config.captain_student_teacher_cache_dir,
        f"{config.dataset_name}_{phase}_qwen_teacher_multistage.npz",
    )


def parse_args():
    config = Config()
    parser = argparse.ArgumentParser(description="Generate multi-stage Qwen teacher cache.")
    parser.add_argument("--phase", choices=["train", "valid", "test", "all"], default="all")
    parser.add_argument("--model-path", default=config.qwen_model_path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=config.qwen_cache_batch_size)
    parser.add_argument("--stage-lengths", default=",".join(str(x) for x in config.captain_student_stage_lengths))
    parser.add_argument("--prompt-max-points", type=int, default=config.captain_student_max_points)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = Config()
    stage_lengths = parse_stage_lengths(args.stage_lengths)
    os.makedirs(config.captain_student_teacher_cache_dir, exist_ok=True)

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

    old_prompt_max_points = getattr(config, "qwen_prompt_max_points", config.init_seqlen)
    config.qwen_prompt_max_points = int(args.prompt_max_points)

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
        out_path = teacher_cache_path(config, phase)
        if os.path.exists(out_path) and not args.overwrite:
            print(f"======= Skip existing teacher cache: {out_path}")
            continue

        data = all_data[phase]
        n_total = len(data)
        n_use = n_total if args.max_samples <= 0 else min(args.max_samples, n_total)
        data_use = data[:n_use]

        flat_prompts, flat_meta, flat_hashes = [], [], []
        stage_masks = np.zeros((n_total, len(stage_lengths)), dtype=np.float32)
        turn_labels = np.zeros((n_total, len(stage_lengths)), dtype=np.int64)
        turn_deltas = np.zeros((n_total, len(stage_lengths)), dtype=np.float32)
        complexity_labels = np.zeros((n_total, len(stage_lengths)), dtype=np.float32)
        mmsis = np.zeros((n_total,), dtype=np.int64)
        time_starts = np.zeros((n_total,), dtype=np.int64)

        print(f"======= Building {phase} multi-stage prompts: {n_use}/{n_total}")
        for idx, vessel in enumerate(tqdm(data_use, desc=f"{phase} prompts")):
            traj = vessel["traj"]
            mmsis[idx] = int(vessel["mmsi"])
            time_starts[idx] = int(traj[0, 4])
            for s_idx, stage_len in enumerate(stage_lengths):
                if len(traj) < stage_len:
                    continue
                prefix = np.asarray(traj[:stage_len, :4], dtype=np.float32)
                summary = build_sequence_summary(
                    prefix,
                    config,
                    prior=prior,
                    trajectory_id=idx,
                    mmsi=int(vessel["mmsi"]),
                )
                summary["teacher_stage_len"] = int(stage_len)
                summary["teacher_stage_index"] = int(s_idx)
                prompt = build_semantic_prompt(summary)
                flat_prompts.append(prompt)
                flat_hashes.append(prompt_hash(prompt))
                flat_meta.append((idx, s_idx))
                stage_masks[idx, s_idx] = 1.0
                label, delta = turn_label_from_prefix(
                    traj,
                    stage_len,
                    config,
                    future_window=int(getattr(config, "captain_student_turn_future_window", 3)),
                )
                turn_labels[idx, s_idx] = int(label)
                turn_deltas[idx, s_idx] = float(delta)
                complexity_labels[idx, s_idx] = complexity_label_from_prefix(prefix, prior, config)

        if args.dry_run:
            print("======= First dry-run prompt")
            print(flat_prompts[0][:4000] if flat_prompts else "NO_PROMPT")
            print("======= Dry run only; no teacher cache file was saved.")
            continue
        else:
            print(f"======= Encoding {len(flat_prompts)} {phase} stage prompts with Qwen")
            flat_vectors = encoder.encode_prompts(
                flat_prompts,
                batch_size=max(1, int(args.batch_size)),
                desc=f"Qwen teacher {phase}",
            )
            hidden_size = flat_vectors.shape[1]

        teacher_vecs = np.zeros((n_total, len(stage_lengths), hidden_size), dtype=np.float32)
        for vec, (idx, s_idx) in zip(flat_vectors, flat_meta):
            teacher_vecs[idx, s_idx] = vec

        np.savez_compressed(
            out_path,
            teacher_vecs=teacher_vecs.astype(np.float32),
            stage_masks=stage_masks.astype(np.float32),
            stage_lengths=np.asarray(stage_lengths, dtype=np.int64),
            turn_labels=turn_labels,
            turn_deltas=turn_deltas,
            complexity_labels=complexity_labels.astype(np.float32),
            mmsis=mmsis,
            time_starts=time_starts,
            prompt_hashes=np.asarray(flat_hashes),
            prompt_meta=np.asarray(flat_meta, dtype=np.int64),
            phase=np.asarray(phase),
            model_path=np.asarray(args.model_path),
            prompt_max_points=np.asarray(int(args.prompt_max_points)),
        )
        print(f"======= Saved {phase} teacher cache: {out_path}  shape={teacher_vecs.shape}")

    config.qwen_prompt_max_points = old_prompt_max_points


if __name__ == "__main__":
    main()

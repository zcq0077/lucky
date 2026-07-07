# coding=utf-8
"""Train a lightweight Captain Student from multi-stage Qwen teacher cache."""

import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import map_prior
from captain_student import (
    CaptainStudent,
    pad_sequence_prefix,
    query_map_features_np,
    save_student_checkpoint,
    student_distillation_loss,
)
from config_trAISformer import Config


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


class CaptainStudentDataset(Dataset):
    def __init__(self, data, teacher_cache, config, prior=None, max_samples=0):
        self.data = data
        self.teacher_vecs = teacher_cache["teacher_vecs"].astype(np.float32)
        self.stage_masks = teacher_cache["stage_masks"].astype(np.float32)
        self.stage_lengths = teacher_cache["stage_lengths"].astype(np.int64)
        self.turn_labels = teacher_cache["turn_labels"].astype(np.int64)
        self.complexity_labels = teacher_cache["complexity_labels"].astype(np.float32)
        self.max_points = int(getattr(config, "captain_student_max_points", 42))
        self.use_map_features = bool(getattr(config, "captain_student_use_map_features", True))
        self.map_channels = int(getattr(config, "map_prior_channels", 4))
        self.prior = prior

        if len(self.teacher_vecs) != len(data):
            raise ValueError(
                f"Teacher cache length mismatch: cache has {len(self.teacher_vecs)}, "
                f"data has {len(data)}"
            )

        self.items = []
        n_data = len(data) if max_samples <= 0 else min(int(max_samples), len(data))
        for traj_idx in range(n_data):
            for stage_idx, stage_len in enumerate(self.stage_lengths):
                if self.stage_masks[traj_idx, stage_idx] <= 0:
                    continue
                if len(data[traj_idx]["traj"]) < int(stage_len):
                    continue
                self.items.append((traj_idx, stage_idx))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, item_idx):
        traj_idx, stage_idx = self.items[item_idx]
        stage_len = int(self.stage_lengths[stage_idx])
        traj = self.data[traj_idx]["traj"][:stage_len, :4]
        seq, seq_mask = pad_sequence_prefix(traj, self.max_points)
        if self.use_map_features:
            map_features = query_map_features_np(
                seq[np.newaxis, ...],
                self.prior,
                self.max_points,
                self.map_channels,
            )[0]
        else:
            map_features = np.zeros((self.max_points, 0), dtype=np.float32)

        return {
            "seq": torch.as_tensor(seq, dtype=torch.float32),
            "seq_mask": torch.as_tensor(seq_mask, dtype=torch.float32),
            "map_features": torch.as_tensor(map_features, dtype=torch.float32),
            "teacher_vec": torch.as_tensor(self.teacher_vecs[traj_idx, stage_idx], dtype=torch.float32),
            "turn_label": torch.as_tensor(self.turn_labels[traj_idx, stage_idx], dtype=torch.long),
            "complexity_label": torch.as_tensor(self.complexity_labels[traj_idx, stage_idx], dtype=torch.float32),
            "teacher_mask": torch.as_tensor(self.stage_masks[traj_idx, stage_idx], dtype=torch.float32),
        }


def parse_args():
    config = Config()
    parser = argparse.ArgumentParser(description="Train Captain Student from Qwen teacher cache.")
    parser.add_argument("--epochs", type=int, default=config.captain_student_epochs)
    parser.add_argument("--batch-size", type=int, default=config.captain_student_batch_size)
    parser.add_argument("--lr", type=float, default=config.captain_student_lr)
    parser.add_argument("--device", default=str(config.device))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--output", default=config.captain_student_ckpt_path)
    return parser.parse_args()


def run_epoch(model, loader, optimizer, config, device, is_train):
    model.train(is_train)
    total_loss = 0.0
    total_n = 0
    pbar = tqdm(loader, desc="train" if is_train else "valid")
    for batch in pbar:
        seq = batch["seq"].to(device)
        seq_mask = batch["seq_mask"].to(device)
        map_features = batch["map_features"].to(device)
        if map_features.size(-1) == 0:
            map_features = None
        teacher_vec = batch["teacher_vec"].to(device)
        turn_label = batch["turn_label"].to(device)
        complexity_label = batch["complexity_label"].to(device)
        teacher_mask = batch["teacher_mask"].to(device)

        with torch.set_grad_enabled(is_train):
            outputs = model(seq, seq_mask, map_features)
            loss, parts = student_distillation_loss(
                outputs,
                teacher_vec,
                turn_label=turn_label,
                complexity_label=complexity_label,
                mask=teacher_mask,
                cos_w=float(getattr(config, "captain_student_cos_loss_w", 1.0)),
                mse_w=float(getattr(config, "captain_student_mse_loss_w", 0.05)),
                intent_w=float(getattr(config, "captain_student_intent_loss_w", 0.2)),
                complexity_w=float(getattr(config, "captain_student_complexity_loss_w", 0.05)),
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(getattr(config, "captain_student_grad_clip", 1.0)),
                )
                optimizer.step()

        batch_n = seq.size(0)
        total_loss += float(loss.detach().cpu()) * batch_n
        total_n += batch_n
        pbar.set_postfix(loss=f"{total_loss / max(total_n, 1):.4f}", **parts)
    return total_loss / max(total_n, 1)


def main():
    args = parse_args()
    config = Config()
    device = torch.device(args.device)

    print("======= Loading AIS data")
    train_data = load_phase_data(config, "train")
    valid_data = load_phase_data(config, "valid")

    print("======= Building map prior from train split")
    old_plot = getattr(config, "map_prior_plot", True)
    config.map_prior_plot = False
    prior = map_prior.build_map_prior(train_data, config, savedir=None)
    config.map_prior_plot = old_plot

    train_cache_path = teacher_cache_path(config, "train")
    valid_cache_path = teacher_cache_path(config, "valid")
    if not os.path.exists(train_cache_path) or not os.path.exists(valid_cache_path):
        raise FileNotFoundError(
            "Teacher cache not found. Generate it first, for example:\n"
            "python generate_qwen_multistage_teacher_cache.py --phase all --device cuda --overwrite"
        )
    train_cache = np.load(train_cache_path, allow_pickle=True)
    valid_cache = np.load(valid_cache_path, allow_pickle=True)

    train_ds = CaptainStudentDataset(train_data, train_cache, config, prior=prior, max_samples=args.max_samples)
    valid_ds = CaptainStudentDataset(valid_data, valid_cache, config, prior=prior, max_samples=args.max_samples)
    print(f"======= Captain Student train samples: {len(train_ds)}")
    print(f"======= Captain Student valid samples: {len(valid_ds)}")

    output_dim = int(train_cache["teacher_vecs"].shape[-1])
    map_channels = int(getattr(config, "map_prior_channels", 4))
    use_map_features = bool(getattr(config, "captain_student_use_map_features", True))
    input_dim = 4 + (map_channels if use_map_features else 0)
    model = CaptainStudent(
        input_dim=input_dim,
        hidden_dim=int(getattr(config, "captain_student_hidden_dim", 256)),
        output_dim=output_dim,
        num_layers=int(getattr(config, "captain_student_num_layers", 1)),
        dropout=float(getattr(config, "captain_student_dropout", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(getattr(config, "captain_student_weight_decay", 1e-4)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(getattr(config, "num_workers", 0)),
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(getattr(config, "num_workers", 0)),
        pin_memory=True,
    )

    metadata = {
        "input_dim": input_dim,
        "hidden_dim": int(getattr(config, "captain_student_hidden_dim", 256)),
        "output_dim": output_dim,
        "num_layers": int(getattr(config, "captain_student_num_layers", 1)),
        "dropout": float(getattr(config, "captain_student_dropout", 0.1)),
        "max_points": int(getattr(config, "captain_student_max_points", 42)),
        "use_map_features": use_map_features,
        "map_channels": map_channels,
        "teacher_cache_dir": config.captain_student_teacher_cache_dir,
        "stage_lengths": [int(x) for x in train_cache["stage_lengths"].tolist()],
    }

    best_valid = float("inf")
    for epoch in range(int(args.epochs)):
        print(f"======= Captain Student epoch {epoch + 1}/{args.epochs}")
        train_loss = run_epoch(model, train_loader, optimizer, config, device, is_train=True)
        valid_loss = run_epoch(model, valid_loader, optimizer, config, device, is_train=False)
        print(f"======= epoch {epoch + 1}: train={train_loss:.5f} valid={valid_loss:.5f}")
        if valid_loss < best_valid:
            best_valid = valid_loss
            save_student_checkpoint(args.output, model, metadata)
            print(f"======= Saved best Captain Student: {args.output} valid={best_valid:.5f}")


if __name__ == "__main__":
    main()

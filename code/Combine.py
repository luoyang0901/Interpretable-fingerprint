#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train one or more TextCNN(FPtand) models for one data branch.

Reviewer-revision features:
- target-blind structure-only group split (default);
- identical encoded representations never cross split boundaries;
- optional legacy target-informed HSPXY benchmark retained with an explicit name;
- legacy and role-aware/balanced FPtand encodings;
- full active-bit and truncation audit;
- calibration, residual, stratified-error and ranking metrics;
- validation-RMSE model selection and machine-readable predictions.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset

from common_utils import (
    encode_fptand,
    environment_report,
    factorize_hashes,
    paired_row_hashes,
    paired_similarity_matrix,
    random_group_split,
    ranking_metrics,
    regression_metrics,
    residual_table,
    save_json,
    stratified_error_table,
    structure_ks_group_split,
    validate_group_disjoint,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SeqRegDataset(Dataset):
    def __init__(self, seq: np.ndarray, y: np.ndarray):
        self.seq = torch.as_tensor(seq, dtype=torch.long)
        self.y = torch.as_tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, idx: int):
        return self.seq[idx], self.y[idx]


class GaoStrictTextCNN(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, channels: int, kernel_size: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        nn.init.uniform_(self.embedding.weight, -1.0, 1.0)
        with torch.no_grad():
            self.embedding.weight[0].fill_(0.0)
        self.conv = nn.Conv2d(1, channels, kernel_size=(kernel_size, embedding_dim))
        nn.init.normal_(self.conv.weight, std=0.01)
        nn.init.constant_(self.conv.bias, 0.01)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).unsqueeze(1)
        x = torch.relu(self.conv(x)).squeeze(3)
        x = torch.max(x, dim=2)[0]
        return self.fc(self.dropout(x))


class StrongTextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        channels: int,
        kernel_sizes: Tuple[int, ...],
        dropout: float,
        hidden_dim: int,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        with torch.no_grad():
            self.embedding.weight[0].fill_(0.0)
        self.convs = nn.ModuleList([nn.Conv2d(1, channels, kernel_size=(k, embedding_dim)) for k in kernel_sizes])
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
            nn.init.constant_(conv.bias, 0.0)
        self.ln = nn.LayerNorm(len(kernel_sizes) * channels)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(len(kernel_sizes) * channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x).unsqueeze(1)
        pooled = []
        for conv in self.convs:
            z = torch.relu(conv(x)).squeeze(3)
            pooled.append(torch.max(z, dim=2)[0])
        x = self.dropout(self.ln(torch.cat(pooled, dim=1)))
        return self.head(x)


@dataclass
class TrainConfig:
    profile: str
    split_method: str
    split_seed: int
    model_seeds: List[int]
    test_size: float
    valid_fraction_of_trainval: float
    encoding_mode: str
    batch_size: int
    epochs: int
    patience: int
    lr: float
    weight_decay: float
    grad_clip: float
    max_len: int
    embedding_dim: int
    channels: int
    dropout: float
    kernel_sizes: List[int]
    hidden_dim: int
    loss: str
    device: str
    high_pce_threshold: float


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def load_arrays(fd_path: str, fa_path: str, y_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fd = np.asarray(np.load(fd_path))
    fa = np.asarray(np.load(fa_path))
    y = np.asarray(np.load(y_path)).reshape(-1)
    if fd.shape != fa.shape or fd.ndim != 2 or fd.shape[1] != 1024:
        raise ValueError(f"Expected equal (n,1024) fd/fa arrays, got {fd.shape} and {fa.shape}")
    if len(y) != len(fd):
        raise ValueError(f"Target length mismatch: {y.shape} vs {fd.shape}")
    for name, arr in [("fd", fd), ("fa", fa)]:
        if not set(np.unique(arr).tolist()).issubset({0, 1}):
            raise ValueError(f"{name} is not binary")
    return fd.astype(np.int8), fa.astype(np.int8), y.astype(np.float32)


def target_informed_hspxy_group_split(
    fd: np.ndarray,
    fa: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    valid_fraction_of_trainval: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Legacy benchmark: group-safe HSPXY using both representation and target."""
    unique_groups = np.unique(groups)
    reps = np.asarray([np.flatnonzero(groups == g)[0] for g in unique_groups], dtype=int)
    sizes = np.asarray([np.sum(groups == g) for g in unique_groups], dtype=int)
    y_group = np.asarray([np.mean(y[groups == g]) for g in unique_groups], dtype=float)
    structure_distance = 1.0 - paired_similarity_matrix(fd[reps], fa[reps], fd[reps], fa[reps])
    y_std = (y_group - np.mean(y_group)) / (np.std(y_group) + 1e-12)
    target_distance = np.abs(y_std[:, None] - y_std[None, :])
    smax = max(float(np.max(structure_distance)), 1e-12)
    ymax = max(float(np.max(target_distance)), 1e-12)
    distance = structure_distance / smax + target_distance / ymax
    np.fill_diagonal(distance, 0.0)
    rng = np.random.default_rng(seed)
    max_value = np.max(distance)
    pairs = np.argwhere(np.isclose(distance, max_value))
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    first, second = (0, 1) if len(pairs) == 0 else pairs[int(rng.integers(0, len(pairs)))]
    selected = [int(first), int(second)]
    mask = np.zeros(len(unique_groups), dtype=bool)
    mask[selected] = True
    target_n = int(round((1.0 - test_size) * len(y)))
    current_n = int(sizes[selected].sum())
    while current_n < target_n and int(mask.sum()) < len(unique_groups) - 1:
        remain = np.flatnonzero(~mask)
        mind = np.min(distance[np.ix_(remain, np.asarray(selected))], axis=1)
        candidates = remain[np.isclose(mind, np.max(mind))]
        chosen = int(candidates[int(rng.integers(0, len(candidates)))])
        selected.append(chosen)
        mask[chosen] = True
        current_n += int(sizes[chosen])
    trainval_groups = set(unique_groups[np.asarray(selected)].tolist())
    trainval_idx = np.asarray([i for i, g in enumerate(groups) if g in trainval_groups], dtype=int)
    test_idx = np.asarray([i for i, g in enumerate(groups) if g not in trainval_groups], dtype=int)
    gss = GroupShuffleSplit(n_splits=1, test_size=valid_fraction_of_trainval, random_state=seed + 300007)
    tr_local, va_local = next(gss.split(trainval_idx, groups=groups[trainval_idx]))
    return {"train_idx": trainval_idx[tr_local], "valid_idx": trainval_idx[va_local], "test_idx": test_idx}


def make_split(
    fd: np.ndarray,
    fa: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cfg: TrainConfig,
    precomputed_split: str | None,
) -> dict[str, np.ndarray]:
    if cfg.split_method == "structure_ks":
        split = structure_ks_group_split(fd, fa, groups, cfg.test_size, cfg.valid_fraction_of_trainval, cfg.split_seed)
    elif cfg.split_method == "random_group":
        split = random_group_split(len(y), groups, cfg.test_size, cfg.valid_fraction_of_trainval, cfg.split_seed)
    elif cfg.split_method == "hspxy_legacy":
        split = target_informed_hspxy_group_split(
            fd, fa, y, groups, cfg.test_size, cfg.valid_fraction_of_trainval, cfg.split_seed
        )
    elif cfg.split_method == "precomputed":
        if not precomputed_split:
            raise ValueError("--precomputed-split is required for split_method=precomputed")
        z = np.load(precomputed_split)
        split = {name: np.asarray(z[f"{name}_idx"], dtype=int) for name in ["train", "valid", "test"]}
    else:
        raise ValueError(f"Unknown split method: {cfg.split_method}")
    validate_group_disjoint(split, groups)
    return split


def make_model(cfg: TrainConfig, vocab_size: int) -> nn.Module:
    if cfg.profile == "strict_gao":
        if len(cfg.kernel_sizes) != 1:
            raise ValueError("strict_gao requires one kernel size")
        return GaoStrictTextCNN(vocab_size, cfg.embedding_dim, cfg.channels, cfg.kernel_sizes[0], cfg.dropout)
    if cfg.profile == "strong":
        return StrongTextCNN(
            vocab_size, cfg.embedding_dim, cfg.channels, tuple(cfg.kernel_sizes), cfg.dropout, cfg.hidden_dim
        )
    raise ValueError(f"Unknown profile: {cfg.profile}")


def loss_function(name: str):
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError(name)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    true, pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            out = model(xb.to(device)).detach().cpu().numpy().reshape(-1)
            true.append(yb.numpy().reshape(-1))
            pred.append(out)
    y_true = np.concatenate(true)
    y_pred = np.concatenate(pred)
    return y_true, y_pred, regression_metrics(y_true, y_pred)


def train_one_seed(
    seq: np.ndarray,
    y: np.ndarray,
    split: dict[str, np.ndarray],
    cfg: TrainConfig,
    vocab_size: int,
    model_seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    seed_everything(model_seed)
    tr, va, te = split["train_idx"], split["valid_idx"], split["test_idx"]
    dl_train = DataLoader(SeqRegDataset(seq[tr], y[tr]), batch_size=cfg.batch_size, shuffle=True)
    dl_valid = DataLoader(SeqRegDataset(seq[va], y[va]), batch_size=cfg.batch_size, shuffle=False)
    dl_test = DataLoader(SeqRegDataset(seq[te], y[te]), batch_size=cfg.batch_size, shuffle=False)
    model = make_model(cfg, vocab_size).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(5, cfg.patience // 4)
    )
    loss_fn = loss_function(cfg.loss)
    best_rmse = float("inf")
    best_state = None
    wait = 0
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = []
        for xb, yb in dl_train:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            losses.append(float(loss.item()))
        _, _, train_m = evaluate(model, dl_train, cfg.device)
        _, _, valid_m = evaluate(model, dl_valid, cfg.device)
        scheduler.step(valid_m["rmse"])
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else np.nan,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"valid_{k}": v for k, v in valid_m.items()},
            "lr": optimizer.param_groups[0]["lr"],
        })
        if valid_m["rmse"] < best_rmse:
            best_rmse = valid_m["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    y_tr, p_tr, m_tr = evaluate(model, dl_train, cfg.device)
    y_va, p_va, m_va = evaluate(model, dl_valid, cfg.device)
    y_te, p_te, m_te = evaluate(model, dl_test, cfg.device)
    seed_dir = out_dir / f"model_seed_{model_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(seed_dir / "train_history.csv", index=False, encoding="utf-8-sig")
    residual_table(y_va, p_va, va).to_csv(seed_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    residual_table(y_te, p_te, te).to_csv(seed_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    stratified_error_table(y_te, p_te).to_csv(seed_dir / "test_pce_stratified_errors.csv", index=False, encoding="utf-8-sig")
    ranking_metrics(y_te, p_te, high_threshold=cfg.high_pce_threshold).to_csv(
        seed_dir / "test_ranking_metrics.csv", index=False, encoding="utf-8-sig"
    )
    torch.save(best_state, seed_dir / "best_model.pt")
    save_json(seed_dir / "model_metrics.json", {
        "model_seed": model_seed,
        "best_validation_rmse_during_training": best_rmse,
        "train_metrics": m_tr,
        "validation_metrics": m_va,
        "test_metrics": m_te,
    })
    return {
        "model_seed": model_seed,
        "best_valid_rmse": float(best_rmse),
        "valid_metrics": m_va,
        "test_metrics": m_te,
        "test_true": y_te,
        "test_pred": p_te,
        "valid_true": y_va,
        "valid_pred": p_va,
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fd-path", required=True)
    p.add_argument("--fa-path", required=True)
    p.add_argument("--y-path", required=True)
    p.add_argument("--group-path", help="NPY group IDs for the encoded representation. Identical groups stay together.")
    p.add_argument("--metadata-csv", help="Optional sample metadata copied into the run directory.")
    p.add_argument("--precomputed-split")
    p.add_argument("--output-dir", default="textcnn_run")
    p.add_argument("--profile", choices=["strict_gao", "strong"], default="strong")
    p.add_argument("--split-method", choices=["structure_ks", "random_group", "hspxy_legacy", "precomputed"], default="structure_ks")
    p.add_argument("--split-seed", type=int, default=12)
    p.add_argument("--model-seeds", default="12")
    p.add_argument("--encoding-mode", choices=["legacy", "role_aware"], default="role_aware")
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--valid-fraction-of-trainval", type=float, default=0.125)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.35)
    p.add_argument("--kernel-sizes", default="3,5,7")
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--loss", choices=["mse", "huber"], default="huber")
    p.add_argument("--high-pce-threshold", type=float, default=16.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = TrainConfig(
        profile=args.profile,
        split_method=args.split_method,
        split_seed=args.split_seed,
        model_seeds=parse_int_list(args.model_seeds),
        test_size=args.test_size,
        valid_fraction_of_trainval=args.valid_fraction_of_trainval,
        encoding_mode=args.encoding_mode,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_len=args.max_len,
        embedding_dim=args.embedding_dim,
        channels=args.channels,
        dropout=args.dropout,
        kernel_sizes=parse_int_list(args.kernel_sizes),
        hidden_dim=args.hidden_dim,
        loss=args.loss,
        device=args.device,
        high_pce_threshold=args.high_pce_threshold,
    )
    save_json(out / "environment.json", environment_report())
    fd, fa, y = load_arrays(args.fd_path, args.fa_path, args.y_path)
    seq, encoding_audit, vocab_size = encode_fptand(fd, fa, cfg.max_len, cfg.encoding_mode)
    encoding_audit.to_csv(out / "encoding_audit.csv", index=False, encoding="utf-8-sig")
    save_json(out / "encoding_summary.json", {
        "encoding_mode": cfg.encoding_mode,
        "vocab_size": vocab_size,
        "max_len": cfg.max_len,
        "n_samples": int(len(seq)),
        "truncated_n": int(encoding_audit["truncated"].sum()),
        "truncated_fraction": float(encoding_audit["truncated"].mean()),
        "total_donor_tokens_dropped": int(encoding_audit["donor_tokens_dropped"].sum()),
        "total_acceptor_tokens_dropped": int(encoding_audit["acceptor_tokens_dropped"].sum()),
    })
    if args.group_path:
        groups = np.asarray(np.load(args.group_path)).reshape(-1)
        if len(groups) != len(y):
            raise ValueError(f"group-path length {len(groups)} does not match n={len(y)}")
    else:
        groups = factorize_hashes(paired_row_hashes(fd, fa))
    split = make_split(fd, fa, y, groups, cfg, args.precomputed_split)
    group_audit = validate_group_disjoint(split, groups)
    np.savez_compressed(out / "split_indices.npz", **split)
    save_json(out / "split_audit.json", {
        "split_method": cfg.split_method,
        "target_values_used_to_construct_split": cfg.split_method == "hspxy_legacy",
        "representation_group_path": args.group_path,
        "counts": {name: int(len(split[f"{name}_idx"])) for name in ["train", "valid", "test"]},
        **group_audit,
    })
    if args.metadata_csv:
        meta = pd.read_csv(args.metadata_csv)
        if len(meta) != len(y):
            raise ValueError("metadata row count does not match arrays")
        meta.to_csv(out / "sample_metadata.csv", index=False, encoding="utf-8-sig")

    runs = [train_one_seed(seq, y, split, cfg, vocab_size, seed, out) for seed in cfg.model_seeds]
    best = min(runs, key=lambda x: (x["best_valid_rmse"], x["model_seed"]))
    pred_matrix = np.column_stack([r["test_pred"] for r in runs])
    ensemble_pred = pred_matrix.mean(axis=1)
    ensemble_metrics = regression_metrics(best["test_true"], ensemble_pred)
    residual_table(best["test_true"], ensemble_pred, split["test_idx"]).to_csv(
        out / "ensemble_predictions.csv", index=False, encoding="utf-8-sig"
    )
    ranking_metrics(best["test_true"], ensemble_pred, high_threshold=cfg.high_pce_threshold).to_csv(
        out / "ensemble_ranking_metrics.csv", index=False, encoding="utf-8-sig"
    )
    stratified_error_table(best["test_true"], ensemble_pred).to_csv(
        out / "ensemble_pce_stratified_errors.csv", index=False, encoding="utf-8-sig"
    )
    rows = []
    for r in runs:
        rows.append({
            "model_seed": r["model_seed"],
            "best_valid_rmse": r["best_valid_rmse"],
            **{f"valid_{k}": v for k, v in r["valid_metrics"].items()},
            **{f"test_{k}": v for k, v in r["test_metrics"].items()},
        })
    pd.DataFrame(rows).to_csv(out / "per_seed_summary.csv", index=False, encoding="utf-8-sig")
    summary = {
        "config": asdict(cfg),
        "n_samples_total": int(len(y)),
        "n_train": int(len(split["train_idx"])),
        "n_valid": int(len(split["valid_idx"])),
        "n_test": int(len(split["test_idx"])),
        "best_single_model_seed": int(best["model_seed"]),
        "selection_criterion": "lowest validation RMSE",
        "best_single_test_metrics": best["test_metrics"],
        "ensemble_test_metrics": ensemble_metrics,
        "target_values_used_to_construct_split": cfg.split_method == "hspxy_legacy",
        "identical_encoded_representations_kept_in_one_subset": True,
    }
    save_json(out / "run_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

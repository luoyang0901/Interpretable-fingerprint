#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    if len(y_true) < 2:
        return float("nan")
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return {
        "r": safe_pearsonr(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def tanimoto_distance_binary(a: np.ndarray, b: np.ndarray) -> float:
    inter = float(np.dot(a, b))
    denom = float(np.dot(a, a) + np.dot(b, b) - inter)
    if denom <= 1e-12:
        return 0.0
    return 1.0 - inter / denom


def custom_distance_gao(
    x1: np.ndarray,
    x2: np.ndarray,
    fp_length: int = 1024,
    gamma_d: float = 1.0,
    gamma_a: float = 1.0,
) -> float:
    d_part = tanimoto_distance_binary(x1[:fp_length], x2[:fp_length])
    a_part = tanimoto_distance_binary(
        x1[fp_length:fp_length * 2],
        x2[fp_length:fp_length * 2],
    )
    return gamma_d * d_part + gamma_a * a_part


def hspxy_split_gao(
    x: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    fp_length: int = 1024,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1)
    n_samples = x.shape[0]
    n_trainval = round((1 - test_size) * n_samples)
    sample_ids = np.arange(n_samples)

    y_std = (y - np.mean(y)) / (np.std(y) + 1e-12)
    d_struct = np.zeros((n_samples, n_samples), dtype=np.float32)
    d_target = np.zeros((n_samples, n_samples), dtype=np.float32)
    d_cos = np.zeros((n_samples, n_samples), dtype=np.float32)

    for i in range(n_samples - 1):
        xa = x[i, :]
        ya = y_std[i]
        for j in range(i + 1, n_samples):
            xb = x[j, :]
            yb = y_std[j]
            pair = np.vstack((xa, xb))
            d_struct[i, j] = custom_distance_gao(
                xa,
                xb,
                fp_length=fp_length,
                gamma_d=1,
                gamma_a=1,
            )
            d_target[i, j] = np.linalg.norm(ya - yb)
            try:
                d_cos[i, j] = abs(1 - pdist(pair, "cosine")[0])
            except Exception:
                d_cos[i, j] = 0.0

    d_struct_max = float(np.max(d_struct)) if np.max(d_struct) > 0 else 1.0
    d_target_max = float(np.max(d_target)) if np.max(d_target) > 0 else 1.0
    d_cos_max = float(np.max(d_cos)) if np.max(d_cos) > 0 else 1.0
    d_mix = 1 + d_struct / d_struct_max - d_cos / d_cos_max + d_target / d_target_max

    max_dist = d_mix.max(axis=0)
    index_row = d_mix.argmax(axis=0)
    index_col = max_dist.argmax()

    selected = np.zeros(n_trainval, dtype=int)
    selected[0] = index_row[index_col]
    selected[1] = index_col

    for i in range(2, n_trainval):
        pool = np.delete(sample_ids, selected[:i])
        d_min = np.zeros(len(pool), dtype=np.float32)
        for j, idx_a in enumerate(pool):
            d_vals = np.zeros(i, dtype=np.float32)
            for k in range(i):
                idx_b = selected[k]
                if idx_a < idx_b:
                    d_vals[k] = d_mix[idx_a, idx_b]
                else:
                    d_vals[k] = d_mix[idx_b, idx_a]
            d_min[j] = np.min(d_vals)
        selected[i] = pool[np.argmax(d_min)]

    train_idx = selected
    test_idx = np.delete(np.arange(n_samples), selected)
    return (
        x[train_idx],
        x[test_idx],
        y[train_idx],
        y[test_idx],
        train_idx,
        test_idx,
    )


def encode_fptand_from_fd_fa(
    fd: np.ndarray,
    fa: np.ndarray,
    max_len: int = 200,
) -> np.ndarray:
    fd = np.asarray(fd)
    fa = np.asarray(fa)
    assert fd.shape == fa.shape, "fd/fa shape mismatch"
    assert fd.shape[1] == 1024, "Expect 1024-bit donor/acceptor fingerprints"

    n_samples = fd.shape[0]
    seq = np.zeros((n_samples, max_len), dtype=np.int64)

    donor_on = [np.where(fd[i] == 1)[0] + 1 for i in range(n_samples)]
    acceptor_on = [np.where(fa[i] == 1)[0] + 1 for i in range(n_samples)]

    for i in range(n_samples):
        tokens = np.concatenate([donor_on[i], acceptor_on[i]], axis=0)[:max_len]
        seq[i, : len(tokens)] = tokens
    return seq


class SeqRegDataset(Dataset):
    def __init__(self, seq: np.ndarray, y: np.ndarray):
        self.seq = torch.as_tensor(seq, dtype=torch.long)
        self.y = torch.as_tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32)

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        return self.seq[idx], self.y[idx]


class GaoStrictTextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1024,
        embedding_dim: int = 100,
        channels: int = 128,
        kernel_size: int = 5,
        dropout: float = 0.5,
    ):
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
        nn.init.normal_(self.fc.weight, std=0.01)
        nn.init.constant_(self.fc.bias, 0.01)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = torch.relu(self.conv(x))
        x = x.squeeze(3)
        x = torch.max(x, dim=2)[0]
        x = self.dropout(x)
        return self.fc(x)


class StrongTextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int = 1024,
        embedding_dim: int = 128,
        channels: int = 128,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.35,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=0)
        nn.init.uniform_(self.embedding.weight, -0.5, 0.5)
        with torch.no_grad():
            self.embedding.weight[0].fill_(0.0)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, channels, kernel_size=(k, embedding_dim)) for k in kernel_sizes]
        )
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
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        pooled = []
        for conv in self.convs:
            z = torch.relu(conv(x)).squeeze(3)
            z = torch.max(z, dim=2)[0]
            pooled.append(z)
        x = torch.cat(pooled, dim=1)
        x = self.ln(x)
        x = self.dropout(x)
        return self.head(x)


@dataclass
class TrainConfig:
    profile: str
    split_method: str
    split_seed: int
    model_seeds: List[int]
    test_size: float
    valid_fraction_of_trainval: float
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


def make_model(cfg: TrainConfig) -> nn.Module:
    if cfg.profile == "strict_gao":
        if len(cfg.kernel_sizes) != 1:
            raise ValueError("strict_gao profile expects exactly one kernel size.")
        return GaoStrictTextCNN(
            vocab_size=1024,
            embedding_dim=cfg.embedding_dim,
            channels=cfg.channels,
            kernel_size=cfg.kernel_sizes[0],
            dropout=cfg.dropout,
        )
    if cfg.profile == "strong":
        return StrongTextCNN(
            vocab_size=1024,
            embedding_dim=cfg.embedding_dim,
            channels=cfg.channels,
            kernel_sizes=tuple(cfg.kernel_sizes),
            dropout=cfg.dropout,
            hidden_dim=cfg.hidden_dim,
        )
    raise ValueError(f"Unknown profile: {cfg.profile}")


def get_loss_fn(name: str):
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.SmoothL1Loss(beta=1.0)
    raise ValueError(f"Unknown loss: {name}")


def evaluate_model(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    y_list, pred_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy().reshape(-1)
            pred_list.append(out)
            y_list.append(yb.numpy().reshape(-1))
    y_true = np.concatenate(y_list, axis=0)
    y_pred = np.concatenate(pred_list, axis=0)
    return y_true, y_pred, regression_metrics(y_true, y_pred)


def train_one_seed(
    seq_train,
    y_train,
    seq_valid,
    y_valid,
    seq_test,
    y_test,
    cfg: TrainConfig,
    model_seed: int,
    out_dir: Path,
):
    seed_everything(model_seed)
    device = cfg.device

    ds_train = SeqRegDataset(seq_train, y_train)
    ds_valid = SeqRegDataset(seq_valid, y_valid)
    ds_test = SeqRegDataset(seq_test, y_test)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    dl_valid = DataLoader(ds_valid, batch_size=cfg.batch_size, shuffle=False, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = make_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(5, cfg.patience // 4),
    )
    loss_fn = get_loss_fn(cfg.loss)

    best_val_rmse = float("inf")
    best_state = None
    patience_count = 0
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        batch_losses = []
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            batch_losses.append(float(loss.item()))

        _, _, train_metrics = evaluate_model(model, dl_train, device)
        _, _, valid_metrics = evaluate_model(model, dl_valid, device)
        scheduler.step(valid_metrics["rmse"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)) if batch_losses else np.nan,
                "train_r": train_metrics["r"],
                "train_rmse": train_metrics["rmse"],
                "valid_r": valid_metrics["r"],
                "valid_rmse": valid_metrics["rmse"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        if valid_metrics["rmse"] < best_val_rmse:
            best_val_rmse = valid_metrics["rmse"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    y_tr, p_tr, m_tr = evaluate_model(model, dl_train, device)
    y_va, p_va, m_va = evaluate_model(model, dl_valid, device)
    y_te, p_te, m_te = evaluate_model(model, dl_test, device)

    seed_dir = out_dir / f"model_seed_{model_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(history).to_csv(seed_dir / "training_curve.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "y_true": y_te,
            "y_pred": p_te,
            "abs_error": np.abs(y_te - p_te),
            "rel_error_percent": np.where(
                np.abs(y_te) > 1e-12,
                np.abs(y_te - p_te) / np.abs(y_te) * 100,
                np.nan,
            ),
        }
    ).to_csv(seed_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    torch.save(best_state, seed_dir / "best_model.pt")

    fig, ax = plt.subplots(figsize=(7, 5))
    hist_df = pd.DataFrame(history)
    ax.plot(hist_df["epoch"], hist_df["train_rmse"], label="train_rmse")
    ax.plot(hist_df["epoch"], hist_df["valid_rmse"], label="valid_rmse")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Training curve (seed={model_seed})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(seed_dir / "training_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "model_seed": model_seed,
        "best_valid_rmse": best_val_rmse,
        "train_metrics": m_tr,
        "valid_metrics": m_va,
        "test_metrics": m_te,
        "test_pred": p_te,
        "test_true": y_te,
    }


def ensemble_seed_runs(runs):
    y_true = runs[0]["test_true"]
    pred_matrix = np.column_stack([item["test_pred"] for item in runs])
    pred_mean = pred_matrix.mean(axis=1)
    metrics = regression_metrics(y_true, pred_mean)
    return {
        "y_true": y_true,
        "pred_mean": pred_mean,
        "pred_all": pred_matrix,
        "metrics": metrics,
    }


def load_fd_fa_y(fd_path: str, fa_path: str, y_path: str):
    fd = np.asarray(np.load(fd_path))
    fa = np.asarray(np.load(fa_path))
    y = np.asarray(np.load(y_path)).reshape(-1)
    if fd.shape[0] != fa.shape[0] or fd.shape[0] != y.shape[0]:
        raise ValueError(f"Sample count mismatch: fd={fd.shape}, fa={fa.shape}, y={y.shape}")
    if fd.shape[1] != 1024 or fa.shape[1] != 1024:
        raise ValueError(f"Expect 1024-bit fd/fa, got {fd.shape} and {fa.shape}")
    return fd.astype(np.int64), fa.astype(np.int64), y.astype(np.float32)


def split_dataset(
    fd,
    fa,
    y,
    split_method: str,
    split_seed: int,
    test_size: float,
    valid_fraction_of_trainval: float,
):
    x_concat = np.hstack([fd, fa])
    if split_method == "hspxy":
        _, _, _, _, trainval_idx, test_idx = hspxy_split_gao(
            x_concat,
            y,
            test_size=test_size,
            fp_length=1024,
        )
        train_idx_local, valid_idx_local = train_test_split(
            np.arange(len(trainval_idx)),
            test_size=valid_fraction_of_trainval,
            random_state=split_seed,
            shuffle=True,
        )
        train_idx = trainval_idx[train_idx_local]
        valid_idx = trainval_idx[valid_idx_local]
    elif split_method == "random":
        all_idx = np.arange(len(y))
        trainval_idx, test_idx = train_test_split(
            all_idx,
            test_size=test_size,
            random_state=split_seed,
            shuffle=True,
        )
        train_idx, valid_idx = train_test_split(
            trainval_idx,
            test_size=valid_fraction_of_trainval,
            random_state=split_seed,
            shuffle=True,
        )
    else:
        raise ValueError("split_method must be one of: hspxy, random")

    return {
        "train_idx": np.asarray(train_idx),
        "valid_idx": np.asarray(valid_idx),
        "test_idx": np.asarray(test_idx),
        "fd_train": fd[train_idx],
        "fa_train": fa[train_idx],
        "y_train": y[train_idx],
        "fd_valid": fd[valid_idx],
        "fa_valid": fa[valid_idx],
        "y_valid": y[valid_idx],
        "fd_test": fd[test_idx],
        "fa_test": fa[test_idx],
        "y_test": y[test_idx],
    }


def save_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str, path: Path):
    upper = float(max(np.max(y_true), np.max(y_pred)) + 1.0)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=40, alpha=0.65)
    ax.plot([0, upper], [0, upper], ls="--", color="gray")
    metrics = regression_metrics(y_true, y_pred)
    ax.set_xlabel("Experimental PCE (%)")
    ax.set_ylabel("Predicted PCE (%)")
    ax.set_title(
        f"{title}\n"
        f"r={metrics['r']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=600, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_combined_scatter_figure(
    y_true_single: np.ndarray,
    y_pred_single: np.ndarray,
    y_true_ens: np.ndarray,
    y_pred_ens: np.ndarray,
    out_path: Path,
):
    single_metrics = regression_metrics(y_true_single, y_pred_single)
    ensemble_metrics = regression_metrics(y_true_ens, y_pred_ens)
    upper = float(
        max(
            np.max(y_true_single),
            np.max(y_pred_single),
            np.max(y_true_ens),
            np.max(y_pred_ens),
        )
        + 1.0
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.0), constrained_layout=True)
    panels = [
        (
            axes[0],
            y_true_single,
            y_pred_single,
            "Best single TextCNN(FPtand) model",
            single_metrics,
            "(a)",
        ),
        (
            axes[1],
            y_true_ens,
            y_pred_ens,
            "Ensemble TextCNN(FPtand) prediction",
            ensemble_metrics,
            "(b)",
        ),
    ]

    for ax, y_true, y_pred, title, metrics, panel_label in panels:
        ax.scatter(y_true, y_pred, s=40, alpha=0.65)
        ax.plot([0, upper], [0, upper], ls="--", color="gray")
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_xlabel("Experimental PCE (%)")
        ax.set_ylabel("Predicted PCE (%)")
        ax.set_title(
            f"{title}\n"
            f"r={metrics['r']:.4f}, RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}"
        )
        ax.grid(True, alpha=0.3)
        ax.text(
            0.02,
            0.98,
            panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
        )

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def save_top_pce_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_csv: Path,
    top_n: int = 10,
    threshold: float = 16.0,
):
    df = pd.DataFrame({"true_pce": y_true, "pred_pce": y_pred})
    df["diff"] = df["pred_pce"] - df["true_pce"]
    df["abs_error"] = np.abs(df["diff"])
    df["rel_error_percent"] = np.where(
        np.abs(df["true_pce"]) > 1e-12,
        df["abs_error"] / np.abs(df["true_pce"]) * 100,
        np.nan,
    )

    high = df[df["true_pce"] > threshold].copy().sort_values("true_pce", ascending=False)
    if len(high) < top_n:
        fill = df.sort_values("true_pce", ascending=False).head(top_n)
        high = pd.concat([high, fill], axis=0).drop_duplicates().head(top_n)
    else:
        high = high.head(top_n)

    high.to_csv(out_csv, index=False, encoding="utf-8-sig")


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd-path", required=True)
    parser.add_argument("--fa-path", required=True)
    parser.add_argument("--y-path", required=True)
    parser.add_argument("--output-dir", default="gao_textcnn_fptand_run")
    parser.add_argument("--profile", choices=["strict_gao", "strong"], default="strong")
    parser.add_argument("--split-method", choices=["hspxy", "random"], default="hspxy")
    parser.add_argument("--split-seed", type=int, default=12)
    parser.add_argument("--model-seeds", type=str, default="12,22,32")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--valid-fraction-of-trainval", type=float, default=0.125)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--kernel-sizes", type=str, default="3,5,7")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--loss", choices=["mse", "huber"], default="huber")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def build_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        profile=args.profile,
        split_method=args.split_method,
        split_seed=args.split_seed,
        model_seeds=parse_int_list(args.model_seeds),
        test_size=args.test_size,
        valid_fraction_of_trainval=args.valid_fraction_of_trainval,
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
    )


def run_training(cfg: TrainConfig, fd: np.ndarray, fa: np.ndarray, y: np.ndarray, out_dir: Path):
    split = split_dataset(
        fd,
        fa,
        y,
        cfg.split_method,
        cfg.split_seed,
        cfg.test_size,
        cfg.valid_fraction_of_trainval,
    )

    seq_train = encode_fptand_from_fd_fa(split["fd_train"], split["fa_train"], max_len=cfg.max_len)
    seq_valid = encode_fptand_from_fd_fa(split["fd_valid"], split["fa_valid"], max_len=cfg.max_len)
    seq_test = encode_fptand_from_fd_fa(split["fd_test"], split["fa_test"], max_len=cfg.max_len)

    np.savez_compressed(
        out_dir / "split_indices.npz",
        train_idx=split["train_idx"],
        valid_idx=split["valid_idx"],
        test_idx=split["test_idx"],
    )

    seed_runs = []
    for model_seed in cfg.model_seeds:
        seed_runs.append(
            train_one_seed(
                seq_train,
                split["y_train"],
                seq_valid,
                split["y_valid"],
                seq_test,
                split["y_test"],
                cfg,
                model_seed,
                out_dir,
            )
        )

    return split, seed_runs


def save_run_outputs(
    cfg: TrainConfig,
    out_dir: Path,
    y: np.ndarray,
    split,
    seed_runs,
):
    best_single = min(seed_runs, key=lambda item: item["best_valid_rmse"])
    ensemble = ensemble_seed_runs(seed_runs)

    pd.DataFrame(
        {
            "y_true": ensemble["y_true"],
            "y_pred_ensemble": ensemble["pred_mean"],
        }
    ).to_csv(out_dir / "ensemble_test_predictions.csv", index=False, encoding="utf-8-sig")

    save_scatter(
        best_single["test_true"],
        best_single["test_pred"],
        "Best single TextCNN(FPtand) model",
        out_dir / "best_single_scatter.png",
    )
    save_scatter(
        ensemble["y_true"],
        ensemble["pred_mean"],
        "Ensemble TextCNN(FPtand) prediction",
        out_dir / "ensemble_scatter.png",
    )
    save_combined_scatter_figure(
        best_single["test_true"],
        best_single["test_pred"],
        ensemble["y_true"],
        ensemble["pred_mean"],
        out_dir / "combined_scatter_figure.png",
    )
    save_top_pce_errors(
        ensemble["y_true"],
        ensemble["pred_mean"],
        out_dir / "top10_high_pce_errors.csv",
        top_n=10,
        threshold=16.0,
    )

    per_seed_rows = []
    for item in seed_runs:
        row = {
            "model_seed": item["model_seed"],
            "best_valid_rmse": item["best_valid_rmse"],
        }
        row.update({f"test_{k}": v for k, v in item["test_metrics"].items()})
        row.update({f"valid_{k}": v for k, v in item["valid_metrics"].items()})
        per_seed_rows.append(row)
    pd.DataFrame(per_seed_rows).to_csv(
        out_dir / "per_seed_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "config": asdict(cfg),
        "n_samples_total": int(len(y)),
        "n_train": int(len(split["y_train"])),
        "n_valid": int(len(split["y_valid"])),
        "n_test": int(len(split["y_test"])),
        "best_single_model_seed": int(best_single["model_seed"]),
        "best_single_test_metrics": best_single["test_metrics"],
        "ensemble_test_metrics": ensemble["metrics"],
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("=" * 88)
    print("Gao TextCNN(FPtand) run finished")
    print(f"Output dir: {out_dir.resolve()}")
    print(f"Best single seed: {best_single['model_seed']}")
    print("Best single test metrics:", best_single["test_metrics"])
    print("Ensemble test metrics:", ensemble["metrics"])
    print("=" * 88)


def main():
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_config(args)
    fd, fa, y = load_fd_fa_y(args.fd_path, args.fa_path, args.y_path)
    split, seed_runs = run_training(cfg, fd, fa, y, out_dir)
    save_run_outputs(cfg, out_dir, y, split, seed_runs)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Shared utilities for the OSC PCE reviewer-revision workflow."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(pearsonr(y_true, y_pred)[0])


def safe_spearmanr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) < 2 or np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    return float(spearmanr(y_true, y_pred)[0])


def calibration_parameters(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Fit observed = intercept + slope * predicted by ordinary least squares."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) < 2 or np.std(y_pred) < 1e-12:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(y_pred, y_true, deg=1)
    return float(intercept), float(slope)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    intercept, slope = calibration_parameters(y_true, y_pred)
    residual = y_pred - y_true
    return {
        "r": safe_pearsonr(y_true, y_pred),
        "spearman_rho": safe_spearmanr(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mean_signed_error": float(np.mean(residual)),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
    }


def ranking_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_fractions: Sequence[float] = (0.10, 0.20),
    high_threshold: float = 16.0,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = len(y_true)
    if n == 0:
        return pd.DataFrame()
    true_order = np.argsort(-y_true, kind="stable")
    pred_order = np.argsort(-y_pred, kind="stable")
    high_set = set(np.flatnonzero(y_true >= high_threshold).tolist())
    rows: list[dict[str, float | int | str]] = []
    for frac in top_fractions:
        k = max(1, min(n, int(math.ceil(n * float(frac)))))
        true_top = set(true_order[:k].tolist())
        pred_top = set(pred_order[:k].tolist())
        hit = len(true_top & pred_top)
        precision = hit / k
        recall = hit / len(true_top) if true_top else float("nan")
        prevalence = len(true_top) / n
        enrichment = precision / prevalence if prevalence > 0 else float("nan")
        high_hit = len(high_set & pred_top)
        high_recall = high_hit / len(high_set) if high_set else float("nan")
        rows.append(
            {
                "top_fraction": float(frac),
                "k": int(k),
                "top_k_hits": int(hit),
                "top_k_precision": float(precision),
                "top_k_recall": float(recall),
                "enrichment_factor": float(enrichment),
                "high_pce_threshold": float(high_threshold),
                "n_high_pce": int(len(high_set)),
                "high_pce_recall_in_predicted_top_k": float(high_recall),
            }
        )
    return pd.DataFrame(rows)


def residual_table(y_true: np.ndarray, y_pred: np.ndarray, indices: np.ndarray | None = None) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if indices is None:
        indices = np.arange(len(y_true), dtype=int)
    df = pd.DataFrame(
        {
            "array_index": np.asarray(indices, dtype=int),
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    df["residual_pred_minus_true"] = df["y_pred"] - df["y_true"]
    df["absolute_error"] = np.abs(df["residual_pred_minus_true"])
    df["squared_error"] = df["residual_pred_minus_true"] ** 2
    df["relative_error_percent"] = np.where(
        np.abs(df["y_true"]) > 1e-12,
        df["absolute_error"] / np.abs(df["y_true"]) * 100.0,
        np.nan,
    )
    return df


def stratified_error_table(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    df = residual_table(y_true, y_pred)
    try:
        df["pce_bin"] = pd.qcut(df["y_true"], q=n_bins, duplicates="drop")
    except ValueError:
        df["pce_bin"] = "all"
    rows = []
    for bin_name, sub in df.groupby("pce_bin", observed=True):
        m = regression_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy())
        rows.append(
            {
                "pce_bin": str(bin_name),
                "n": int(len(sub)),
                "experimental_min": float(sub["y_true"].min()),
                "experimental_max": float(sub["y_true"].max()),
                **m,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 2026,
    confidence: float = 0.95,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) != len(y_pred) or len(y_true) < 2:
        raise ValueError("Bootstrap requires equal-length arrays with at least two samples.")
    rng = np.random.default_rng(seed)
    metric_names = list(regression_metrics(y_true, y_pred).keys())
    draws: dict[str, list[float]] = {name: [] for name in metric_names}
    n = len(y_true)
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        m = regression_metrics(y_true[idx], y_pred[idx])
        for name, value in m.items():
            if np.isfinite(value):
                draws[name].append(float(value))
    alpha = (1.0 - confidence) / 2.0
    point = regression_metrics(y_true, y_pred)
    rows = []
    for name in metric_names:
        arr = np.asarray(draws[name], dtype=float)
        rows.append(
            {
                "metric": name,
                "point_estimate": point[name],
                "bootstrap_n_valid": int(len(arr)),
                "percentile_low": float(np.quantile(arr, alpha)) if len(arr) else np.nan,
                "percentile_high": float(np.quantile(arr, 1.0 - alpha)) if len(arr) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def empirical_summary(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "sd": np.nan,
            "p2_5": np.nan,
            "p25": np.nan,
            "median": np.nan,
            "p75": np.nan,
            "p97_5": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "p2_5": float(np.quantile(arr, 0.025)),
        "p25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p97_5": float(np.quantile(arr, 0.975)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def binary_row_hashes(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    packed = np.packbits(arr.astype(np.uint8), axis=1)
    return np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in packed], dtype=object)


def paired_row_hashes(fd: np.ndarray, fa: np.ndarray) -> np.ndarray:
    return binary_row_hashes(np.hstack([np.asarray(fd), np.asarray(fa)]))


def sequence_row_hashes(seq: np.ndarray) -> np.ndarray:
    seq = np.asarray(seq, dtype=np.int64)
    return np.asarray([hashlib.sha256(row.tobytes()).hexdigest() for row in seq], dtype=object)


def factorize_hashes(hashes: Sequence[str]) -> np.ndarray:
    codes, _ = pd.factorize(np.asarray(hashes, dtype=object), sort=True)
    return codes.astype(np.int64)


def tanimoto_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    inter = a @ b.T
    a_sum = np.sum(a, axis=1, keepdims=True)
    b_sum = np.sum(b, axis=1, keepdims=True).T
    union = a_sum + b_sum - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)


def paired_similarity_matrix(fd_a: np.ndarray, fa_a: np.ndarray, fd_b: np.ndarray, fa_b: np.ndarray) -> np.ndarray:
    return 0.5 * (
        tanimoto_similarity_matrix(fd_a, fd_b) + tanimoto_similarity_matrix(fa_a, fa_b)
    )


def nearest_similarity_audit(
    test_fd: np.ndarray,
    test_fa: np.ndarray,
    train_fd: np.ndarray,
    train_fa: np.ndarray,
    test_indices: np.ndarray | None = None,
    train_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    if len(train_fd) == 0:
        raise ValueError("Training/reference set is empty.")
    d_sim = tanimoto_similarity_matrix(test_fd, train_fd)
    a_sim = tanimoto_similarity_matrix(test_fa, train_fa)
    p_sim = 0.5 * (d_sim + a_sim)
    if test_indices is None:
        test_indices = np.arange(len(test_fd))
    if train_indices is None:
        train_indices = np.arange(len(train_fd))
    rows = []
    for i in range(len(test_fd)):
        d_j = int(np.argmax(d_sim[i]))
        a_j = int(np.argmax(a_sim[i]))
        p_j = int(np.argmax(p_sim[i]))
        rows.append(
            {
                "test_array_index": int(test_indices[i]),
                "nearest_donor_train_index": int(train_indices[d_j]),
                "nearest_donor_similarity": float(d_sim[i, d_j]),
                "nearest_acceptor_train_index": int(train_indices[a_j]),
                "nearest_acceptor_similarity": float(a_sim[i, a_j]),
                "nearest_pair_train_index": int(train_indices[p_j]),
                "nearest_pair_similarity": float(p_sim[i, p_j]),
            }
        )
    return pd.DataFrame(rows)


def encode_fptand(
    fd: np.ndarray,
    fa: np.ndarray,
    max_len: int = 200,
    mode: str = "legacy",
) -> tuple[np.ndarray, pd.DataFrame, int]:
    """Encode fingerprints and return sequence, per-sample audit, and vocabulary size.

    legacy: donor and acceptor share token IDs 1..1024; donor tokens precede acceptor tokens.
    role_aware: donor IDs 1..1024, separator 1025, acceptor IDs 1026..2049,
    with deterministic balanced truncation when the full sequence exceeds max_len.
    """
    fd = np.asarray(fd)
    fa = np.asarray(fa)
    if fd.shape != fa.shape or fd.ndim != 2 or fd.shape[1] != 1024:
        raise ValueError(f"Expected equal (n,1024) donor/acceptor arrays, got {fd.shape} and {fa.shape}")
    if max_len < 3:
        raise ValueError("max_len must be at least 3.")
    mode = str(mode).lower()
    if mode not in {"legacy", "role_aware"}:
        raise ValueError("mode must be 'legacy' or 'role_aware'.")
    seq = np.zeros((len(fd), max_len), dtype=np.int64)
    audit_rows = []
    for i in range(len(fd)):
        donor_bits = np.flatnonzero(fd[i] == 1).astype(np.int64)
        acceptor_bits = np.flatnonzero(fa[i] == 1).astype(np.int64)
        d_count = int(len(donor_bits))
        a_count = int(len(acceptor_bits))
        if mode == "legacy":
            full = np.concatenate([donor_bits + 1, acceptor_bits + 1])
            kept = full[:max_len]
            d_kept = min(d_count, max_len)
            a_kept = max(0, min(a_count, max_len - d_kept))
            separator_kept = 0
            vocab_size = 1024
        else:
            budget = max_len - 1
            if d_count + a_count <= budget:
                d_kept, a_kept = d_count, a_count
            elif d_count == 0:
                d_kept, a_kept = 0, min(a_count, budget)
            elif a_count == 0:
                d_kept, a_kept = min(d_count, budget), 0
            else:
                d_kept = int(round(budget * d_count / (d_count + a_count)))
                d_kept = max(1, min(d_count, d_kept))
                a_kept = min(a_count, budget - d_kept)
                if a_kept < 1:
                    a_kept = 1
                    d_kept = min(d_count, budget - 1)
                remaining = budget - d_kept - a_kept
                if remaining > 0:
                    add_d = min(remaining, d_count - d_kept)
                    d_kept += add_d
                    remaining -= add_d
                    a_kept += min(remaining, a_count - a_kept)
            donor_tokens = donor_bits[:d_kept] + 1
            acceptor_tokens = acceptor_bits[:a_kept] + 1026
            kept = np.concatenate([donor_tokens, np.asarray([1025], dtype=np.int64), acceptor_tokens])
            separator_kept = 1
            vocab_size = 2049
        seq[i, : len(kept)] = kept
        audit_rows.append(
            {
                "array_index": i,
                "encoding_mode": mode,
                "donor_active_bits": d_count,
                "acceptor_active_bits": a_count,
                "total_active_bits": d_count + a_count,
                "full_sequence_length": d_count + a_count + (1 if mode == "role_aware" else 0),
                "max_len": int(max_len),
                "truncated": bool((d_count + a_count + (1 if mode == "role_aware" else 0)) > max_len),
                "donor_tokens_retained": int(d_kept),
                "acceptor_tokens_retained": int(a_kept),
                "separator_retained": int(separator_kept),
                "donor_tokens_dropped": int(d_count - d_kept),
                "acceptor_tokens_dropped": int(a_count - a_kept),
                "encoded_nonzero_length": int(len(kept)),
            }
        )
    return seq, pd.DataFrame(audit_rows), vocab_size


def random_group_split(
    n_samples: int,
    groups: np.ndarray,
    test_size: float,
    valid_fraction_of_trainval: float,
    seed: int,
) -> dict[str, np.ndarray]:
    groups = np.asarray(groups)
    all_idx = np.arange(n_samples)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_local, test_local = next(gss.split(all_idx, groups=groups))
    trainval_idx = all_idx[trainval_local]
    test_idx = all_idx[test_local]
    trainval_groups = groups[trainval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=valid_fraction_of_trainval, random_state=seed + 100003)
    train_local, valid_local = next(gss2.split(trainval_idx, groups=trainval_groups))
    return {
        "train_idx": trainval_idx[train_local],
        "valid_idx": trainval_idx[valid_local],
        "test_idx": test_idx,
    }


def structure_ks_group_split(
    fd: np.ndarray,
    fa: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    valid_fraction_of_trainval: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Target-blind Kennard-Stone-like split at the representation-group level."""
    fd = np.asarray(fd)
    fa = np.asarray(fa)
    groups = np.asarray(groups)
    unique_groups, first_positions = np.unique(groups, return_index=True)
    order = np.argsort(first_positions)
    unique_groups = unique_groups[order]
    reps = np.asarray([np.flatnonzero(groups == g)[0] for g in unique_groups], dtype=int)
    group_sizes = np.asarray([np.sum(groups == g) for g in unique_groups], dtype=int)
    if len(unique_groups) < 3:
        raise ValueError("At least three representation groups are required for train/valid/test splitting.")
    sim = paired_similarity_matrix(fd[reps], fa[reps], fd[reps], fa[reps])
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    rng = np.random.default_rng(seed)
    max_dist = float(np.max(dist))
    candidate_pairs = np.argwhere(np.isclose(dist, max_dist))
    candidate_pairs = candidate_pairs[candidate_pairs[:, 0] < candidate_pairs[:, 1]]
    if len(candidate_pairs) == 0:
        first, second = 0, 1
    else:
        first, second = candidate_pairs[int(rng.integers(0, len(candidate_pairs)))]
    selected = [int(first), int(second)]
    selected_mask = np.zeros(len(unique_groups), dtype=bool)
    selected_mask[selected] = True
    target_trainval_n = int(round((1.0 - test_size) * len(fd)))
    selected_n = int(group_sizes[selected].sum())
    while selected_n < target_trainval_n and int(selected_mask.sum()) < len(unique_groups) - 1:
        remaining = np.flatnonzero(~selected_mask)
        min_dist = np.min(dist[np.ix_(remaining, np.asarray(selected, dtype=int))], axis=1)
        best_value = np.max(min_dist)
        candidates = remaining[np.isclose(min_dist, best_value)]
        chosen = int(candidates[int(rng.integers(0, len(candidates)))])
        selected.append(chosen)
        selected_mask[chosen] = True
        selected_n += int(group_sizes[chosen])
    trainval_groups = set(unique_groups[np.asarray(selected, dtype=int)].tolist())
    trainval_idx = np.asarray([i for i, g in enumerate(groups) if g in trainval_groups], dtype=int)
    test_idx = np.asarray([i for i, g in enumerate(groups) if g not in trainval_groups], dtype=int)
    if len(test_idx) == 0:
        raise RuntimeError("Structure-only split produced an empty test set; adjust test_size.")
    trainval_groups_arr = groups[trainval_idx]
    gss = GroupShuffleSplit(n_splits=1, test_size=valid_fraction_of_trainval, random_state=seed + 200003)
    train_local, valid_local = next(gss.split(trainval_idx, groups=trainval_groups_arr))
    return {
        "train_idx": trainval_idx[train_local],
        "valid_idx": trainval_idx[valid_local],
        "test_idx": test_idx,
    }


def validate_group_disjoint(split: dict[str, np.ndarray], groups: np.ndarray) -> dict[str, int]:
    groups = np.asarray(groups)
    sets = {name: set(groups[np.asarray(split[f"{name}_idx"], dtype=int)].tolist()) for name in ["train", "valid", "test"]}
    overlaps = {
        "train_valid_group_overlap": len(sets["train"] & sets["valid"]),
        "train_test_group_overlap": len(sets["train"] & sets["test"]),
        "valid_test_group_overlap": len(sets["valid"] & sets["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Representation groups cross split boundaries: {overlaps}")
    return overlaps


def save_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "cwd": os.getcwd(),
    }
    for name in ["numpy", "pandas", "scipy", "sklearn", "rdkit", "torch", "shap", "joblib", "matplotlib"]:
        try:
            module = __import__(name)
            report[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover
            report[name] = f"not available: {exc}"
    return report

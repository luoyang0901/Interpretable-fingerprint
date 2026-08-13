#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reviewer-revision conventional baseline for OSC donor-acceptor PCE.

Major changes implemented here:
1. A strict 20-pair scaffold-disjoint hold-out is constructed. Any remaining
   row sharing a donor or acceptor scaffold with the 20 test pairs is placed in
   a quarantine set and is not used for model development.
2. The ordinary fixed train/validation/test partition is target-blind and is
   based only on fingerprint geometry while keeping identical paired
   representations in one subset.
3. Model selection uses validation RMSE only. The test set is evaluated once.
4. Mutual-information feature selection is fitted only on the training subset.
5. Calibration, residual, ranking, PCE-stratified errors, similarity audit, and
   bootstrap intervals are archived as machine-readable files.
6. A predefined no-augmentation Random Forest pipeline is archived for the
   single-model SHAP and structural back-mapping workflow.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVR

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception as exc:  # pragma: no cover
    raise ImportError("RDKit is required. Install it from conda-forge.") from exc

from common_utils import (
    binary_row_hashes,
    bootstrap_metric_intervals,
    empirical_summary,
    environment_report,
    factorize_hashes,
    nearest_similarity_audit,
    paired_row_hashes,
    ranking_metrics,
    regression_metrics,
    residual_table,
    save_json,
    stratified_error_table,
    structure_ks_group_split,
    validate_group_disjoint,
)

MODEL_ORDER = ["RandomForest", "GradientBoosting", "SVR", "Ridge", "ElasticNet"]


@dataclass
class BaselineConfig:
    da_csv: str
    output_dir: str
    aggregation_rule: str = "maximum"
    scaffold_test_size: int = 20
    scaffold_seed: int = 42
    fixed_split_seed: int = 12
    radius: int = 3
    n_bits: int = 1024
    mi_k: int = 600
    test_size: float = 0.20
    valid_fraction_of_trainval: float = 0.125
    bootstrap_repeats: int = 2000
    high_pce_threshold: float = 16.0
    run_nested_baseline: bool = False
    nested_split_seeds: str = "0,1,2,3,4,5,6,7,8,9"
    nested_model_seeds: str = "0,1,2,3,4"


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def guess_columns(df: pd.DataFrame) -> Tuple[str, str, str]:
    lower = {str(c).lower().strip(): c for c in df.columns}
    donor_candidates = ["donor_smiles", "donor smiles", "donor", "d_smiles", "d"]
    acceptor_candidates = ["acceptor_smiles", "acceptor smiles", "acceptor", "a_smiles", "a"]
    pce_candidates = ["pce", "power conversion efficiency", "pce (%)", "efficiency"]

    def pick(candidates: list[str], contains: str) -> Optional[str]:
        for candidate in candidates:
            if candidate in lower:
                return lower[candidate]
        for col in df.columns:
            if contains in str(col).lower():
                return col
        return None

    donor = pick(donor_candidates, "donor")
    acceptor = pick(acceptor_candidates, "acceptor")
    pce = pick(pce_candidates, "pce")
    if donor is None or acceptor is None or pce is None:
        raise ValueError(f"Could not identify donor, acceptor and PCE columns in {list(df.columns)}")
    return donor, acceptor, pce


def load_structure_records(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    donor_col, acceptor_col, pce_col = guess_columns(raw)
    df = raw[[donor_col, acceptor_col, pce_col]].copy()
    df.columns = ["donor_smiles", "acceptor_smiles", "PCE"]
    df["donor_smiles"] = df["donor_smiles"].astype(str).str.strip()
    df["acceptor_smiles"] = df["acceptor_smiles"].astype(str).str.strip()
    df["PCE"] = pd.to_numeric(df["PCE"], errors="coerce")
    df = df.dropna(subset=["PCE"])
    df = df[(df["donor_smiles"] != "") & (df["acceptor_smiles"] != "")].copy()
    valid = []
    for i, row in df.iterrows():
        d_ok = Chem.MolFromSmiles(row["donor_smiles"]) is not None
        a_ok = Chem.MolFromSmiles(row["acceptor_smiles"]) is not None
        valid.append(bool(d_ok and a_ok))
    df = df.loc[np.asarray(valid, dtype=bool)].reset_index(drop=True)
    df["pair_key"] = df["donor_smiles"] + "||" + df["acceptor_smiles"]
    return df


def aggregate_pair_records(records: pd.DataFrame, rule: str) -> pd.DataFrame:
    rule = str(rule).lower()
    aliases = {"max": "maximum", "mean": "mean", "median": "median", "maximum": "maximum"}
    if rule not in aliases:
        raise ValueError("aggregation rule must be maximum, mean, or median")
    rule = aliases[rule]
    aggfunc = {"maximum": "max", "mean": "mean", "median": "median"}[rule]
    grouped = (
        records.groupby(["pair_key", "donor_smiles", "acceptor_smiles"], as_index=False)
        .agg(PCE=("PCE", aggfunc), n_source_records=("PCE", "size"), PCE_min=("PCE", "min"), PCE_max=("PCE", "max"))
        .reset_index(drop=True)
    )
    grouped["aggregation_rule"] = rule
    return grouped


_MORGAN_GENERATORS: dict[tuple[int, int], Any] = {}


def morgan_bits(smiles: str, radius: int, n_bits: int) -> np.ndarray:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    key = (int(radius), int(n_bits))
    if key not in _MORGAN_GENERATORS:
        try:
            _MORGAN_GENERATORS[key] = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        except Exception:
            _MORGAN_GENERATORS[key] = None
    if _MORGAN_GENERATORS[key] is not None:
        fp = _MORGAN_GENERATORS[key].GetFingerprint(mol)
    else:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(np.int8)


def build_pair_fingerprints(
    df: pd.DataFrame, radius: int, n_bits: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    donor_cache: dict[str, np.ndarray] = {}
    acceptor_cache: dict[str, np.ndarray] = {}
    fd_rows, fa_rows = [], []
    for row in df.itertuples(index=False):
        d = str(row.donor_smiles)
        a = str(row.acceptor_smiles)
        if d not in donor_cache:
            donor_cache[d] = morgan_bits(d, radius, n_bits)
        if a not in acceptor_cache:
            acceptor_cache[a] = morgan_bits(a, radius, n_bits)
        fd_rows.append(donor_cache[d])
        fa_rows.append(acceptor_cache[a])
    fd = np.vstack(fd_rows).astype(np.int8)
    fa = np.vstack(fa_rows).astype(np.int8)
    X = np.hstack([fd, fa]).astype(np.float32)
    y = df["PCE"].to_numpy(dtype=float)
    feature_names = [f"fd_{i}" for i in range(n_bits)] + [f"fa_{i}" for i in range(n_bits)]
    return fd, fa, X, y, feature_names


def scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    if scaffold:
        return scaffold
    return "ACYCLIC:" + Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def add_scaffolds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["donor_scaffold"] = [scaffold_smiles(x) for x in out["donor_smiles"]]
    out["acceptor_scaffold"] = [scaffold_smiles(x) for x in out["acceptor_smiles"]]
    out["pair_scaffold_key"] = out["donor_scaffold"] + "||" + out["acceptor_scaffold"]
    return out


def strict_scaffold_holdout(df: pd.DataFrame, n_test: int, seed: int) -> dict[str, np.ndarray]:
    """Select exactly n_test pairs and quarantine all scaffold-overlapping rows.

    Development rows share neither donor scaffold nor acceptor scaffold with any
    test row. The greedy objective minimizes the number of quarantined rows.
    """
    rng = np.random.default_rng(seed)
    n = len(df)
    if n_test >= n:
        raise ValueError("scaffold test size must be smaller than the dataset")
    d_scaf = df["donor_scaffold"].astype(str).to_numpy()
    a_scaf = df["acceptor_scaffold"].astype(str).to_numpy()
    selected: list[int] = []
    selected_d: set[str] = set()
    selected_a: set[str] = set()
    currently_excluded = np.zeros(n, dtype=bool)
    for _ in range(n_test):
        candidates = [i for i in range(n) if i not in selected and d_scaf[i] not in selected_d and a_scaf[i] not in selected_a]
        if not candidates:
            raise RuntimeError(
                f"Only {len(selected)} strict scaffold-disjoint test pairs could be selected. "
                "Reduce --scaffold-test-size or inspect scaffold diversity."
            )
        costs = []
        for i in candidates:
            newly = ((d_scaf == d_scaf[i]) | (a_scaf == a_scaf[i])) & (~currently_excluded)
            costs.append(int(np.sum(newly)))
        min_cost = min(costs)
        best = [c for c, cost in zip(candidates, costs) if cost == min_cost]
        chosen = int(best[int(rng.integers(0, len(best)))])
        selected.append(chosen)
        selected_d.add(d_scaf[chosen])
        selected_a.add(a_scaf[chosen])
        currently_excluded |= (d_scaf == d_scaf[chosen]) | (a_scaf == a_scaf[chosen])
    test_idx = np.asarray(sorted(selected), dtype=int)
    is_test = np.zeros(n, dtype=bool)
    is_test[test_idx] = True
    overlap_mask = np.isin(d_scaf, list(selected_d)) | np.isin(a_scaf, list(selected_a))
    quarantine_idx = np.flatnonzero(overlap_mask & (~is_test)).astype(int)
    development_idx = np.flatnonzero(~overlap_mask).astype(int)
    if set(d_scaf[test_idx]) & set(d_scaf[development_idx]):
        raise AssertionError("Donor scaffold leakage detected")
    if set(a_scaf[test_idx]) & set(a_scaf[development_idx]):
        raise AssertionError("Acceptor scaffold leakage detected")
    return {
        "development_idx": development_idx,
        "test_idx": test_idx,
        "quarantine_idx": quarantine_idx,
    }


def build_models(seed: int) -> dict[str, Any]:
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=seed,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.03,
            min_samples_split=3,
            random_state=seed,
        ),
        "SVR": SVR(kernel="rbf", C=10, gamma="scale"),
        "Ridge": Ridge(alpha=0.05),
        "ElasticNet": ElasticNet(alpha=0.005, l1_ratio=0.3, max_iter=5000, random_state=seed),
    }


def augment_train_only(X_train: np.ndarray, y_train: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, 0.003, size=X_train.shape)
    X_noise = np.clip(X_train + noise, 0.0, 1.0)
    scale = rng.uniform(0.98, 1.02, size=X_train.shape[1])
    X_scaled = np.clip(X_train * scale, 0.0, 1.0)
    return np.vstack([X_train, X_noise, X_scaled]), np.concatenate([y_train, y_train, y_train])


def fit_conventional_models(
    X: np.ndarray,
    y: np.ndarray,
    split: dict[str, np.ndarray],
    feature_names: list[str],
    mi_k: int,
    feature_selection_seed: int,
    model_seed: int,
    augment: bool = False,
) -> dict[str, Any]:
    train_idx = np.asarray(split["train_idx"], dtype=int)
    valid_idx = np.asarray(split["valid_idx"], dtype=int)
    test_idx = np.asarray(split["test_idx"], dtype=int)
    X_train, y_train = X[train_idx], y[train_idx]
    X_valid, y_valid = X[valid_idx], y[valid_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    if augment:
        X_fit, y_fit = augment_train_only(X_train, y_train, model_seed)
    else:
        X_fit, y_fit = X_train, y_train
    k = min(int(mi_k), X_fit.shape[1])
    selector = SelectKBest(
        score_func=lambda x_, y_: mutual_info_regression(x_, y_, random_state=feature_selection_seed),
        k=k,
    )
    X_fit_sel = selector.fit_transform(X_fit, y_fit)
    X_valid_sel = selector.transform(X_valid)
    X_test_sel = selector.transform(X_test)
    scaler = RobustScaler()
    X_fit_scaled = scaler.fit_transform(X_fit_sel)
    X_valid_scaled = scaler.transform(X_valid_sel)
    X_test_scaled = scaler.transform(X_test_sel)
    selected_indices = selector.get_support(indices=True)
    models = build_models(model_seed)
    results: dict[str, Any] = {}
    for name in MODEL_ORDER:
        model = clone(models[name])
        model.fit(X_fit_scaled, y_fit)
        valid_pred = model.predict(X_valid_scaled)
        test_pred = model.predict(X_test_scaled)
        results[name] = {
            "model": model,
            "valid_pred": valid_pred,
            "test_pred": test_pred,
            "valid_metrics": regression_metrics(y_valid, valid_pred),
            "test_metrics": regression_metrics(y_test, test_pred),
        }
    best_name = min(MODEL_ORDER, key=lambda name: (results[name]["valid_metrics"]["rmse"], MODEL_ORDER.index(name)))
    return {
        "selector": selector,
        "scaler": scaler,
        "selected_indices": selected_indices,
        "selected_feature_names": [feature_names[i] for i in selected_indices],
        "results": results,
        "best_model_name": best_name,
        "selection_rule": "lowest validation RMSE",
        "split": split,
        "augment": augment,
    }


def predict_pipeline(fitted: dict[str, Any], model_name: str, X_external: np.ndarray) -> np.ndarray:
    X_sel = fitted["selector"].transform(X_external)
    X_scaled = fitted["scaler"].transform(X_sel)
    return fitted["results"][model_name]["model"].predict(X_scaled)


def save_prediction_evaluation(
    out_dir: Path,
    stem: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    indices: np.ndarray,
    high_threshold: float,
    bootstrap_repeats: int | None = None,
) -> dict[str, float]:
    metrics = regression_metrics(y_true, y_pred)
    residual_table(y_true, y_pred, indices).to_csv(out_dir / f"{stem}_predictions.csv", index=False, encoding="utf-8-sig")
    stratified_error_table(y_true, y_pred).to_csv(out_dir / f"{stem}_pce_stratified_errors.csv", index=False, encoding="utf-8-sig")
    ranking_metrics(y_true, y_pred, high_threshold=high_threshold).to_csv(
        out_dir / f"{stem}_ranking_metrics.csv", index=False, encoding="utf-8-sig"
    )
    save_json(out_dir / f"{stem}_metrics.json", metrics)
    if bootstrap_repeats:
        bootstrap_metric_intervals(y_true, y_pred, n_bootstrap=bootstrap_repeats).to_csv(
            out_dir / f"{stem}_bootstrap_intervals.csv", index=False, encoding="utf-8-sig"
        )
    return metrics


def run_nested_baseline(
    X: np.ndarray,
    y: np.ndarray,
    fd: np.ndarray,
    fa: np.ndarray,
    development_idx: np.ndarray,
    pair_groups: np.ndarray,
    feature_names: list[str],
    cfg: BaselineConfig,
    out_dir: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    split_seeds = parse_int_list(cfg.nested_split_seeds)
    model_seeds = parse_int_list(cfg.nested_model_seeds)
    for split_seed in split_seeds:
        local_groups = pair_groups[development_idx]
        split_local = structure_ks_group_split(
            fd[development_idx], fa[development_idx], local_groups,
            cfg.test_size, cfg.valid_fraction_of_trainval, split_seed,
        )
        split = {key: development_idx[np.asarray(value, dtype=int)] for key, value in split_local.items()}
        for model_seed in model_seeds:
            fitted = fit_conventional_models(
                X, y, split, feature_names, cfg.mi_k, feature_selection_seed=split_seed,
                model_seed=model_seed, augment=False,
            )
            # Use the pre-specified Random Forest to separate model stochasticity cleanly.
            m = fitted["results"]["RandomForest"]["test_metrics"]
            rows.append({"split_seed": split_seed, "model_seed": model_seed, "model": "RandomForest", **m})
            print(f"Nested baseline split={split_seed}, model={model_seed} complete")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "nested_baseline_metrics.csv", index=False, encoding="utf-8-sig")
    summary_rows = []
    variance_rows = []
    for metric in ["r", "spearman_rho", "r2", "rmse", "mae", "mean_signed_error", "calibration_slope"]:
        split_means = df.groupby("split_seed")[metric].mean()
        within_vars = df.groupby("split_seed")[metric].var(ddof=1).fillna(0.0)
        stats = empirical_summary(split_means.to_numpy())
        summary_rows.append({"metric": metric, **stats})
        between = float(np.var(split_means.to_numpy(), ddof=1)) if len(split_means) > 1 else 0.0
        within = float(np.mean(within_vars.to_numpy())) if len(within_vars) else 0.0
        total = float(np.var(df[metric].to_numpy(), ddof=1)) if len(df) > 1 else 0.0
        variance_rows.append(
            {
                "metric": metric,
                "between_split_variance_of_split_means": between,
                "mean_within_split_model_seed_variance": within,
                "total_individual_run_variance": total,
                "between_fraction_of_between_plus_within": between / (between + within) if between + within > 0 else np.nan,
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "nested_baseline_split_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(variance_rows).to_csv(out_dir / "nested_baseline_variance_decomposition.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--da-csv", required=True, help="Raw or pair-level structure-resolved CSV.")
    p.add_argument("--output-dir", default="baseline_reviewer_revision")
    p.add_argument("--aggregation-rule", choices=["maximum", "mean", "median"], default="maximum")
    p.add_argument("--scaffold-test-size", type=int, default=20)
    p.add_argument("--scaffold-seed", type=int, default=42)
    p.add_argument("--fixed-split-seed", type=int, default=12)
    p.add_argument("--radius", type=int, default=3)
    p.add_argument("--n-bits", type=int, default=1024)
    p.add_argument("--mi-k", type=int, default=600)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--valid-fraction-of-trainval", type=float, default=0.125)
    p.add_argument("--bootstrap-repeats", type=int, default=2000)
    p.add_argument("--high-pce-threshold", type=float, default=16.0)
    p.add_argument("--run-nested-baseline", action="store_true")
    p.add_argument("--nested-split-seeds", default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--nested-model-seeds", default="0,1,2,3,4")
    args = p.parse_args()
    cfg = BaselineConfig(**vars(args))
    out = Path(cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "environment.json", environment_report())
    save_json(out / "config.json", asdict(cfg))

    records = load_structure_records(cfg.da_csv)
    records.to_csv(out / "structure_available_records.csv", index=False, encoding="utf-8-sig")
    parent = add_scaffolds(aggregate_pair_records(records, cfg.aggregation_rule))
    parent.insert(0, "array_index", np.arange(len(parent), dtype=int))
    parent.to_csv(out / "parent_pairs.csv", index=False, encoding="utf-8-sig")

    print(f"Structure-available records: {len(records)}; unique pairs: {len(parent)}")
    fd, fa, X, y, feature_names = build_pair_fingerprints(parent, cfg.radius, cfg.n_bits)
    np.save(out / "radius3_fd.npy", fd)
    np.save(out / "radius3_fa.npy", fa)
    np.save(out / "radius3_X.npy", X)
    np.save(out / "radius3_y.npy", y)

    scaffold = strict_scaffold_holdout(parent, cfg.scaffold_test_size, cfg.scaffold_seed)
    development_idx = scaffold["development_idx"]
    scaffold_test_idx = scaffold["test_idx"]
    quarantine_idx = scaffold["quarantine_idx"]
    parent.iloc[scaffold_test_idx].to_csv(out / "scaffold_disjoint_20_pairs.csv", index=False, encoding="utf-8-sig")
    parent.iloc[quarantine_idx].to_csv(out / "scaffold_overlap_quarantine.csv", index=False, encoding="utf-8-sig")
    parent.iloc[development_idx].to_csv(out / "development_pairs.csv", index=False, encoding="utf-8-sig")
    parent.iloc[np.sort(np.concatenate([scaffold_test_idx, quarantine_idx]))].to_csv(
        out / "excluded_from_model_development.csv", index=False, encoding="utf-8-sig"
    )

    pair_hash = paired_row_hashes(fd, fa)
    pair_groups = factorize_hashes(pair_hash)
    local_split = structure_ks_group_split(
        fd[development_idx], fa[development_idx], pair_groups[development_idx],
        cfg.test_size, cfg.valid_fraction_of_trainval, cfg.fixed_split_seed,
    )
    split = {key: development_idx[np.asarray(value, dtype=int)] for key, value in local_split.items()}
    overlap_report = validate_group_disjoint(split, pair_groups)
    np.savez_compressed(
        out / "split_indices.npz",
        train_idx=split["train_idx"], valid_idx=split["valid_idx"], test_idx=split["test_idx"],
        scaffold_test_idx=scaffold_test_idx, quarantine_idx=quarantine_idx, development_idx=development_idx,
    )
    save_json(out / "split_audit.json", {
        "split_method": "target-blind structure-only Kennard-Stone-like group split",
        "identical_pair_fingerprint_groups_kept_together": True,
        **overlap_report,
        "counts": {k: int(len(v)) for k, v in split.items()},
        "scaffold_test_n": int(len(scaffold_test_idx)),
        "quarantine_n": int(len(quarantine_idx)),
        "development_n": int(len(development_idx)),
        "donor_scaffold_overlap_test_vs_development": int(len(set(parent.iloc[scaffold_test_idx]["donor_scaffold"]) & set(parent.iloc[development_idx]["donor_scaffold"]))),
        "acceptor_scaffold_overlap_test_vs_development": int(len(set(parent.iloc[scaffold_test_idx]["acceptor_scaffold"]) & set(parent.iloc[development_idx]["acceptor_scaffold"]))),
    })

    no_aug = fit_conventional_models(
        X, y, split, feature_names, cfg.mi_k, cfg.fixed_split_seed, cfg.fixed_split_seed, augment=False
    )
    with_aug = fit_conventional_models(
        X, y, split, feature_names, cfg.mi_k, cfg.fixed_split_seed, cfg.fixed_split_seed, augment=True
    )

    model_rows = []
    selected_rows = []
    for setting, fitted in [("no_augmentation", no_aug), ("with_augmentation", with_aug)]:
        for name in MODEL_ORDER:
            model_rows.append({
                "setting": setting, "model": name,
                **{f"validation_{k}": v for k, v in fitted["results"][name]["valid_metrics"].items()},
                **{f"test_{k}": v for k, v in fitted["results"][name]["test_metrics"].items()},
            })
        best = fitted["best_model_name"]
        selected_rows.append({
            "setting": setting, "selected_model": best, "selection_rule": fitted["selection_rule"],
            **{f"validation_{k}": v for k, v in fitted["results"][best]["valid_metrics"].items()},
            **{f"test_{k}": v for k, v in fitted["results"][best]["test_metrics"].items()},
        })
    pd.DataFrame(model_rows).to_csv(out / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(out / "selected_models.csv", index=False, encoding="utf-8-sig")

    best_name = no_aug["best_model_name"]
    fixed_pred = no_aug["results"][best_name]["test_pred"]
    fixed_metrics = save_prediction_evaluation(
        out, "target_blind_fixed_best_model", y[split["test_idx"]], fixed_pred, split["test_idx"], cfg.high_pce_threshold
    )
    rf_fixed_pred = no_aug["results"]["RandomForest"]["test_pred"]
    rf_fixed_metrics = save_prediction_evaluation(
        out, "target_blind_fixed_random_forest", y[split["test_idx"]], rf_fixed_pred, split["test_idx"], cfg.high_pce_threshold
    )

    scaffold_pred = predict_pipeline(no_aug, "RandomForest", X[scaffold_test_idx])
    scaffold_metrics = save_prediction_evaluation(
        out, "scaffold_disjoint_holdout_random_forest", y[scaffold_test_idx], scaffold_pred,
        scaffold_test_idx, cfg.high_pce_threshold, bootstrap_repeats=cfg.bootstrap_repeats,
    )
    scaffold_detail = parent.iloc[scaffold_test_idx].copy().reset_index(drop=True)
    scaffold_detail.insert(0, "sample_id", [f"SD{i+1}" for i in range(len(scaffold_detail))])
    scaffold_detail["predicted_PCE"] = scaffold_pred
    scaffold_detail["signed_error"] = scaffold_pred - y[scaffold_test_idx]
    scaffold_detail["absolute_error"] = np.abs(scaffold_detail["signed_error"])
    scaffold_detail.to_csv(out / "scaffold_disjoint_predictions_with_structures.csv", index=False, encoding="utf-8-sig")

    similarity = nearest_similarity_audit(
        fd[scaffold_test_idx], fa[scaffold_test_idx], fd[development_idx], fa[development_idx],
        scaffold_test_idx, development_idx,
    )
    similarity = similarity.merge(parent[["array_index", "pair_key", "donor_scaffold", "acceptor_scaffold"]], left_on="test_array_index", right_on="array_index", how="left").drop(columns=["array_index"])
    similarity.to_csv(out / "scaffold_disjoint_similarity_audit.csv", index=False, encoding="utf-8-sig")
    save_json(out / "scaffold_disjoint_similarity_summary.json", {
        col: empirical_summary(similarity[col].to_numpy())
        for col in ["nearest_donor_similarity", "nearest_acceptor_similarity", "nearest_pair_similarity"]
    })

    selected_idx = no_aug["selected_indices"]
    pd.DataFrame({
        "selected_position": np.arange(len(selected_idx), dtype=int),
        "original_feature_index_zero_based": selected_idx,
        "feature_name_zero_based": [feature_names[i] for i in selected_idx],
    }).to_csv(out / "selected_features_for_interpretation.csv", index=False, encoding="utf-8-sig")

    interpretation_bundle = {
        "model_name": "RandomForest",
        "model_selection_status": "pre-specified interpretation model; not selected using the test set",
        "model": no_aug["results"]["RandomForest"]["model"],
        "scaler": no_aug["scaler"],
        "feature_names": feature_names,
        "selected_indices": selected_idx,
        "radius": cfg.radius,
        "n_bits": cfg.n_bits,
        "train_idx": split["train_idx"],
        "valid_idx": split["valid_idx"],
        "test_idx": split["test_idx"],
        "development_idx": development_idx,
        "scaffold_test_idx": scaffold_test_idx,
        "parent_csv": str(out / "parent_pairs.csv"),
        "X_path": str(out / "radius3_X.npy"),
        "fd_path": str(out / "radius3_fd.npy"),
        "fa_path": str(out / "radius3_fa.npy"),
        "y_path": str(out / "radius3_y.npy"),
        "feature_selection_fitted_on": "training subset only",
        "interpretation_model_training_data": "no-augmentation training subset only",
    }
    joblib.dump(interpretation_bundle, out / "interpretation_pipeline.joblib", compress=3)

    if cfg.run_nested_baseline:
        run_nested_baseline(X, y, fd, fa, development_idx, pair_groups, feature_names, cfg, out)

    summary = {
        "config": asdict(cfg),
        "n_structure_available_records": int(len(records)),
        "n_unique_parent_pairs": int(len(parent)),
        "n_scaffold_disjoint_test": int(len(scaffold_test_idx)),
        "n_scaffold_quarantine": int(len(quarantine_idx)),
        "n_development": int(len(development_idx)),
        "fixed_split_counts": {k.replace("_idx", ""): int(len(v)) for k, v in split.items()},
        "fixed_best_model_selected_by_validation_rmse": best_name,
        "fixed_best_model_metrics": fixed_metrics,
        "fixed_random_forest_metrics": rf_fixed_metrics,
        "scaffold_disjoint_random_forest_metrics": scaffold_metrics,
        "interpretation_model": "no-augmentation RandomForest",
        "target_values_used_for_fixed_split": False,
    }
    save_json(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Outputs written to {out}")


if __name__ == "__main__":
    main()

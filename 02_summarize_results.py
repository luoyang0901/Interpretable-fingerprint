#!/usr/bin/env python3
"""Summarize fixed and nested TextCNN campaigns.

Fixed campaign outputs:
- best single model selected by validation RMSE;
- common-test-set ensemble across model seeds;
- calibration, residual, PCE-stratified and ranking metrics;
- source-specific joint-test metrics;
- exact repeated-prediction audit.

Nested campaign outputs:
- one ensemble per split, averaging only models trained on the same split;
- empirical percentile ranges across split ensembles;
- separate between-split and within-split/model-seed variance components.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common_utils import (
    empirical_summary,
    ranking_metrics,
    regression_metrics,
    residual_table,
    save_json,
    stratified_error_table,
)

METRICS = [
    "r", "spearman_rho", "r2", "rmse", "mae", "mean_signed_error",
    "calibration_intercept", "calibration_slope",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixed-config", required=True)
    p.add_argument("--nested-config")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--high-pce-threshold", type=float, default=16.0)
    p.add_argument("--allow-incomplete", action="store_true")
    return p.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def resolve(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def array_hash(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(arr).tobytes())
    return h.hexdigest()


def load_campaign(config_path: Path, allow_incomplete: bool) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    cfg = load_json(config_path)
    root = resolve(config_path.parent, cfg["output_root"])
    manifest_path = root / "run_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(manifest_path)
    good = manifest[manifest["status"].isin(["complete", "skipped_complete"])]
    if not allow_incomplete and len(good) != len(manifest):
        raise RuntimeError(f"Campaign has {len(manifest)-len(good)} incomplete/failed rows")
    campaign: dict[str, list[dict[str, Any]]] = {}
    for row in good.itertuples(index=False):
        run_dir = Path(str(row.output_dir))
        if not run_dir.is_absolute():
            run_dir = (root / run_dir).resolve()
        summary = load_json(run_dir / "run_summary.json")
        per_seed = pd.read_csv(run_dir / "per_seed_summary.csv")
        if len(per_seed) != 1:
            raise ValueError(f"Expected one model seed in {run_dir}")
        model_seed = int(row.model_seed)
        pred_path = run_dir / f"model_seed_{model_seed}" / "test_predictions.csv"
        pred = pd.read_csv(pred_path)
        split = np.load(run_dir / "split_indices.npz")
        rec = {
            "branch": str(row.branch),
            "split_seed": int(row.split_seed),
            "model_seed": model_seed,
            "run_dir": run_dir,
            "summary": summary,
            "best_valid_rmse": float(per_seed.iloc[0]["best_valid_rmse"]),
            "valid_metrics": {m: float(per_seed.iloc[0][f"valid_{m}"]) for m in METRICS},
            "test_metrics": {m: float(per_seed.iloc[0][f"test_{m}"]) for m in METRICS},
            "test_idx": np.asarray(split["test_idx"], dtype=int),
            "train_idx": np.asarray(split["train_idx"], dtype=int),
            "valid_idx": np.asarray(split["valid_idx"], dtype=int),
            "y_true": pred["y_true"].to_numpy(dtype=float),
            "y_pred": pred["y_pred"].to_numpy(dtype=float),
        }
        campaign.setdefault(str(row.branch), []).append(rec)
    for branch in campaign:
        campaign[branch] = sorted(campaign[branch], key=lambda r: (r["split_seed"], r["model_seed"]))
    return cfg, campaign


def metadata_for_branch(config_path: Path, cfg: dict[str, Any], branch: str) -> pd.DataFrame | None:
    value = cfg["branches"][branch].get("metadata_csv")
    if not value:
        return None
    meta = pd.read_csv(resolve(config_path.parent, value))
    if "array_index" not in meta.columns:
        meta = meta.reset_index().rename(columns={"index": "array_index"})
    return meta


def write_prediction_bundle(
    branch_out: Path,
    stem: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_idx: np.ndarray,
    high_threshold: float,
    metadata: pd.DataFrame | None,
) -> tuple[dict[str, float], pd.DataFrame]:
    pred = residual_table(y_true, y_pred, test_idx)
    if metadata is not None:
        pred = pred.merge(metadata, on="array_index", how="left", suffixes=("", "_metadata"))
    pred.to_csv(branch_out / f"{stem}_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = regression_metrics(y_true, y_pred)
    save_json(branch_out / f"{stem}_metrics.json", metrics)
    ranking_metrics(y_true, y_pred, high_threshold=high_threshold).to_csv(
        branch_out / f"{stem}_ranking_metrics.csv", index=False, encoding="utf-8-sig"
    )
    stratified_error_table(y_true, y_pred).to_csv(
        branch_out / f"{stem}_pce_stratified_errors.csv", index=False, encoding="utf-8-sig"
    )
    return metrics, pred


def prediction_collision_audit(pred: pd.DataFrame) -> pd.DataFrame:
    work = pred.copy()
    work["prediction_key_12dp"] = work["y_pred"].round(12).map(lambda x: f"{x:.12f}")
    rows = []
    for key, sub in work.groupby("prediction_key_12dp"):
        if len(sub) < 2:
            continue
        row = {
            "prediction_key_12dp": key,
            "group_size": int(len(sub)),
            "array_indices": ";".join(map(str, sub["array_index"].astype(int).tolist())),
            "y_true_min": float(sub["y_true"].min()),
            "y_true_max": float(sub["y_true"].max()),
            "y_true_range": float(sub["y_true"].max() - sub["y_true"].min()),
        }
        for col in ["source", "pair_fp_hash", "legacy_sequence_hash", "role_aware_sequence_hash"]:
            if col in sub.columns:
                row[f"unique_{col}"] = int(sub[col].astype(str).nunique())
                row[f"{col}_values"] = ";".join(sub[col].astype(str).tolist())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_fixed(
    config_path: Path,
    cfg: dict[str, Any],
    campaign: dict[str, list[dict[str, Any]]],
    out: Path,
    high_threshold: float,
) -> dict[str, Any]:
    if cfg.get("campaign_mode") != "fixed_model_seeds":
        raise ValueError("fixed-config must use campaign_mode=fixed_model_seeds")
    cross_rows = []
    fixed_seed_stats = []
    branch_summary: dict[str, Any] = {}
    for branch, records in campaign.items():
        branch_out = out / branch
        branch_out.mkdir(parents=True, exist_ok=True)
        first_idx = records[0]["test_idx"]
        first_true = records[0]["y_true"]
        for rec in records[1:]:
            if not np.array_equal(first_idx, rec["test_idx"]):
                raise ValueError(f"{branch}: fixed campaign test indices differ")
            if not np.allclose(first_true, rec["y_true"], atol=1e-7, rtol=0):
                raise ValueError(f"{branch}: fixed campaign y_true differs")
        best = min(records, key=lambda r: (r["best_valid_rmse"], r["model_seed"]))
        pred_matrix = np.column_stack([r["y_pred"] for r in records])
        ensemble = pred_matrix.mean(axis=1)
        metadata = metadata_for_branch(config_path, cfg, branch)
        best_metrics, best_pred = write_prediction_bundle(
            branch_out, "best_single", first_true, best["y_pred"], first_idx, high_threshold, metadata
        )
        ensemble_metrics, ensemble_pred = write_prediction_bundle(
            branch_out, "ensemble", first_true, ensemble, first_idx, high_threshold, metadata
        )
        ensemble_pred["prediction_sd_across_model_seeds"] = pred_matrix.std(axis=1, ddof=1) if len(records) > 1 else 0.0
        ensemble_pred.to_csv(branch_out / "ensemble_predictions.csv", index=False, encoding="utf-8-sig")
        np.save(branch_out / "all_model_seed_prediction_matrix.npy", pred_matrix.astype(np.float32))
        collision = prediction_collision_audit(ensemble_pred)
        collision.to_csv(branch_out / "ensemble_repeated_prediction_audit.csv", index=False, encoding="utf-8-sig")

        source_rows = []
        if metadata is not None and "source" in ensemble_pred.columns:
            for setting, frame in [("best_single", best_pred), ("ensemble", ensemble_pred)]:
                for source, sub in frame.groupby("source"):
                    if len(sub) < 2:
                        continue
                    source_rows.append({
                        "setting": setting,
                        "source": source,
                        "n": int(len(sub)),
                        **regression_metrics(sub["y_true"].to_numpy(), sub["y_pred"].to_numpy()),
                    })
        pd.DataFrame(source_rows).to_csv(branch_out / "source_specific_test_metrics.csv", index=False, encoding="utf-8-sig")

        per_seed_rows = []
        for rec in records:
            per_seed_rows.append({
                "branch": branch,
                "split_seed": rec["split_seed"],
                "model_seed": rec["model_seed"],
                "best_valid_rmse": rec["best_valid_rmse"],
                **{f"test_{m}": rec["test_metrics"][m] for m in METRICS},
                **{f"valid_{m}": rec["valid_metrics"][m] for m in METRICS},
            })
        pd.DataFrame(per_seed_rows).to_csv(branch_out / "fixed_all_model_seed_metrics.csv", index=False, encoding="utf-8-sig")
        for metric in METRICS:
            fixed_seed_stats.append({
                "branch": branch,
                "variability_source": "training stochasticity on one fixed split",
                "metric": metric,
                **empirical_summary([r["test_metrics"][metric] for r in records]),
            })
        row = {
            "branch": branch,
            "n_model_seeds": len(records),
            "fixed_split_seed": records[0]["split_seed"],
            "best_single_model_seed": best["model_seed"],
            **{f"best_{k}": v for k, v in best_metrics.items()},
            **{f"ensemble_{k}": v for k, v in ensemble_metrics.items()},
        }
        cross_rows.append(row)
        branch_summary[branch] = {
            "best_single_model_seed": best["model_seed"],
            "selection_rule": "lowest validation RMSE",
            "best_single_metrics": best_metrics,
            "ensemble_metrics": ensemble_metrics,
            "test_index_hash": array_hash(first_idx),
            "n_repeated_prediction_groups_12dp": int(len(collision)),
        }
    pd.DataFrame(cross_rows).to_csv(out / "fixed_cross_branch_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fixed_seed_stats).to_csv(out / "fixed_model_seed_variability.csv", index=False, encoding="utf-8-sig")
    return branch_summary


def summarize_nested(
    config_path: Path,
    cfg: dict[str, Any],
    campaign: dict[str, list[dict[str, Any]]],
    out: Path,
) -> dict[str, Any]:
    if cfg.get("campaign_mode") != "nested":
        raise ValueError("nested-config must use campaign_mode=nested")
    nested_root = out / "nested"
    nested_root.mkdir(parents=True, exist_ok=True)
    global_summary: dict[str, Any] = {}
    for branch, records in campaign.items():
        branch_out = nested_root / branch
        branch_out.mkdir(parents=True, exist_ok=True)
        individual_rows = []
        split_ensemble_rows = []
        for rec in records:
            individual_rows.append({
                "split_seed": rec["split_seed"],
                "model_seed": rec["model_seed"],
                **rec["test_metrics"],
            })
        individual = pd.DataFrame(individual_rows)
        individual.to_csv(branch_out / "nested_individual_run_metrics.csv", index=False, encoding="utf-8-sig")
        for split_seed, split_records in pd.Series(records).groupby(lambda i: records[i]["split_seed"]):
            split_records = list(split_records)
            first_idx = split_records[0]["test_idx"]
            first_true = split_records[0]["y_true"]
            for rec in split_records[1:]:
                if not np.array_equal(first_idx, rec["test_idx"]):
                    raise ValueError(f"{branch} split {split_seed}: test indices differ across model seeds")
            pred = np.column_stack([r["y_pred"] for r in split_records]).mean(axis=1)
            split_ensemble_rows.append({
                "split_seed": int(split_seed),
                "n_model_seeds": int(len(split_records)),
                "n_test": int(len(first_true)),
                **regression_metrics(first_true, pred),
            })
        split_ensemble = pd.DataFrame(split_ensemble_rows).sort_values("split_seed")
        split_ensemble.to_csv(branch_out / "nested_split_ensemble_metrics.csv", index=False, encoding="utf-8-sig")
        stats_rows = []
        variance_rows = []
        for metric in METRICS:
            stats_rows.append({"metric": metric, **empirical_summary(split_ensemble[metric].to_numpy())})
            split_means = individual.groupby("split_seed")[metric].mean()
            within_vars = individual.groupby("split_seed")[metric].var(ddof=1).fillna(0.0)
            between = float(np.var(split_means.to_numpy(), ddof=1)) if len(split_means) > 1 else 0.0
            within = float(np.mean(within_vars.to_numpy())) if len(within_vars) else 0.0
            total = float(np.var(individual[metric].to_numpy(), ddof=1)) if len(individual) > 1 else 0.0
            variance_rows.append({
                "metric": metric,
                "between_split_variance_of_split_means": between,
                "mean_within_split_model_seed_variance": within,
                "total_individual_run_variance": total,
                "between_fraction_of_between_plus_within": between / (between + within) if between + within > 0 else np.nan,
                "interpretation": "between=partition sensitivity; within=training stochasticity conditional on a fixed partition",
            })
        stats = pd.DataFrame(stats_rows)
        variance = pd.DataFrame(variance_rows)
        stats.to_csv(branch_out / "nested_empirical_split_statistics.csv", index=False, encoding="utf-8-sig")
        variance.to_csv(branch_out / "nested_variance_decomposition.csv", index=False, encoding="utf-8-sig")
        global_summary[branch] = {
            "n_splits": int(split_ensemble["split_seed"].nunique()),
            "model_seeds_per_split_min": int(split_ensemble["n_model_seeds"].min()),
            "model_seeds_per_split_max": int(split_ensemble["n_model_seeds"].max()),
            "statistics_interpretation": "empirical variability across overlapping resamples of one finite dataset; not independent experimental datasets",
        }
    return global_summary


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    fixed_path = Path(args.fixed_config).resolve()
    fixed_cfg, fixed_campaign = load_campaign(fixed_path, args.allow_incomplete)
    fixed_summary = summarize_fixed(fixed_path, fixed_cfg, fixed_campaign, out, args.high_pce_threshold)
    nested_summary = None
    if args.nested_config:
        nested_path = Path(args.nested_config).resolve()
        nested_cfg, nested_campaign = load_campaign(nested_path, args.allow_incomplete)
        if set(nested_campaign) != set(fixed_campaign):
            raise ValueError("Fixed and nested campaigns must contain the same branch names")
        nested_summary = summarize_nested(nested_path, nested_cfg, nested_campaign, out)
    save_json(out / "publication_summary.json", {
        "fixed_config": str(fixed_path),
        "nested_config": str(Path(args.nested_config).resolve()) if args.nested_config else None,
        "fixed": fixed_summary,
        "nested": nested_summary,
        "primary_performance_recommendation": "Use nested split-ensemble mean, SD and empirical percentile range as the realistic internal-performance estimate; treat the fixed split as a common-test comparison benchmark.",
    })
    print(f"Summary written to {out}")


if __name__ == "__main__":
    main()

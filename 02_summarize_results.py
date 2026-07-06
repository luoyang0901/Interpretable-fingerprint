#!/usr/bin/env python3
"""Summarize fixed-split and optional repeated-split 101-seed TextCNN runs.

Outputs include:
- per-seed metric tables;
- mean, SD and 95% CI tables;
- best-single and 101-seed ensemble predictions for the fixed-split campaign;
- cross-branch performance table;
- high-PCE case tables;
- LaTeX-ready tables.

The fixed campaign must use seed_mode=model_only so that all seeds share the
same held-out samples. The optional repeated campaign may use
seed_mode=split_and_model; it is used only for uncertainty statistics and is
never averaged into an ensemble.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, t
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

METRICS = ["r", "r2", "rmse", "mae"]
DISPLAY = {"r": "Pearson r", "r2": "R2", "rmse": "RMSE", "mae": "MAE"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixed-config", required=True, help="Config used for fixed-split/model-seed campaign.")
    p.add_argument("--repeated-config", help="Optional config used for split+model-seed campaign.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--allow-incomplete", action="store_true", help="Summarize fewer than the expected 101 seeds.")
    p.add_argument("--high-pce-threshold", type=float, default=16.0)
    p.add_argument("--minimum-case-count", type=int, default=10)
    return p.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p).resolve()


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "r": float(pearsonr(y_true, y_pred)[0]),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def hash_array(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(arr).tobytes())
    return h.hexdigest()


def find_prediction_file(seed_dir: Path, seed: int) -> Path:
    direct = seed_dir / f"model_seed_{seed}" / "test_predictions.csv"
    if direct.exists():
        return direct
    matches = list(seed_dir.glob("model_seed_*/test_predictions.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one test_predictions.csv in {seed_dir}; found {len(matches)}")
    return matches[0]


def load_seed_record(branch_dir: Path, seed: int) -> dict[str, Any]:
    seed_dir = branch_dir / f"seed_{seed:03d}"
    summary_path = seed_dir / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = load_json(summary_path)
    per_seed_path = seed_dir / "per_seed_summary.csv"
    if not per_seed_path.exists():
        raise FileNotFoundError(per_seed_path)
    per_seed = pd.read_csv(per_seed_path)
    if len(per_seed) != 1:
        raise ValueError(f"Expected one row in {per_seed_path}, got {len(per_seed)}")
    row = per_seed.iloc[0]
    pred_path = find_prediction_file(seed_dir, seed)
    pred = pd.read_csv(pred_path)
    required = {"y_true", "y_pred"}
    missing = required.difference(pred.columns)
    if missing:
        raise ValueError(f"{pred_path} lacks {sorted(missing)}")
    split_path = seed_dir / "split_indices.npz"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    split = np.load(split_path)
    return {
        "seed": seed,
        "seed_dir": seed_dir,
        "summary": summary,
        "best_valid_rmse": float(row["best_valid_rmse"]),
        "test_metrics": {metric: float(row[f"test_{metric}"]) for metric in METRICS},
        "valid_metrics": {metric: float(row[f"valid_{metric}"]) for metric in METRICS},
        "y_true": pred["y_true"].to_numpy(dtype=float),
        "y_pred": pred["y_pred"].to_numpy(dtype=float),
        "test_idx": np.asarray(split["test_idx"], dtype=int),
        "train_idx": np.asarray(split["train_idx"], dtype=int),
        "valid_idx": np.asarray(split["valid_idx"], dtype=int),
    }


def expected_seeds(config: dict[str, Any]) -> list[int]:
    return list(range(int(config.get("seed_start", 0)), int(config.get("seed_end", 100)) + 1))


def collect_campaign(config_path: Path, allow_incomplete: bool) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    cfg = load_json(config_path)
    root = resolve_path(config_path.parent, cfg["output_root"])
    seeds = expected_seeds(cfg)
    campaign: dict[str, list[dict[str, Any]]] = {}
    for branch in cfg["branches"]:
        records = []
        missing = []
        for seed in seeds:
            try:
                records.append(load_seed_record(root / branch, seed))
            except FileNotFoundError:
                missing.append(seed)
        if missing and not allow_incomplete:
            raise RuntimeError(f"{branch}: missing {len(missing)} seeds: {missing[:20]}")
        if len(records) < 2:
            raise RuntimeError(f"{branch}: only {len(records)} complete seed runs found.")
        campaign[branch] = sorted(records, key=lambda x: x["seed"])
    return cfg, campaign


def mean_sd_ci(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan, "ci_low": math.nan, "ci_high": math.nan, "min": math.nan, "max": math.nan}
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    critical = float(t.ppf(0.975, df=n - 1)) if n > 1 else math.nan
    half = critical * sd / math.sqrt(n) if n > 1 else math.nan
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "ci_low": mean - half if n > 1 else math.nan,
        "ci_high": mean + half if n > 1 else math.nan,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def validate_fixed_branch(records: list[dict[str, Any]], branch: str) -> None:
    first_true = records[0]["y_true"]
    first_test_idx = records[0]["test_idx"]
    for rec in records[1:]:
        if not np.array_equal(rec["test_idx"], first_test_idx):
            raise ValueError(f"{branch}: test indices differ across fixed-split seeds.")
        if not np.allclose(rec["y_true"], first_true, rtol=0, atol=1e-7):
            raise ValueError(f"{branch}: y_true differs across fixed-split seeds.")


def latex_escape(text: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def dataframe_to_latex(df: pd.DataFrame, path: Path, float_format: str = "%.4f") -> None:
    latex = df.to_latex(index=False, escape=True, float_format=lambda x: float_format % x)
    path.write_text(latex, encoding="utf-8")


def load_metadata(config_path: Path, cfg: dict[str, Any], branch: str) -> pd.DataFrame | None:
    spec = cfg["branches"][branch]
    value = spec.get("metadata_csv")
    if not value:
        return None
    path = resolve_path(config_path.parent, value)
    if not path.exists():
        raise FileNotFoundError(f"Metadata CSV for {branch} not found: {path}")
    meta = pd.read_csv(path)
    if "array_index" not in meta.columns:
        meta = meta.reset_index().rename(columns={"index": "array_index"})
    return meta


def select_high_pce_cases(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    test_idx: np.ndarray,
    threshold: float,
    minimum_count: int,
    metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    base = pd.DataFrame(
        {
            "test_position": np.arange(len(y_true), dtype=int),
            "array_index": test_idx.astype(int),
            "experimental_PCE": y_true,
            "predicted_PCE": y_pred,
        }
    )
    base["absolute_error"] = np.abs(base["experimental_PCE"] - base["predicted_PCE"])
    base["relative_error_percent"] = np.where(
        np.abs(base["experimental_PCE"]) > 1e-12,
        base["absolute_error"] / np.abs(base["experimental_PCE"]) * 100.0,
        np.nan,
    )
    above = base.loc[base["experimental_PCE"] > threshold].sort_values("experimental_PCE", ascending=False)
    if len(above) < minimum_count:
        fill = base.loc[~base.index.isin(above.index)].sort_values("experimental_PCE", ascending=False).head(minimum_count - len(above))
        selected = pd.concat([above, fill], axis=0)
    else:
        selected = above
    selected = selected.sort_values("experimental_PCE", ascending=False).reset_index(drop=True)
    selected.insert(0, "case_rank", np.arange(1, len(selected) + 1, dtype=int))
    selected["selection_rule"] = np.where(
        selected["experimental_PCE"] > threshold,
        f"Experimental PCE > {threshold:g}%",
        f"Highest remaining PCE to reach at least {minimum_count} cases",
    )
    if metadata is not None:
        selected = selected.merge(metadata, on="array_index", how="left", suffixes=("", "_metadata"))
    return selected


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    fixed_path = Path(args.fixed_config).resolve()
    fixed_cfg, fixed = collect_campaign(fixed_path, args.allow_incomplete)
    if fixed_cfg.get("seed_mode", "model_only") != "model_only":
        raise ValueError("The fixed campaign must use seed_mode='model_only'.")

    repeated_cfg: dict[str, Any] | None = None
    repeated: dict[str, list[dict[str, Any]]] | None = None
    repeated_path: Path | None = None
    if args.repeated_config:
        repeated_path = Path(args.repeated_config).resolve()
        repeated_cfg, repeated = collect_campaign(repeated_path, args.allow_incomplete)
        if set(repeated) != set(fixed):
            raise ValueError("Fixed and repeated campaigns contain different branch names.")

    per_seed_rows: list[dict[str, Any]] = []
    stats_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    branch_summary: dict[str, Any] = {}

    for branch, records in fixed.items():
        validate_fixed_branch(records, branch)
        branch_out = out / branch
        branch_out.mkdir(parents=True, exist_ok=True)

        for rec in records:
            row: dict[str, Any] = {
                "campaign": "fixed_split_model_seed",
                "branch": branch,
                "seed": rec["seed"],
                "best_valid_rmse": rec["best_valid_rmse"],
            }
            for metric in METRICS:
                row[f"test_{metric}"] = rec["test_metrics"][metric]
                row[f"valid_{metric}"] = rec["valid_metrics"][metric]
            per_seed_rows.append(row)

        # Select best single by validation RMSE, never by test performance.
        best = min(records, key=lambda x: (x["best_valid_rmse"], x["seed"]))
        y_true = best["y_true"]
        pred_matrix = np.column_stack([rec["y_pred"] for rec in records])
        ensemble_pred = pred_matrix.mean(axis=1)
        best_metrics = regression_metrics(y_true, best["y_pred"])
        ensemble_metrics = regression_metrics(y_true, ensemble_pred)

        best_pred_df = pd.DataFrame(
            {
                "test_position": np.arange(len(y_true), dtype=int),
                "array_index": best["test_idx"],
                "y_true": y_true,
                "y_pred": best["y_pred"],
                "absolute_error": np.abs(y_true - best["y_pred"]),
            }
        )
        ensemble_pred_df = pd.DataFrame(
            {
                "test_position": np.arange(len(y_true), dtype=int),
                "array_index": best["test_idx"],
                "y_true": y_true,
                "y_pred": ensemble_pred,
                "prediction_sd_across_seeds": pred_matrix.std(axis=1, ddof=1),
                "absolute_error": np.abs(y_true - ensemble_pred),
            }
        )
        best_pred_df.to_csv(branch_out / "best_single_predictions.csv", index=False, encoding="utf-8-sig")
        ensemble_pred_df.to_csv(branch_out / "ensemble_101_predictions.csv", index=False, encoding="utf-8-sig")
        np.save(branch_out / "all_seed_prediction_matrix.npy", pred_matrix.astype(np.float32))

        metadata = load_metadata(fixed_path, fixed_cfg, branch)
        high_cases = select_high_pce_cases(
            y_true,
            ensemble_pred,
            best["test_idx"],
            args.high_pce_threshold,
            args.minimum_case_count,
            metadata,
        )
        high_cases.to_csv(branch_out / "high_PCE_cases.csv", index=False, encoding="utf-8-sig")
        latex_cols = [
            c for c in [
                "case_rank", "array_index", "source", "experimental_PCE", "predicted_PCE",
                "absolute_error", "relative_error_percent", "selection_rule",
            ] if c in high_cases.columns
        ]
        dataframe_to_latex(high_cases[latex_cols], branch_out / "high_PCE_cases.tex")

        selection = {
            "branch": branch,
            "n_fixed_seeds": len(records),
            "best_single_seed": int(best["seed"]),
            "selection_criterion": "lowest validation RMSE",
            "best_valid_rmse": best["best_valid_rmse"],
            "best_single_test_metrics": best_metrics,
            "ensemble_test_metrics": ensemble_metrics,
            "test_index_hash": hash_array(best["test_idx"]),
            "y_true_hash": hash_array(y_true),
        }
        (branch_out / "selected_models_and_metrics.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
        branch_summary[branch] = selection

        cross_rows.append(
            {
                "Branch": branch,
                "Fixed-split seeds": len(records),
                "Best single seed": int(best["seed"]),
                "Best single r": best_metrics["r"],
                "Best single R2": best_metrics["r2"],
                "Best single RMSE": best_metrics["rmse"],
                "Best single MAE": best_metrics["mae"],
                "Ensemble r": ensemble_metrics["r"],
                "Ensemble R2": ensemble_metrics["r2"],
                "Ensemble RMSE": ensemble_metrics["rmse"],
                "Ensemble MAE": ensemble_metrics["mae"],
            }
        )

        # Statistics source: repeated-split if provided, otherwise fixed-split model-seed variability.
        stats_records = repeated[branch] if repeated is not None else records
        stats_label = "repeated_split_and_model_seed" if repeated is not None else "fixed_split_model_seed"
        for rec in stats_records:
            row = {
                "campaign": stats_label,
                "branch": branch,
                "seed": rec["seed"],
                "best_valid_rmse": rec["best_valid_rmse"],
            }
            for metric in METRICS:
                row[f"test_{metric}"] = rec["test_metrics"][metric]
                row[f"valid_{metric}"] = rec["valid_metrics"][metric]
            if repeated is not None:
                per_seed_rows.append(row)

        for metric in METRICS:
            stats = mean_sd_ci(rec["test_metrics"][metric] for rec in stats_records)
            stats_rows.append(
                {
                    "Campaign": stats_label,
                    "Branch": branch,
                    "Metric": DISPLAY[metric],
                    "N": stats["n"],
                    "Mean": stats["mean"],
                    "SD": stats["sd"],
                    "95% CI lower": stats["ci_low"],
                    "95% CI upper": stats["ci_high"],
                    "Minimum": stats["min"],
                    "Maximum": stats["max"],
                }
            )

    per_seed_df = pd.DataFrame(per_seed_rows)
    per_seed_df.to_csv(out / "TextCNN_all_per_seed_metrics.csv", index=False, encoding="utf-8-sig")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(out / "Table_TextCNN_101seed_statistics_long.csv", index=False, encoding="utf-8-sig")
    dataframe_to_latex(stats_df, out / "Table_TextCNN_101seed_statistics_long.tex")

    # Publication-oriented wide table with formatted mean +/- SD and CI.
    wide_rows = []
    for branch in fixed:
        sub = stats_df.loc[stats_df["Branch"] == branch]
        row: dict[str, Any] = {"Branch": branch, "N": int(sub["N"].iloc[0]), "Campaign": sub["Campaign"].iloc[0]}
        for metric in ["Pearson r", "R2", "RMSE", "MAE"]:
            m = sub.loc[sub["Metric"] == metric].iloc[0]
            key = metric.replace(" ", "_")
            row[f"{key}_mean"] = m["Mean"]
            row[f"{key}_SD"] = m["SD"]
            row[f"{key}_CI_low"] = m["95% CI lower"]
            row[f"{key}_CI_high"] = m["95% CI upper"]
            row[f"{key}_formatted"] = f"{m['Mean']:.4f} +/- {m['SD']:.4f}; 95% CI [{m['95% CI lower']:.4f}, {m['95% CI upper']:.4f}]"
        wide_rows.append(row)
    wide_df = pd.DataFrame(wide_rows)
    wide_df.to_csv(out / "Table_TextCNN_101seed_statistics_wide.csv", index=False, encoding="utf-8-sig")
    dataframe_to_latex(wide_df, out / "Table_TextCNN_101seed_statistics_wide.tex")

    cross_df = pd.DataFrame(cross_rows)
    cross_df.to_csv(out / "Table_TextCNN_cross_branch.csv", index=False, encoding="utf-8-sig")
    dataframe_to_latex(cross_df, out / "Table_TextCNN_cross_branch.tex")

    publication_summary = {
        "fixed_config": str(fixed_path),
        "repeated_config": str(repeated_path) if repeated_path else None,
        "uncertainty_statistics_source": "repeated split + model seeds" if repeated is not None else "model seeds on a fixed split",
        "branches": branch_summary,
        "high_pce_threshold": args.high_pce_threshold,
        "minimum_case_count": args.minimum_case_count,
    }
    (out / "publication_summary.json").write_text(json.dumps(publication_summary, indent=2), encoding="utf-8")

    print(f"Summary written to: {out}")
    print(f"Cross-branch table: {out / 'Table_TextCNN_cross_branch.csv'}")
    print(f"101-seed table: {out / 'Table_TextCNN_101seed_statistics_wide.csv'}")


if __name__ == "__main__":
    main()

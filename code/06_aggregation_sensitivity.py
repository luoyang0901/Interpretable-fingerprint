#!/usr/bin/env python3
"""Clean maximum/mean/median duplicate-aggregation sensitivity analysis.

All aggregation rules use the same target-blind representation-group split.
For each rule, candidate models are selected exclusively by validation RMSE and
then evaluated once on the untouched test set. This avoids test-set model
selection and separates the aggregation rule from the selection protocol.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Baseline import (
    MODEL_ORDER,
    aggregate_pair_records,
    build_pair_fingerprints,
    fit_conventional_models,
    load_structure_records,
)
from common_utils import factorize_hashes, paired_row_hashes, save_json, structure_ks_group_split, validate_group_disjoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-da-csv", required=True, help="Pre-deduplication structure-available records.")
    p.add_argument("--excluded-pairs-csv", help="Optional baseline scaffold-test + quarantine exclusions.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--radius", type=int, default=3)
    p.add_argument("--n-bits", type=int, default=1024)
    p.add_argument("--mi-k", type=int, default=600)
    p.add_argument("--split-seed", type=int, default=12)
    p.add_argument("--model-seed", type=int, default=12)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--valid-fraction-of-trainval", type=float, default=0.125)
    return p.parse_args()


def load_excluded_keys(path: str | None) -> set[str]:
    if not path:
        return set()
    df = pd.read_csv(path)
    if "pair_key" in df.columns:
        return set(df["pair_key"].astype(str))
    if {"donor_smiles", "acceptor_smiles"}.issubset(df.columns):
        return set(df["donor_smiles"].astype(str) + "||" + df["acceptor_smiles"].astype(str))
    raise ValueError("excluded-pairs-csv must contain pair_key or donor_smiles/acceptor_smiles")


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = load_structure_records(args.raw_da_csv)
    excluded = load_excluded_keys(args.excluded_pairs_csv)
    if excluded:
        records = records.loc[~records["pair_key"].isin(excluded)].copy()
    aggregated = {rule: aggregate_pair_records(records, rule) for rule in ["maximum", "mean", "median"]}
    key_sets = {rule: set(df["pair_key"]) for rule, df in aggregated.items()}
    if len({frozenset(x) for x in key_sets.values()}) != 1:
        raise ValueError("Aggregation rules produced different pair identities")
    base = aggregated["maximum"].sort_values("pair_key").reset_index(drop=True)
    fd, fa, X, _, feature_names = build_pair_fingerprints(base, args.radius, args.n_bits)
    groups = factorize_hashes(paired_row_hashes(fd, fa))
    split = structure_ks_group_split(
        fd, fa, groups, args.test_size, args.valid_fraction_of_trainval, args.split_seed
    )
    validate_group_disjoint(split, groups)
    np.savez_compressed(out / "aggregation_common_split_indices.npz", **split)
    base[["pair_key", "donor_smiles", "acceptor_smiles"]].to_csv(
        out / "aggregation_pair_identity_order.csv", index=False, encoding="utf-8-sig"
    )

    all_rows = []
    selected_rows = []
    for rule, df in aggregated.items():
        aligned = base[["pair_key"]].merge(df[["pair_key", "PCE"]], on="pair_key", how="left", validate="one_to_one")
        y = aligned["PCE"].to_numpy(dtype=float)
        fitted = fit_conventional_models(
            X, y, split, feature_names, args.mi_k,
            feature_selection_seed=args.split_seed,
            model_seed=args.model_seed,
            augment=False,
        )
        for model in MODEL_ORDER:
            all_rows.append({
                "aggregation_rule": rule,
                "model": model,
                "selection_candidate": True,
                **{f"validation_{k}": v for k, v in fitted["results"][model]["valid_metrics"].items()},
                **{f"test_{k}": v for k, v in fitted["results"][model]["test_metrics"].items()},
            })
        best = fitted["best_model_name"]
        selected_rows.append({
            "aggregation_rule": rule,
            "selected_model": best,
            "selection_rule": "lowest validation RMSE",
            **{f"validation_{k}": v for k, v in fitted["results"][best]["valid_metrics"].items()},
            **{f"test_{k}": v for k, v in fitted["results"][best]["test_metrics"].items()},
        })
        pred = pd.DataFrame({
            "array_index": split["test_idx"],
            "pair_key": base.iloc[split["test_idx"]]["pair_key"].to_numpy(),
            "y_true": y[split["test_idx"]],
            "y_pred": fitted["results"][best]["test_pred"],
            "aggregation_rule": rule,
            "selected_model": best,
        })
        pred["residual_pred_minus_true"] = pred["y_pred"] - pred["y_true"]
        pred.to_csv(out / f"aggregation_{rule}_test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_rows).to_csv(out / "aggregation_all_candidate_model_metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selected_rows).to_csv(out / "aggregation_validation_selected_test_metrics.csv", index=False, encoding="utf-8-sig")
    save_json(out / "aggregation_protocol.json", {
        "raw_records_n": int(len(records)),
        "unique_pairs_n": int(len(base)),
        "aggregation_rules": ["maximum", "mean", "median"],
        "same_pair_identities_and_same_target_blind_split": True,
        "model_selection": "validation RMSE only",
        "test_set_used_for_model_selection": False,
        "mutual_information_fitted_on": "training subset within each aggregation rule",
        "split_seed": args.split_seed,
        "model_seed": args.model_seed,
    })
    print(json.dumps(pd.DataFrame(selected_rows).to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()

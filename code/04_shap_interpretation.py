#!/usr/bin/env python3
"""Regenerate the complete SHAP and structural interpretation from one model.

Inputs are the interpretation_pipeline.joblib and source arrays written by
Baseline.py. The script applies one predefined selection rule (top K by mean
absolute SHAP) to every downstream output: SHAP plots, correlation matrix,
feature table, and Morgan-bit structural back-mapping.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from rdkit import Chem
from rdkit.Chem import AllChem

from common_utils import environment_report, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline", required=True, help="baseline/interpretation_pipeline.joblib")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--background-size", type=int, default=200)
    p.add_argument("--background-seed", type=int, default=2026)
    p.add_argument("--explain-scope", choices=["test", "development", "all"], default="test")
    return p.parse_args()


def resolve_from_bundle(pipeline_path: Path, value: str) -> Path:
    p = Path(value)
    if p.exists():
        return p.resolve()
    candidate = pipeline_path.parent / p.name
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(value)


def feature_side_and_bit(name: str) -> tuple[str, int]:
    prefix, bit = name.split("_", 1)
    if prefix == "fd":
        return "donor", int(bit)
    if prefix == "fa":
        return "acceptor", int(bit)
    raise ValueError(name)


def environment_smiles(mol: Chem.Mol, atom_idx: int, radius: int) -> str:
    if radius <= 0:
        return Chem.MolFragmentToSmiles(
            mol, atomsToUse=[int(atom_idx)], rootedAtAtom=int(atom_idx),
            canonical=True, isomericSmiles=False,
        )
    bonds = list(Chem.FindAtomEnvironmentOfRadiusN(mol, int(radius), int(atom_idx)))
    atoms = {int(atom_idx)}
    for bond_idx in bonds:
        bond = mol.GetBondWithIdx(int(bond_idx))
        atoms.add(bond.GetBeginAtomIdx())
        atoms.add(bond.GetEndAtomIdx())
    return Chem.MolFragmentToSmiles(
        mol,
        atomsToUse=sorted(atoms),
        bondsToUse=bonds,
        rootedAtAtom=int(atom_idx),
        canonical=True,
        isomericSmiles=False,
    )


def map_bit_across_molecules(smiles_values: list[str], bit: int, radius: int, n_bits: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    fragment_molecule_support: dict[str, set[str]] = defaultdict(set)
    fragment_occurrences: Counter[str] = Counter()
    activated_molecules = 0
    total_bit_occurrences = 0
    for smiles in sorted(set(smiles_values)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        bit_info: dict[int, list[tuple[int, int]]] = {}
        AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits, bitInfo=bit_info)
        occurrences = bit_info.get(int(bit), [])
        if not occurrences:
            continue
        activated_molecules += 1
        for atom_idx, env_radius in occurrences:
            fragment = environment_smiles(mol, int(atom_idx), int(env_radius))
            fragment_occurrences[fragment] += 1
            fragment_molecule_support[fragment].add(smiles)
            total_bit_occurrences += 1
    rows = []
    for fragment, count in fragment_occurrences.most_common():
        rows.append({
            "environment_smiles": fragment,
            "environment_occurrence_count": int(count),
            "unique_molecule_support": int(len(fragment_molecule_support[fragment])),
        })
    summary = {
        "n_unique_input_molecules": int(len(set(smiles_values))),
        "n_unique_molecules_with_bit": int(activated_molecules),
        "n_total_bit_environment_occurrences": int(total_bit_occurrences),
        "n_distinct_mapped_environments": int(len(rows)),
    }
    return pd.DataFrame(rows), summary


def save_shap_plots(values: np.ndarray, features: np.ndarray, names: list[str], out: Path) -> None:
    shap.summary_plot(values, features=features, feature_names=names, max_display=min(20, len(names)), show=False)
    plt.tight_layout()
    plt.savefig(out / "shap_summary_beeswarm.png", dpi=600, bbox_inches="tight")
    plt.savefig(out / "shap_summary_beeswarm.pdf", bbox_inches="tight")
    plt.close()
    mean_abs = np.mean(np.abs(values), axis=0)
    order = np.argsort(mean_abs)[::-1][: min(20, len(names))]
    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.barh(np.arange(len(order)), mean_abs[order][::-1])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([names[i] for i in order][::-1])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Global feature importance from the predefined interpretation model")
    fig.tight_layout()
    fig.savefig(out / "shap_global_importance.png", dpi=600, bbox_inches="tight")
    fig.savefig(out / "shap_global_importance.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    pipeline_path = Path(args.pipeline).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "environment.json", environment_report())
    bundle = joblib.load(pipeline_path)
    X_path = resolve_from_bundle(pipeline_path, bundle["X_path"])
    parent_path = resolve_from_bundle(pipeline_path, bundle["parent_csv"])
    X = np.asarray(np.load(X_path), dtype=np.float32)
    parent = pd.read_csv(parent_path)
    feature_names: list[str] = list(bundle["feature_names"])
    selected_indices = np.asarray(bundle["selected_indices"], dtype=int)
    scaler = bundle["scaler"]
    model = bundle["model"]
    X_selected_scaled = scaler.transform(X[:, selected_indices])

    train_idx = np.asarray(bundle["train_idx"], dtype=int)
    valid_idx = np.asarray(bundle["valid_idx"], dtype=int)
    test_idx = np.asarray(bundle["test_idx"], dtype=int)
    development_idx = np.asarray(bundle["development_idx"], dtype=int)
    if args.explain_scope == "test":
        explain_idx = test_idx
    elif args.explain_scope == "development":
        explain_idx = development_idx
    else:
        explain_idx = np.arange(len(X), dtype=int)
    rng = np.random.default_rng(args.background_seed)
    background_idx = np.sort(rng.choice(train_idx, size=min(args.background_size, len(train_idx)), replace=False))
    background = X_selected_scaled[background_idx]
    explained = X_selected_scaled[explain_idx]

    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    explanation = explainer(explained, check_additivity=False)
    shap_values = np.asarray(explanation.values)
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]
    if shap_values.shape != explained.shape:
        raise ValueError(f"Unexpected SHAP shape {shap_values.shape}; expected {explained.shape}")
    np.save(out / "shap_values.npy", shap_values.astype(np.float32))
    np.save(out / "explained_feature_matrix_scaled.npy", explained.astype(np.float32))
    pd.DataFrame({"array_index": explain_idx}).to_csv(out / "shap_explained_samples.csv", index=False)
    pd.DataFrame({"array_index": background_idx}).to_csv(out / "shap_background_samples.csv", index=False)

    selected_names = [feature_names[i] for i in selected_indices]
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    selected_table = pd.DataFrame({
        "selected_model_input_position": np.arange(len(selected_indices), dtype=int),
        "original_feature_index_zero_based": selected_indices,
        "feature_name_zero_based": selected_names,
        "feature_display_index_one_based": [int(name.split("_")[1]) + 1 for name in selected_names],
        "mean_absolute_shap": mean_abs,
    })
    selected_table["rank_by_mean_absolute_shap"] = selected_table["mean_absolute_shap"].rank(method="first", ascending=False).astype(int)
    selected_table = selected_table.sort_values("rank_by_mean_absolute_shap").reset_index(drop=True)
    selected_table.to_csv(out / "complete_selected_feature_shap_importance.csv", index=False, encoding="utf-8-sig")

    full = pd.DataFrame({
        "original_feature_index_zero_based": np.arange(len(feature_names), dtype=int),
        "feature_name_zero_based": feature_names,
        "feature_display_index_one_based": [int(name.split("_")[1]) + 1 for name in feature_names],
        "molecular_side": [feature_side_and_bit(name)[0] for name in feature_names],
        "selected_by_training_only_MI": False,
        "mean_absolute_shap": np.nan,
        "rank_by_mean_absolute_shap": pd.Series([pd.NA] * len(feature_names), dtype="Int64"),
    })
    full.loc[selected_indices, "selected_by_training_only_MI"] = True
    full.loc[selected_indices, "mean_absolute_shap"] = mean_abs
    rank_map = dict(zip(selected_table["original_feature_index_zero_based"], selected_table["rank_by_mean_absolute_shap"]))
    for idx, rank in rank_map.items():
        full.loc[int(idx), "rank_by_mean_absolute_shap"] = int(rank)
    full.to_csv(out / "complete_2048_feature_importance_archive.csv", index=False, encoding="utf-8-sig")

    top = selected_table.head(args.top_k).copy()
    top["selection_rule"] = f"Top {args.top_k} by mean absolute SHAP from the predefined no-augmentation RandomForest"
    top.to_csv(out / "predefined_top_features.csv", index=False, encoding="utf-8-sig")
    top_names = top["feature_name_zero_based"].tolist()
    top_original_idx = top["original_feature_index_zero_based"].to_numpy(dtype=int)

    save_shap_plots(shap_values, explained, selected_names, out)

    corr = pd.DataFrame(X[:, top_original_idx], columns=top_names).corr(method="pearson")
    corr.to_csv(out / "top_feature_correlation_matrix.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    image = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(top_names)))
    ax.set_yticks(np.arange(len(top_names)))
    ax.set_xticklabels(top_names, rotation=45, ha="right")
    ax.set_yticklabels(top_names)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title("Correlation among the same predefined SHAP-selected features")
    fig.tight_layout()
    fig.savefig(out / "top_feature_correlation_heatmap.png", dpi=600, bbox_inches="tight")
    fig.savefig(out / "top_feature_correlation_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    all_environment_rows = []
    dominant_rows = []
    for row in top.itertuples(index=False):
        name = row.feature_name_zero_based
        side, bit = feature_side_and_bit(name)
        smiles_col = "donor_smiles" if side == "donor" else "acceptor_smiles"
        mapping, summary = map_bit_across_molecules(parent[smiles_col].astype(str).tolist(), bit, int(bundle["radius"]), int(bundle["n_bits"]))
        pair_activation_count = int(np.sum(X[:, int(row.original_feature_index_zero_based)] > 0.5))
        if len(mapping):
            mapping.insert(0, "feature_name_zero_based", name)
            mapping.insert(1, "molecular_side", side)
            mapping.insert(2, "bit_index_zero_based", bit)
            mapping.insert(3, "bit_display_index_one_based", bit + 1)
            mapping.insert(4, "shap_rank", int(row.rank_by_mean_absolute_shap))
            all_environment_rows.append(mapping)
            dominant = mapping.iloc[0].to_dict()
        else:
            dominant = {
                "feature_name_zero_based": name,
                "molecular_side": side,
                "bit_index_zero_based": bit,
                "bit_display_index_one_based": bit + 1,
                "shap_rank": int(row.rank_by_mean_absolute_shap),
                "environment_smiles": "UNMAPPED",
                "environment_occurrence_count": 0,
                "unique_molecule_support": 0,
            }
        dominant.update(summary)
        dominant["n_pair_level_activations"] = pair_activation_count
        dominant["mean_absolute_shap"] = float(row.mean_absolute_shap)
        dominant_rows.append(dominant)
    pd.concat(all_environment_rows, ignore_index=True).to_csv(
        out / "all_top_feature_mapped_environments.csv", index=False, encoding="utf-8-sig"
    ) if all_environment_rows else pd.DataFrame().to_csv(out / "all_top_feature_mapped_environments.csv", index=False)
    pd.DataFrame(dominant_rows).to_csv(out / "top_feature_dominant_environment_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "pipeline": str(pipeline_path),
        "model_name": bundle["model_name"],
        "model_selection_status": bundle["model_selection_status"],
        "feature_selection_fitted_on": bundle["feature_selection_fitted_on"],
        "interpretation_model_training_data": bundle["interpretation_model_training_data"],
        "shap_explainer": "shap.TreeExplainer with interventional perturbation",
        "reference_data": "random subset of the training set",
        "background_seed": args.background_seed,
        "background_size": int(len(background_idx)),
        "explained_scope": args.explain_scope,
        "n_explained_samples": int(len(explain_idx)),
        "feature_selection_rule_for_all_downstream_interpretation": f"Top {args.top_k} by mean absolute SHAP",
        "fingerprint_indexing": "feature names are zero-based; display indices are explicitly archived as one-based",
        "correlation_samples": "all structure-resolved parent pairs",
        "back_mapping_samples": "all unique donor or acceptor molecules in the structure-resolved parent dataset",
    }
    save_json(out / "interpretation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

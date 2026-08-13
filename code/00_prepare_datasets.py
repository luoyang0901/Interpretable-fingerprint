#!/usr/bin/env python3
"""Prepare radius-2 TextCNN arrays and perform overlap/representation audits.

The script removes every pair listed in the baseline development-exclusion file
(scaffold-disjoint test + scaffold-overlap quarantine), generates radius-2
in-house fingerprints, aligns them with Gao fingerprint-only arrays, and writes:
- source metadata and exact representation group IDs;
- pair/donor/acceptor cross-source exact-overlap audits;
- nearest cross-source Tanimoto similarities;
- legacy and role-aware FPtand truncation/collision audits;
- group arrays that prevent identical encoded representations from crossing
  train/validation/test boundaries.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator

from common_utils import (
    binary_row_hashes,
    encode_fptand,
    environment_report,
    empirical_summary,
    factorize_hashes,
    paired_row_hashes,
    paired_similarity_matrix,
    save_json,
    sequence_row_hashes,
    tanimoto_similarity_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inhouse-csv", required=True, help="Structure-resolved CSV with donor/acceptor SMILES and PCE.")
    p.add_argument(
        "--excluded-pairs-csv",
        help="CSV produced by Baseline.py: scaffold test plus quarantine. All listed pairs are removed.",
    )
    p.add_argument("--unknown-pairs-csv", help="Deprecated alias of --excluded-pairs-csv.")
    p.add_argument("--gao-fd", required=True)
    p.add_argument("--gao-fa", required=True)
    p.add_argument("--gao-y", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--n-bits", type=int, default=1024)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--expected-gao", type=int, default=535)
    p.add_argument("--near-similarity-thresholds", default="0.90,0.95,0.99")
    return p


def guess_columns(df: pd.DataFrame) -> tuple[str, str, str]:
    lower = {str(c).lower().strip(): c for c in df.columns}
    donor = next((lower[x] for x in ["donor_smiles", "donor smiles", "donor"] if x in lower), None)
    acceptor = next((lower[x] for x in ["acceptor_smiles", "acceptor smiles", "acceptor"] if x in lower), None)
    pce = next((lower[x] for x in ["pce", "pce (%)", "efficiency"] if x in lower), None)
    if donor is None or acceptor is None or pce is None:
        raise ValueError(f"Could not identify donor/acceptor/PCE columns: {list(df.columns)}")
    return donor, acceptor, pce


def load_inhouse(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    d, a, y = guess_columns(raw)
    df = raw[[d, a, y]].copy()
    df.columns = ["donor_smiles", "acceptor_smiles", "PCE"]
    df["donor_smiles"] = df["donor_smiles"].astype(str).str.strip()
    df["acceptor_smiles"] = df["acceptor_smiles"].astype(str).str.strip()
    df["PCE"] = pd.to_numeric(df["PCE"], errors="coerce")
    df = df.dropna(subset=["PCE"])
    df["pair_key"] = df["donor_smiles"] + "||" + df["acceptor_smiles"]
    # If a raw file is supplied, retain maximum PCE to match the main target definition.
    df = df.sort_values("PCE", ascending=False).drop_duplicates("pair_key", keep="first").reset_index(drop=True)
    return df


def excluded_keys(path: str | Path | None) -> set[str]:
    if path is None:
        return set()
    df = pd.read_csv(path)
    if "pair_key" in df.columns:
        return set(df["pair_key"].astype(str))
    if "DA_Key" in df.columns:
        return set(df["DA_Key"].astype(str).str.replace("__", "||", regex=False))
    d, a, _ = guess_columns(df.assign(PCE=0.0) if "PCE" not in df.columns else df)
    return set(df[d].astype(str) + "||" + df[a].astype(str))


def fingerprint_matrix(smiles_values: Iterable[str], radius: int, n_bits: int) -> np.ndarray:
    try:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    except Exception:
        generator = None
    rows = []
    for idx, smiles in enumerate(smiles_values):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            raise ValueError(f"Invalid SMILES at row {idx}: {smiles}")
        fp = generator.GetFingerprint(mol) if generator is not None else AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros(n_bits, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return np.vstack(rows).astype(np.int8)


def validate_gao(fd: np.ndarray, fa: np.ndarray, y: np.ndarray, expected_n: int, n_bits: int) -> None:
    if fd.ndim != 2 or fa.ndim != 2 or fd.shape != fa.shape or fd.shape[1] != n_bits:
        raise ValueError(f"Unexpected Gao shapes: fd={fd.shape}, fa={fa.shape}")
    if len(y.reshape(-1)) != len(fd):
        raise ValueError(f"Gao target length mismatch: {y.shape} vs {fd.shape}")
    if expected_n > 0 and len(fd) != expected_n:
        raise ValueError(f"Expected {expected_n} Gao rows, found {len(fd)}")
    for name, arr in [("gao_fd", fd), ("gao_fa", fa)]:
        values = set(np.unique(arr).tolist())
        if not values.issubset({0, 1}):
            raise ValueError(f"{name} is not binary; values={sorted(values)[:10]}")



def union_group_ids(*hash_arrays: np.ndarray) -> np.ndarray:
    """Transitive union of equality groups across several representations."""
    n = len(hash_arrays[0])
    if any(len(x) != n for x in hash_arrays):
        raise ValueError("hash arrays must have equal length")
    parent = np.arange(n, dtype=int)
    rank = np.zeros(n, dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for hashes in hash_arrays:
        first: dict[str, int] = {}
        for i, value in enumerate(hashes):
            key = str(value)
            if key in first:
                union(first[key], i)
            else:
                first[key] = i
    roots = np.asarray([find(i) for i in range(n)], dtype=int)
    codes, _ = pd.factorize(roots, sort=True)
    return codes.astype(np.int64)


def representation_group_table(meta: pd.DataFrame, hash_col: str, representation: str) -> pd.DataFrame:
    rows = []
    for value, sub in meta.groupby(hash_col, sort=True):
        if len(sub) < 2:
            continue
        rows.append(
            {
                "representation": representation,
                "representation_hash": value,
                "group_size": int(len(sub)),
                "n_inhouse": int((sub["source"] == "in-house").sum()),
                "n_gao": int((sub["source"] == "Gao").sum()),
                "cross_source": bool(sub["source"].nunique() > 1),
                "pce_min": float(sub["experimental_PCE"].min()),
                "pce_max": float(sub["experimental_PCE"].max()),
                "pce_range": float(sub["experimental_PCE"].max() - sub["experimental_PCE"].min()),
                "pce_mean": float(sub["experimental_PCE"].mean()),
                "pce_sd": float(sub["experimental_PCE"].std(ddof=1)) if len(sub) > 1 else 0.0,
                "array_indices": ";".join(map(str, sub["array_index"].astype(int).tolist())),
                "sources": ";".join(sub["source"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)


def nearest_cross_source_table(
    fd_in: np.ndarray, fa_in: np.ndarray, fd_gao: np.ndarray, fa_gao: np.ndarray
) -> pd.DataFrame:
    d = tanimoto_similarity_matrix(fd_in, fd_gao)
    a = tanimoto_similarity_matrix(fa_in, fa_gao)
    p = 0.5 * (d + a)
    rows = []
    for i in range(len(fd_in)):
        j = int(np.argmax(p[i]))
        rows.append({
            "query_source": "in-house", "query_index": i, "nearest_other_source_index": j,
            "donor_similarity": float(d[i, j]), "acceptor_similarity": float(a[i, j]),
            "paired_similarity": float(p[i, j]),
        })
    for j in range(len(fd_gao)):
        i = int(np.argmax(p[:, j]))
        rows.append({
            "query_source": "Gao", "query_index": j, "nearest_other_source_index": i,
            "donor_similarity": float(d[i, j]), "acceptor_similarity": float(a[i, j]),
            "paired_similarity": float(p[i, j]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "environment.json", environment_report())

    inhouse = load_inhouse(args.inhouse_csv)
    exclusion_path = args.excluded_pairs_csv or args.unknown_pairs_csv
    keys = excluded_keys(exclusion_path)
    removed_mask = inhouse["pair_key"].isin(keys)
    filtered = inhouse.loc[~removed_mask].copy().reset_index(drop=False).rename(columns={"index": "original_inhouse_row"})
    excluded_actual = inhouse.loc[removed_mask].copy()
    excluded_actual.to_csv(out / "excluded_inhouse_pairs_confirmed.csv", index=False, encoding="utf-8-sig")

    fd_in = fingerprint_matrix(filtered["donor_smiles"], args.radius, args.n_bits)
    fa_in = fingerprint_matrix(filtered["acceptor_smiles"], args.radius, args.n_bits)
    y_in = filtered["PCE"].to_numpy(dtype=np.float32)
    gao_fd = np.asarray(np.load(args.gao_fd)).astype(np.int8)
    gao_fa = np.asarray(np.load(args.gao_fa)).astype(np.int8)
    gao_y = np.asarray(np.load(args.gao_y)).reshape(-1).astype(np.float32)
    validate_gao(gao_fd, gao_fa, gao_y, args.expected_gao, args.n_bits)

    joint_fd = np.vstack([fd_in, gao_fd]).astype(np.int8)
    joint_fa = np.vstack([fa_in, gao_fa]).astype(np.int8)
    joint_y = np.concatenate([y_in, gao_y]).astype(np.float32)

    for name, arr in [
        ("inhouse_r2_fd.npy", fd_in), ("inhouse_r2_fa.npy", fa_in), ("inhouse_r2_y.npy", y_in),
        ("gao_fd.npy", gao_fd), ("gao_fa.npy", gao_fa), ("gao_y.npy", gao_y),
        ("joint_fd.npy", joint_fd), ("joint_fa.npy", joint_fa), ("joint_y.npy", joint_y),
    ]:
        np.save(out / name, arr)

    in_meta = pd.DataFrame({
        "array_index": np.arange(len(filtered), dtype=int),
        "source": "in-house",
        "source_row": filtered["original_inhouse_row"].astype(int),
        "donor_smiles": filtered["donor_smiles"].astype(str),
        "acceptor_smiles": filtered["acceptor_smiles"].astype(str),
        "experimental_PCE": y_in,
        "pair_key": filtered["pair_key"].astype(str),
    })
    gao_meta = pd.DataFrame({
        "array_index": np.arange(len(gao_y), dtype=int),
        "source": "Gao",
        "source_row": np.arange(len(gao_y), dtype=int),
        "donor_smiles": "",
        "acceptor_smiles": "",
        "experimental_PCE": gao_y,
        "pair_key": [f"Gao_{i:04d}" for i in range(len(gao_y))],
    })
    joint_meta = pd.concat([
        in_meta.assign(array_index=np.arange(len(in_meta), dtype=int)),
        gao_meta.assign(array_index=np.arange(len(in_meta), len(in_meta) + len(gao_meta), dtype=int)),
    ], ignore_index=True)

    donor_hash = binary_row_hashes(joint_fd)
    acceptor_hash = binary_row_hashes(joint_fa)
    pair_hash = paired_row_hashes(joint_fd, joint_fa)
    legacy_seq, legacy_audit, _ = encode_fptand(joint_fd, joint_fa, args.max_len, "legacy")
    role_seq, role_audit, _ = encode_fptand(joint_fd, joint_fa, args.max_len, "role_aware")
    legacy_hash = sequence_row_hashes(legacy_seq)
    role_hash = sequence_row_hashes(role_seq)
    joint_meta["donor_fp_hash"] = donor_hash
    joint_meta["acceptor_fp_hash"] = acceptor_hash
    joint_meta["pair_fp_hash"] = pair_hash
    joint_meta["legacy_sequence_hash"] = legacy_hash
    joint_meta["role_aware_sequence_hash"] = role_hash
    joint_meta["pair_fp_group_id"] = factorize_hashes(pair_hash)
    joint_meta["legacy_sequence_group_id"] = factorize_hashes(legacy_hash)
    joint_meta["role_aware_sequence_group_id"] = factorize_hashes(role_hash)
    joint_meta["safe_any_encoding_group_id"] = union_group_ids(pair_hash, legacy_hash, role_hash)

    in_meta_full = joint_meta.iloc[:len(in_meta)].copy().reset_index(drop=True)
    gao_meta_full = joint_meta.iloc[len(in_meta):].copy().reset_index(drop=True)
    gao_meta_full["array_index"] = np.arange(len(gao_meta_full), dtype=int)
    in_meta_full.to_csv(out / "inhouse_r2_sample_metadata.csv", index=False, encoding="utf-8-sig")
    gao_meta_full.to_csv(out / "gao_sample_metadata.csv", index=False, encoding="utf-8-sig")
    joint_meta.to_csv(out / "joint_sample_metadata.csv", index=False, encoding="utf-8-sig")
    filtered.to_csv(out / "inhouse_r2_filtered.csv", index=False, encoding="utf-8-sig")

    np.save(out / "joint_pair_fp_group_ids.npy", joint_meta["pair_fp_group_id"].to_numpy(dtype=np.int64))
    np.save(out / "joint_legacy_sequence_group_ids.npy", joint_meta["legacy_sequence_group_id"].to_numpy(dtype=np.int64))
    np.save(out / "joint_role_aware_sequence_group_ids.npy", joint_meta["role_aware_sequence_group_id"].to_numpy(dtype=np.int64))
    np.save(out / "joint_safe_any_encoding_group_ids.npy", joint_meta["safe_any_encoding_group_id"].to_numpy(dtype=np.int64))
    np.save(out / "inhouse_pair_fp_group_ids.npy", in_meta_full["pair_fp_group_id"].to_numpy(dtype=np.int64))
    np.save(out / "inhouse_legacy_sequence_group_ids.npy", in_meta_full["legacy_sequence_group_id"].to_numpy(dtype=np.int64))
    np.save(out / "inhouse_role_aware_sequence_group_ids.npy", in_meta_full["role_aware_sequence_group_id"].to_numpy(dtype=np.int64))
    np.save(out / "inhouse_safe_any_encoding_group_ids.npy", in_meta_full["safe_any_encoding_group_id"].to_numpy(dtype=np.int64))
    np.save(out / "gao_pair_fp_group_ids.npy", gao_meta_full["pair_fp_group_id"].to_numpy(dtype=np.int64))
    np.save(out / "gao_legacy_sequence_group_ids.npy", gao_meta_full["legacy_sequence_group_id"].to_numpy(dtype=np.int64))
    np.save(out / "gao_role_aware_sequence_group_ids.npy", gao_meta_full["role_aware_sequence_group_id"].to_numpy(dtype=np.int64))
    np.save(out / "gao_safe_any_encoding_group_ids.npy", gao_meta_full["safe_any_encoding_group_id"].to_numpy(dtype=np.int64))

    all_groups = []
    for col, label in [
        ("donor_fp_hash", "donor_fingerprint"),
        ("acceptor_fp_hash", "acceptor_fingerprint"),
        ("pair_fp_hash", "paired_fingerprint"),
        ("legacy_sequence_hash", "legacy_truncated_sequence"),
        ("role_aware_sequence_hash", "role_aware_truncated_sequence"),
    ]:
        table = representation_group_table(joint_meta, col, label)
        if len(table):
            all_groups.append(table)
    group_table = pd.concat(all_groups, ignore_index=True) if all_groups else pd.DataFrame()
    group_table.to_csv(out / "identical_representation_groups.csv", index=False, encoding="utf-8-sig")
    cross_groups = group_table.loc[group_table.get("cross_source", pd.Series(dtype=bool)) == True].copy() if len(group_table) else pd.DataFrame()
    cross_groups.to_csv(out / "cross_source_exact_overlap_groups.csv", index=False, encoding="utf-8-sig")

    nearest = nearest_cross_source_table(fd_in, fa_in, gao_fd, gao_fa)
    nearest.to_csv(out / "cross_source_nearest_similarity.csv", index=False, encoding="utf-8-sig")
    thresholds = [float(x.strip()) for x in args.near_similarity_thresholds.split(",") if x.strip()]
    near_summary = {
        "nearest_similarity_distributions": {
            col: empirical_summary(nearest[col].to_numpy())
            for col in ["donor_similarity", "acceptor_similarity", "paired_similarity"]
        },
        "counts_at_or_above_threshold": {
            str(t): {
                col: int((nearest[col] >= t).sum())
                for col in ["donor_similarity", "acceptor_similarity", "paired_similarity"]
            }
            for t in thresholds
        },
    }

    legacy_audit = legacy_audit.merge(joint_meta[["array_index", "source"]], on="array_index", how="left")
    role_audit = role_audit.merge(joint_meta[["array_index", "source"]], on="array_index", how="left")
    pd.concat([legacy_audit, role_audit], ignore_index=True).to_csv(
        out / "fptand_encoding_audit_per_sample.csv", index=False, encoding="utf-8-sig"
    )
    encoding_summary_rows = []
    for mode_df in [legacy_audit, role_audit]:
        for source, sub in mode_df.groupby("source"):
            encoding_summary_rows.append({
                "encoding_mode": mode_df["encoding_mode"].iloc[0],
                "source": source,
                "n": int(len(sub)),
                "truncated_n": int(sub["truncated"].sum()),
                "truncated_fraction": float(sub["truncated"].mean()),
                "mean_donor_active_bits": float(sub["donor_active_bits"].mean()),
                "mean_acceptor_active_bits": float(sub["acceptor_active_bits"].mean()),
                "mean_donor_tokens_retained": float(sub["donor_tokens_retained"].mean()),
                "mean_acceptor_tokens_retained": float(sub["acceptor_tokens_retained"].mean()),
                "total_donor_tokens_dropped": int(sub["donor_tokens_dropped"].sum()),
                "total_acceptor_tokens_dropped": int(sub["acceptor_tokens_dropped"].sum()),
            })
    pd.DataFrame(encoding_summary_rows).to_csv(out / "fptand_encoding_summary.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "radius": args.radius,
        "n_bits": args.n_bits,
        "max_len": args.max_len,
        "inhouse_parent_n": int(len(inhouse)),
        "excluded_requested_n": int(len(keys)),
        "excluded_matched_n": int(removed_mask.sum()),
        "inhouse_after_exclusion_n": int(len(fd_in)),
        "gao_n": int(len(gao_y)),
        "joint_n": int(len(joint_y)),
        "cross_source_exact_group_counts": (
            cross_groups.groupby("representation").size().astype(int).to_dict() if len(cross_groups) else {}
        ),
        "all_identical_group_counts": (
            group_table.groupby("representation").size().astype(int).to_dict() if len(group_table) else {}
        ),
        "near_similarity_summary": near_summary,
        "identical_representations_must_be_grouped_during_splitting": True,
    }
    save_json(out / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

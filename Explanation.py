import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

FEATURE_RE = re.compile(r'^(fd|fa)_(\d+)$')

# =========================
# 用户直接填写区（不想写命令行时，用这里）
# =========================
USER_DONOR_SMILES = "CC1=CC=C(C2=C(F)C(F)=C(C3=CC=C(C4=CC5=C(C6=CC=C(CC(CC)CC)S6)C7=C(C=C(S7)C)C(C8=CC=C(CC(CC)CC)S8)=C5S4)S3)C9=NC(C%10=CC(F)=C(CC(CC)CC)S%10)=C(C%11=CC(F)=C(CC(CC)CC)S%11)N=C29)S1"
USER_ACCEPTOR_SMILES = "O=C(C/1=C\C2=C(C3=C(C4=C(C5=C(N4C6=CC=C(C=C6)CCCCCC)C(N(C7=C8SC9=C7SC(/C=C%10C(C%11=CC=CC=C%11C\%10=C(C#N)/C#N)=O)=C9C%12=CC=C(C=C%12)CCCCCC)C%13=CC=C(C=C%13)CCCCCC)=C8C%14=NSN=C5%14)S3)S2)C%15=CC=C(C=C%15)CCCCCC)C%16=CC=CC=C%16C1=C(C#N)/C#N"
USER_FEATURE_LABELS: List[str] = []
USER_RADIUS = 3
USER_N_BITS = 1024
USER_OUTDIR = "bit_backmapping_output"
USER_SAVE_IMAGES = True
USER_MAX_IMAGES_PER_FEATURE = 5

# 数据集模式默认输入：你只要把 DAreal.csv 放在脚本同目录，或命令行提供 --pairs-csv 即可
USER_PAIRS_CSV = "DAreal.csv"
USER_OUTPUT_CSV = "unknown_20_metadata_with_onbits.csv"

# SHAP 图里用到的代表性 feature（按你当前图的顺序）
USER_SHAP_FEATURES: List[str] = [
    "fa_764", "fd_723", "fd_527", "fa_523", "fd_978",
    "fa_74", "fd_82", "fa_188", "fd_875", "fa_536",
]


def smiles_to_morgan_bits_with_bitinfo(
    smiles: str,
    radius: int = 3,
    n_bits: int = 1024,
) -> Tuple[Chem.Mol, np.ndarray, Dict[int, List[Tuple[int, int]]]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    bit_info: Dict[int, List[Tuple[int, int]]] = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits,
        bitInfo=bit_info,
    )
    bits = np.array(list(map(int, fp.ToBitString())), dtype=np.int8)
    return mol, bits, bit_info


def build_paired_fingerprint_with_bitinfo(
    donor_smiles: str,
    acceptor_smiles: str,
    radius: int = 3,
    n_bits: int = 1024,
) -> Dict[str, object]:
    donor_mol, donor_bits, donor_bit_info = smiles_to_morgan_bits_with_bitinfo(
        donor_smiles, radius=radius, n_bits=n_bits
    )
    acceptor_mol, acceptor_bits, acceptor_bit_info = smiles_to_morgan_bits_with_bitinfo(
        acceptor_smiles, radius=radius, n_bits=n_bits
    )

    donor_feature_names = [f"fd_{i}" for i in range(n_bits)]
    acceptor_feature_names = [f"fa_{i}" for i in range(n_bits)]

    paired_vector = np.concatenate([donor_bits, acceptor_bits], axis=0)
    paired_feature_names = donor_feature_names + acceptor_feature_names

    donor_on_bits = np.where(donor_bits == 1)[0].tolist()
    acceptor_on_bits = np.where(acceptor_bits == 1)[0].tolist()

    return {
        "donor_mol": donor_mol,
        "acceptor_mol": acceptor_mol,
        "donor_bits_1024": donor_bits,
        "acceptor_bits_1024": acceptor_bits,
        "paired_bits_2048": paired_vector,
        "donor_feature_names": donor_feature_names,
        "acceptor_feature_names": acceptor_feature_names,
        "paired_feature_names": paired_feature_names,
        "donor_on_bits": donor_on_bits,
        "acceptor_on_bits": acceptor_on_bits,
        "donor_bit_info": donor_bit_info,
        "acceptor_bit_info": acceptor_bit_info,
    }


def parse_feature_label(feature_label: str) -> Tuple[str, int]:
    m = FEATURE_RE.match(feature_label.strip())
    if not m:
        raise ValueError(
            f"Invalid feature label: {feature_label}. Expected format like fd_723 or fa_764."
        )
    prefix, bit_id_text = m.groups()
    side = "donor" if prefix == "fd" else "acceptor"
    return side, int(bit_id_text)


def atom_environment_to_fragment(
    mol: Chem.Mol,
    atom_idx: int,
    env_radius: int,
) -> Dict[str, object]:
    if env_radius < 0:
        raise ValueError("env_radius must be >= 0")

    bond_indices = []
    atom_indices = {atom_idx}

    if env_radius > 0:
        bond_indices = list(Chem.FindAtomEnvironmentOfRadiusN(mol, env_radius, atom_idx))
        for bond_idx in bond_indices:
            bond = mol.GetBondWithIdx(bond_idx)
            atom_indices.add(bond.GetBeginAtomIdx())
            atom_indices.add(bond.GetEndAtomIdx())

    atom_indices_sorted = sorted(atom_indices)

    if bond_indices:
        fragment_smiles = Chem.MolFragmentToSmiles(
            mol,
            atomsToUse=atom_indices_sorted,
            bondsToUse=bond_indices,
            rootedAtAtom=atom_idx,
            canonical=True,
        )
    else:
        atom = mol.GetAtomWithIdx(atom_idx)
        fragment_smiles = atom.GetSymbol()

    atom = mol.GetAtomWithIdx(atom_idx)
    return {
        "center_atom_idx": atom_idx,
        "center_atom_symbol": atom.GetSymbol(),
        "environment_radius": env_radius,
        "atom_indices": atom_indices_sorted,
        "bond_indices": sorted(bond_indices),
        "n_atoms": len(atom_indices_sorted),
        "n_bonds": len(bond_indices),
        "fragment_smiles": fragment_smiles,
    }


def save_highlight_image(
    mol: Chem.Mol,
    atom_indices: List[int],
    bond_indices: List[int],
    output_path: str,
    legend: Optional[str] = None,
    size: Tuple[int, int] = (900, 650),
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Draw.MolToFile(
        mol,
        output_path,
        size=size,
        highlightAtoms=atom_indices,
        highlightBonds=bond_indices,
        legend=legend or "",
    )


def map_feature_in_single_molecule(
    smiles: str,
    feature_label: str,
    radius: int = 3,
    n_bits: int = 1024,
    molecule_id: Optional[str] = None,
    save_images: bool = False,
    image_dir: Optional[str] = None,
) -> List[Dict[str, object]]:
    side, bit_id = parse_feature_label(feature_label)
    mol, bits, bit_info = smiles_to_morgan_bits_with_bitinfo(smiles, radius=radius, n_bits=n_bits)

    rows: List[Dict[str, object]] = []
    if bits[bit_id] != 1:
        return rows

    for env_index, (atom_idx, env_radius) in enumerate(bit_info.get(bit_id, []), start=1):
        env = atom_environment_to_fragment(mol, atom_idx, env_radius)
        image_path = None
        if save_images and image_dir:
            safe_mol_id = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(molecule_id or 'molecule'))
            image_path = os.path.join(
                image_dir,
                f"{feature_label}_{safe_mol_id}_env{env_index}.png",
            )
            save_highlight_image(
                mol,
                env["atom_indices"],
                env["bond_indices"],
                image_path,
                legend=f"{feature_label} | atom={atom_idx} | r={env_radius}",
            )

        rows.append(
            {
                "molecule_id": molecule_id,
                "smiles": smiles,
                "feature_label": feature_label,
                "molecular_side": side,
                "bit_id": bit_id,
                "center_atom_idx": env["center_atom_idx"],
                "center_atom_symbol": env["center_atom_symbol"],
                "environment_radius": env["environment_radius"],
                "n_atoms": env["n_atoms"],
                "n_bonds": env["n_bonds"],
                "atom_indices": json.dumps(env["atom_indices"], ensure_ascii=False),
                "bond_indices": json.dumps(env["bond_indices"], ensure_ascii=False),
                "fragment_smiles": env["fragment_smiles"],
                "highlight_image": image_path,
            }
        )
    return rows


def summarize_feature_details(details_df: pd.DataFrame) -> pd.DataFrame:
    if details_df.empty:
        return pd.DataFrame(
            columns=[
                "feature_label",
                "molecular_side",
                "fragment_smiles",
                "count_occurrences",
                "unique_molecules",
                "support_fraction",
                "interpretation_note",
            ]
        )

    group = (
        details_df.groupby(["feature_label", "molecular_side", "fragment_smiles"], dropna=False)
        .agg(
            count_occurrences=("fragment_smiles", "size"),
            unique_molecules=("molecule_id", "nunique"),
        )
        .reset_index()
    )

    total_by_feature = (
        details_df.groupby(["feature_label", "molecular_side"], dropna=False)
        .agg(total_occurrences=("fragment_smiles", "size"))
        .reset_index()
    )

    merged = group.merge(total_by_feature, on=["feature_label", "molecular_side"], how="left")
    merged["support_fraction"] = merged["count_occurrences"] / merged["total_occurrences"]

    interpretation_notes: List[str] = []
    for _, row in merged.iterrows():
        frac = float(row["support_fraction"])
        if frac >= 0.70:
            note = "dominant mapped environment family"
        elif frac >= 0.40:
            note = "small family of related mapped environments"
        else:
            note = "heterogeneous hashed environments; avoid unique assignment"
        interpretation_notes.append(note)
    merged["interpretation_note"] = interpretation_notes

    return merged.sort_values(
        ["feature_label", "count_occurrences", "fragment_smiles"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def build_table_s17_ready(summary_df: pd.DataFrame, feature_labels: List[str]) -> pd.DataFrame:
    target_order = list(feature_labels)
    if summary_df.empty:
        empty = pd.DataFrame(
            {
                "feature_label": target_order,
                "molecular_side": ["Donor side" if x.startswith("fd_") else "Acceptor side" for x in target_order],
                "observed_in_dataset_level_mapping": ["No"] * len(target_order),
                "dominant_mapped_environment": ["Not recovered in the structure-resolved dataset-level run"] * len(target_order),
                "support_in_mapped_occurrences": ["0 of 0"] * len(target_order),
                "interpretation_note": ["Feature not activated in the structure-resolved dataset-level back-mapping run"] * len(target_order),
            }
        )
        return empty

    first_rows = (
        summary_df.sort_values(["feature_label", "count_occurrences"], ascending=[True, False])
        .groupby(["feature_label", "molecular_side"], as_index=False)
        .first()
    )
    first_rows["support_in_mapped_occurrences"] = first_rows.apply(
        lambda r: f"{int(r['count_occurrences'])} of {int(r['total_occurrences'])}", axis=1
    )
    first_rows["observed_in_dataset_level_mapping"] = "Yes"
    table = first_rows[
        [
            "feature_label",
            "molecular_side",
            "observed_in_dataset_level_mapping",
            "fragment_smiles",
            "support_in_mapped_occurrences",
            "interpretation_note",
        ]
    ].rename(columns={
        "fragment_smiles": "dominant_mapped_environment",
        "molecular_side": "molecular_side_raw",
    })
    table["molecular_side"] = table["molecular_side_raw"].map(
        {"donor": "Donor side", "acceptor": "Acceptor side"}
    )
    table = table.drop(columns=["molecular_side_raw"])

    existing = set(table["feature_label"].tolist())
    missing_rows = []
    for feat in target_order:
        if feat not in existing:
            missing_rows.append(
                {
                    "feature_label": feat,
                    "molecular_side": "Donor side" if feat.startswith("fd_") else "Acceptor side",
                    "observed_in_dataset_level_mapping": "No",
                    "dominant_mapped_environment": "Not recovered in the structure-resolved dataset-level run",
                    "support_in_mapped_occurrences": "0 of 0",
                    "interpretation_note": "Feature not activated in the structure-resolved dataset-level back-mapping run",
                }
            )
    if missing_rows:
        table = pd.concat([table, pd.DataFrame(missing_rows)], ignore_index=True)

    table["feature_label"] = pd.Categorical(table["feature_label"], categories=target_order, ordered=True)
    table = table.sort_values("feature_label").reset_index(drop=True)
    return table


def infer_smiles_columns(df: pd.DataFrame, donor_col: Optional[str], acceptor_col: Optional[str]) -> Tuple[str, str]:
    cols = list(df.columns)
    if donor_col and acceptor_col:
        return donor_col, acceptor_col

    def find_col(candidates: List[str]) -> Optional[str]:
        lower_map = {str(c).lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        for c in cols:
            lc = str(c).lower()
            for cand in candidates:
                if cand.lower() in lc:
                    return c
        return None

    donor = donor_col or find_col(["donor_smiles", "donor smiles", "donor", "smiles_donor"])
    acceptor = acceptor_col or find_col(["acceptor_smiles", "acceptor smiles", "acceptor", "smiles_acceptor"])

    if donor is None or acceptor is None:
        raise ValueError(
            f"Could not infer donor/acceptor columns from: {cols}. Please provide --donor-col and --acceptor-col."
        )
    return donor, acceptor


def infer_sample_id_column(df: pd.DataFrame, sample_id_col: Optional[str]) -> Optional[str]:
    if sample_id_col:
        return sample_id_col
    cols = list(df.columns)
    lower_map = {str(c).lower(): c for c in cols}
    for cand in ["sample_id", "sample id", "id", "pair_id", "pair id"]:
        if cand in lower_map:
            return lower_map[cand]
    return None


def plot_representative_feature_recovery(
    table_s17_df: pd.DataFrame,
    out_png: str,
    out_pdf: Optional[str] = None,
) -> None:
    plot_df = table_s17_df.copy()
    plot_df["mapped_occurrences"] = plot_df["support_in_mapped_occurrences"].map(
        lambda s: int(str(s).split(" of ")[0]) if " of " in str(s) else 0
    )
    plt.figure(figsize=(8.8, 5.6))
    y = np.arange(len(plot_df))
    bars = plt.barh(y, plot_df["mapped_occurrences"].values)
    plt.yticks(y, plot_df["feature_label"].tolist())
    plt.xlabel("Mapped occurrences across structure-resolved dataset-level back-mapping")
    plt.ylabel("Representative fingerprint feature")
    plt.tight_layout()

    for bar, value in zip(bars, plot_df["mapped_occurrences"].values):
        plt.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf:
        plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()


def map_shap_features_in_dareal_dataset(
    csv_path: str,
    feature_labels: List[str],
    donor_col: Optional[str] = None,
    acceptor_col: Optional[str] = None,
    sample_id_col: Optional[str] = None,
    radius: int = 3,
    n_bits: int = 1024,
    outdir: str = "bit_backmapping_output",
    save_images: bool = True,
    max_images_per_feature: int = 5,
) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    donor_col, acceptor_col = infer_smiles_columns(df, donor_col, acceptor_col)
    sample_id_col = infer_sample_id_column(df, sample_id_col)

    work_df = df.copy()
    if sample_id_col is None:
        sample_id_col = "sample_id"
        work_df[sample_id_col] = [f"row_{i}" for i in range(len(work_df))]

    all_rows: List[Dict[str, object]] = []
    feature_image_counter = Counter()
    image_dir = os.path.join(outdir, "images")

    for _, row in work_df.iterrows():
        donor_smiles = str(row[donor_col]).strip()
        acceptor_smiles = str(row[acceptor_col]).strip()
        sample_id = str(row[sample_id_col])

        for feature_label in feature_labels:
            side, _ = parse_feature_label(feature_label)
            smiles = donor_smiles if side == "donor" else acceptor_smiles
            allow_image = save_images and feature_image_counter[feature_label] < max_images_per_feature
            mapped = map_feature_in_single_molecule(
                smiles=smiles,
                feature_label=feature_label,
                radius=radius,
                n_bits=n_bits,
                molecule_id=sample_id,
                save_images=allow_image,
                image_dir=image_dir if allow_image else None,
            )
            if mapped:
                feature_image_counter[feature_label] += 1

            for item in mapped:
                item["sample_id"] = sample_id
                item["donor_smiles"] = donor_smiles
                item["acceptor_smiles"] = acceptor_smiles
                all_rows.append(item)

    details_df = pd.DataFrame(all_rows)
    summary_df = summarize_feature_details(details_df)
    table_s17_df = build_table_s17_ready(summary_df, feature_labels=feature_labels)

    os.makedirs(outdir, exist_ok=True)
    details_path = os.path.join(outdir, "dataset_bit_environment_details.csv")
    summary_path = os.path.join(outdir, "dataset_bit_environment_summary.csv")
    table_s17_path = os.path.join(outdir, "Table_S17_ready.csv")
    figure_png = os.path.join(outdir, "Figure_S10_representative_bit_recovery_dataset_level.png")
    figure_pdf = os.path.join(outdir, "Figure_S10_representative_bit_recovery_dataset_level.pdf")

    details_df.to_csv(details_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    table_s17_df.to_csv(table_s17_path, index=False, encoding="utf-8-sig")
    plot_representative_feature_recovery(table_s17_df, figure_png, figure_pdf)

    meta = {
        "input_csv": os.path.abspath(csv_path),
        "donor_col": donor_col,
        "acceptor_col": acceptor_col,
        "sample_id_col": sample_id_col,
        "n_rows": int(len(work_df)),
        "feature_labels": feature_labels,
        "radius": int(radius),
        "n_bits": int(n_bits),
        "details_path": os.path.abspath(details_path),
        "summary_path": os.path.abspath(summary_path),
        "table_s17_ready_path": os.path.abspath(table_s17_path),
        "figure_png": os.path.abspath(figure_png),
        "figure_pdf": os.path.abspath(figure_pdf),
        "mode": "dataset_level_shap_features_only",
    }
    meta_path = os.path.join(outdir, "dataset_bit_environment_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return {
        "details_path": details_path,
        "summary_path": summary_path,
        "table_s17_ready_path": table_s17_path,
        "figure_png": figure_png,
        "figure_pdf": figure_pdf,
        "meta_path": meta_path,
    }


def format_on_bits_list(bits: List[int]) -> str:
    return "[" + ",".join(str(int(x)) for x in bits) + "]"


def fill_on_bits_in_csv(
    csv_path: str,
    donor_col: Optional[str] = None,
    acceptor_col: Optional[str] = None,
    radius: int = 3,
    n_bits: int = 1024,
    out_csv: Optional[str] = None,
) -> Dict[str, str]:
    df = pd.read_csv(csv_path)
    donor_col, acceptor_col = infer_smiles_columns(df, donor_col, acceptor_col)

    donor_onbits_col = f"Donor on-bits ({n_bits})"
    acceptor_onbits_col = f"Acceptor on-bits ({n_bits})"
    donor_count_col = f"Donor on-bit count ({n_bits}-bit FP)"
    acceptor_count_col = f"Acceptor on-bit count ({n_bits}-bit FP)"

    donor_onbits_values: List[str] = []
    acceptor_onbits_values: List[str] = []
    donor_count_values: List[int] = []
    acceptor_count_values: List[int] = []

    for _, row in df.iterrows():
        donor_smiles = str(row[donor_col]).strip()
        acceptor_smiles = str(row[acceptor_col]).strip()

        paired = build_paired_fingerprint_with_bitinfo(
            donor_smiles=donor_smiles,
            acceptor_smiles=acceptor_smiles,
            radius=radius,
            n_bits=n_bits,
        )

        donor_on_bits = paired["donor_on_bits"]
        acceptor_on_bits = paired["acceptor_on_bits"]

        donor_onbits_values.append(format_on_bits_list(donor_on_bits))
        acceptor_onbits_values.append(format_on_bits_list(acceptor_on_bits))
        donor_count_values.append(len(donor_on_bits))
        acceptor_count_values.append(len(acceptor_on_bits))

    out_df = df.copy()
    out_df[donor_onbits_col] = donor_onbits_values
    out_df[acceptor_onbits_col] = acceptor_onbits_values
    out_df[donor_count_col] = donor_count_values
    out_df[acceptor_count_col] = acceptor_count_values

    if out_csv is None:
        base, ext = os.path.splitext(csv_path)
        out_csv = f"{base}_with_onbits{ext or '.csv'}"

    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    meta = {
        "input_csv": os.path.abspath(csv_path),
        "output_csv": os.path.abspath(out_csv),
        "detected_donor_col": donor_col,
        "detected_acceptor_col": acceptor_col,
        "n_rows": int(len(out_df)),
        "radius": int(radius),
        "n_bits": int(n_bits),
        "donor_onbits_column": donor_onbits_col,
        "acceptor_onbits_column": acceptor_onbits_col,
        "donor_count_column": donor_count_col,
        "acceptor_count_column": acceptor_count_col,
    }
    meta_path = os.path.splitext(out_csv)[0] + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return {"output_csv": out_csv, "meta_path": meta_path}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="基于 RDKit bitInfo 对 Morgan fingerprint bit 做 atom environment 回溯、数据集级 SHAP feature 汇总与绘图。"
    )
    parser.add_argument("--donor-smiles", type=str, default=None, help="单对 D/A 模式下的 donor SMILES")
    parser.add_argument("--acceptor-smiles", type=str, default=None, help="单对 D/A 模式下的 acceptor SMILES")
    parser.add_argument(
        "--feature-labels",
        nargs="+",
        default=None,
        help="要回溯的特征标签，例如 fd_723 fa_764。若不提供，则在 SHAP 数据集模式下使用内置 USER_SHAP_FEATURES。",
    )
    parser.add_argument("--pairs-csv", type=str, default=None, help="全数据集模式下的 CSV 文件路径（例如 DAreal.csv）")
    parser.add_argument("--fill-onbits-csv", action="store_true", help="读取 pairs-csv，重新生成 donor/acceptor on-bits 列表并导出新 CSV")
    parser.add_argument("--dataset-shap-mode", action="store_true", help="只针对 SHAP 图中提到的代表性 fx_xxx 在 DAreal.csv 上做数据集级回溯，并重建 Table S17 与 Figure S10")
    parser.add_argument("--output-csv", type=str, default=None, help="fill-onbits-csv 模式下的输出 CSV 文件名")
    parser.add_argument("--donor-col", type=str, default=None, help="CSV 中 donor SMILES 列名")
    parser.add_argument("--acceptor-col", type=str, default=None, help="CSV 中 acceptor SMILES 列名")
    parser.add_argument("--sample-id-col", type=str, default=None, help="CSV 中样本 ID 列名")
    parser.add_argument("--radius", type=int, default=None, help="Morgan fingerprint 半径，默认读取用户填写区")
    parser.add_argument("--n-bits", type=int, default=None, help="Morgan fingerprint 位数，默认读取用户填写区")
    parser.add_argument("--outdir", type=str, default=None, help="输出目录，默认读取用户填写区")
    parser.add_argument("--save-images", action="store_true", help="是否导出高亮环境图片")
    parser.add_argument(
        "--max-images-per-feature",
        type=int,
        default=None,
        help="数据集模式下，每个特征最多导出几张代表图片，默认读取用户填写区",
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    radius = USER_RADIUS if args.radius is None else args.radius
    n_bits = USER_N_BITS if args.n_bits is None else args.n_bits
    outdir = USER_OUTDIR if args.outdir is None else args.outdir
    max_images_per_feature = USER_MAX_IMAGES_PER_FEATURE if args.max_images_per_feature is None else args.max_images_per_feature
    feature_labels = args.feature_labels if args.feature_labels is not None else list(USER_FEATURE_LABELS)

    if args.fill_onbits_csv:
        csv_path = args.pairs_csv or USER_PAIRS_CSV
        if not csv_path:
            parser.error("fill-onbits-csv 模式下请提供 --pairs-csv，或在脚本顶部填写 USER_PAIRS_CSV。")
        output_csv = args.output_csv or USER_OUTPUT_CSV
        outputs = fill_on_bits_in_csv(
            csv_path=csv_path,
            donor_col=args.donor_col,
            acceptor_col=args.acceptor_col,
            radius=radius,
            n_bits=n_bits,
            out_csv=output_csv,
        )
        print("Fill-onbits CSV mode finished.")
        for k, v in outputs.items():
            print(f"{k}: {v}")
        return

    if args.dataset_shap_mode:
        csv_path = args.pairs_csv or USER_PAIRS_CSV
        if not csv_path:
            parser.error("dataset-shap-mode 下请提供 --pairs-csv，或在脚本顶部填写 USER_PAIRS_CSV=\"DAreal.csv\"。")
        feature_labels = feature_labels if feature_labels else list(USER_SHAP_FEATURES)
        outputs = map_shap_features_in_dareal_dataset(
            csv_path=csv_path,
            feature_labels=feature_labels,
            donor_col=args.donor_col,
            acceptor_col=args.acceptor_col,
            sample_id_col=args.sample_id_col,
            radius=radius,
            n_bits=n_bits,
            outdir=outdir,
            save_images=args.save_images or USER_SAVE_IMAGES,
            max_images_per_feature=max_images_per_feature,
        )
        print("Dataset SHAP-feature mode finished.")
        for k, v in outputs.items():
            print(f"{k}: {v}")
        return

    donor_smiles = args.donor_smiles or USER_DONOR_SMILES
    acceptor_smiles = args.acceptor_smiles or USER_ACCEPTOR_SMILES

    if donor_smiles and acceptor_smiles:
        if not feature_labels:
            paired = build_paired_fingerprint_with_bitinfo(
                donor_smiles=donor_smiles,
                acceptor_smiles=acceptor_smiles,
                radius=radius,
                n_bits=n_bits,
            )
            feature_labels = [f"fd_{i}" for i in paired["donor_on_bits"]] + [f"fa_{i}" for i in paired["acceptor_on_bits"]]
            print(f"未指定 feature labels，已自动使用该对 D/A 的全部 on-bits，共 {len(feature_labels)} 个。")

        # 单对模式保留给你做示意图或快速检查
        outputs = map_shap_features_in_dareal_dataset(
            csv_path=USER_PAIRS_CSV,
            feature_labels=feature_labels,
            donor_col=args.donor_col,
            acceptor_col=args.acceptor_col,
            sample_id_col=args.sample_id_col,
            radius=radius,
            n_bits=n_bits,
            outdir=outdir,
            save_images=args.save_images or USER_SAVE_IMAGES,
            max_images_per_feature=max_images_per_feature,
        ) if False else None
        print("Single pair mode is kept only for manual checking in this script version. Please use --dataset-shap-mode for Table S17/Figure S10 reconstruction.")
        return

    parser.error(
        "请提供 --pairs-csv 并使用 --dataset-shap-mode，或提供 --pairs-csv 并使用 --fill-onbits-csv。"
    )


if __name__ == "__main__":
    main()

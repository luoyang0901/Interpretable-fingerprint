#!/usr/bin/env python3
"""Generate manuscript-style best-single and ensemble parity plots.

The script reads the outputs produced by 02_summarize_results.py and exports:
- one two-panel figure per TextCNN branch;
- one combined 3 x 2 overview figure;
- PNG (600 dpi), TIFF (600 dpi, LZW), vector PDF and EPS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BRANCH_LABELS = {
    "gao_only": "Gao-only",
    "inhouse_r2": "In-house radius-2",
    "joint_inhouse_gao": "Joint in-house + Gao",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--branch-order", nargs="*", default=["gao_only", "inhouse_r2", "joint_inhouse_gao"])
    p.add_argument("--point-size", type=float, default=46.0)
    p.add_argument("--background", default="#e9e9e9")
    p.add_argument("--point-color", default="#5b99c6")
    p.add_argument("--line-color", default="#6d6d6d")
    return p.parse_args()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    return {
        "r": float(pearsonr(y_true, y_pred)[0]),
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def configure_style(background: str) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": background,
            "axes.facecolor": background,
            "savefig.facecolor": background,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def common_limits(*arrays: np.ndarray) -> tuple[float, float]:
    values = np.concatenate([np.asarray(a, dtype=float).reshape(-1) for a in arrays])
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    span = max(hi - lo, 1.0)
    pad = 0.05 * span
    return lo - pad, hi + pad


def format_panel(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    panel_label: str,
    limits: tuple[float, float],
    args: argparse.Namespace,
) -> dict[str, float]:
    m = metrics(y_true, y_pred)
    ax.scatter(
        y_true,
        y_pred,
        s=args.point_size,
        c=args.point_color,
        edgecolors="white",
        linewidths=0.55,
        alpha=1.0,
        zorder=3,
    )
    ax.plot(limits, limits, linestyle="--", color=args.line_color, linewidth=1.35, zorder=2)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel("Experimental PCE (%)")
    ax.set_ylabel("Predicted PCE (%)")
    ax.set_title(
        f"{title}\n"
        f"r = {m['r']:.4f}, RMSE = {m['rmse']:.4f}, R$^2$ = {m['r2']:.4f}",
        pad=8,
    )
    ax.text(-0.12, 1.02, panel_label, transform=ax.transAxes, fontsize=11, va="bottom")
    ax.grid(True, color="#b8b8b8", alpha=0.45, linewidth=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#8d8d8d")
    ax.spines["bottom"].set_color("#8d8d8d")
    return m


def save_all_formats(fig: plt.Figure, output_stem: Path) -> None:
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    eps = output_stem.with_suffix(".eps")
    tiff = output_stem.with_suffix(".tiff")
    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(eps, bbox_inches="tight")
    with Image.open(png) as im:
        im.convert("RGB").save(tiff, format="TIFF", dpi=(600, 600), compression="tiff_lzw")


def branch_label(branch: str) -> str:
    return BRANCH_LABELS.get(branch, branch.replace("_", " ").title())


def main() -> None:
    args = parse_args()
    summary_dir = Path(args.summary_dir).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    configure_style(args.background)

    available = [branch for branch in args.branch_order if (summary_dir / branch).exists()]
    if not available:
        raise RuntimeError("No summarized branch folders were found.")

    plot_source_rows: list[pd.DataFrame] = []
    branch_data: dict[str, dict[str, Any]] = {}

    for branch in available:
        branch_dir = summary_dir / branch
        selection = json.loads((branch_dir / "selected_models_and_metrics.json").read_text(encoding="utf-8"))
        best = pd.read_csv(branch_dir / "best_single_predictions.csv")
        ensemble = pd.read_csv(branch_dir / "ensemble_101_predictions.csv")
        label = branch_label(branch)
        limits = common_limits(best["y_true"], best["y_pred"], ensemble["y_pred"])
        branch_data[branch] = {"best": best, "ensemble": ensemble, "label": label, "selection": selection, "limits": limits}

        fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.25))
        format_panel(
            axes[0],
            best["y_true"].to_numpy(),
            best["y_pred"].to_numpy(),
            f"{label}\nBest single model (seed {selection['best_single_seed']})",
            "(a)",
            limits,
            args,
        )
        format_panel(
            axes[1],
            ensemble["y_true"].to_numpy(),
            ensemble["y_pred"].to_numpy(),
            f"{label}\n101-seed ensemble prediction",
            "(b)",
            limits,
            args,
        )
        fig.tight_layout(w_pad=2.4)
        save_all_formats(fig, out / f"{branch}_best_single_and_ensemble")
        plt.close(fig)

        best_source = best.copy()
        best_source["branch"] = branch
        best_source["setting"] = "best_single"
        best_source["selected_seed"] = selection["best_single_seed"]
        ens_source = ensemble.copy()
        ens_source["branch"] = branch
        ens_source["setting"] = "ensemble_101"
        ens_source["selected_seed"] = selection["best_single_seed"]
        plot_source_rows.extend([best_source, ens_source])

    n_rows = len(available)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12.0, 5.1 * n_rows), squeeze=False)
    panel_index = 0
    for row, branch in enumerate(available):
        data = branch_data[branch]
        best = data["best"]
        ensemble = data["ensemble"]
        selection = data["selection"]
        label = data["label"]
        limits = data["limits"]
        format_panel(
            axes[row, 0],
            best["y_true"].to_numpy(),
            best["y_pred"].to_numpy(),
            f"{label}\nBest single model (seed {selection['best_single_seed']})",
            f"({chr(97 + panel_index)})",
            limits,
            args,
        )
        panel_index += 1
        format_panel(
            axes[row, 1],
            ensemble["y_true"].to_numpy(),
            ensemble["y_pred"].to_numpy(),
            f"{label}\n101-seed ensemble prediction",
            f"({chr(97 + panel_index)})",
            limits,
            args,
        )
        panel_index += 1
    fig.tight_layout(h_pad=2.5, w_pad=2.4)
    save_all_formats(fig, out / "Figure_TextCNN_three_branches_3x2")
    plt.close(fig)

    pd.concat(plot_source_rows, ignore_index=True).to_csv(
        out / "Figure_TextCNN_three_branches_source_data.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(f"Submission figures written to: {out}")


if __name__ == "__main__":
    main()

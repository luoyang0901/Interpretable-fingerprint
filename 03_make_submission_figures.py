#!/usr/bin/env python3
"""Generate parity, residual, calibration, and PCE-stratified error figures.

The script reads the machine-readable outputs of 02_summarize_results.py.
It creates one four-panel diagnostic figure per TextCNN branch and archives the
source data used for every panel.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common_utils import calibration_parameters, regression_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--branch-order", nargs="*", default=["gao_only", "inhouse_r2", "joint_inhouse_gao"])
    p.add_argument("--point-size", type=float, default=34.0)
    return p.parse_args()


def branch_label(name: str) -> str:
    mapping = {
        "gao_only": "Gao-only",
        "inhouse_r2": "In-house radius-2",
        "joint_inhouse_gao": "Joint in-house + Gao",
    }
    return mapping.get(name, name.replace("_", " ").title())


def save_formats(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".eps"), bbox_inches="tight")


def main() -> None:
    args = parse_args()
    summary = Path(args.summary_dir).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    available = [b for b in args.branch_order if (summary / b / "ensemble_predictions.csv").exists()]
    if not available:
        raise RuntimeError("No branch summary folders with ensemble_predictions.csv were found")
    all_sources = []
    for branch in available:
        branch_dir = summary / branch
        pred = pd.read_csv(branch_dir / "ensemble_predictions.csv")
        bins = pd.read_csv(branch_dir / "ensemble_pce_stratified_errors.csv")
        metrics = regression_metrics(pred["y_true"].to_numpy(), pred["y_pred"].to_numpy())
        intercept, slope = calibration_parameters(pred["y_true"].to_numpy(), pred["y_pred"].to_numpy())
        label = branch_label(branch)
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))

        ax = axes[0, 0]
        ax.scatter(pred["y_true"], pred["y_pred"], s=args.point_size, alpha=0.75, edgecolors="white", linewidths=0.4)
        lo = float(min(pred["y_true"].min(), pred["y_pred"].min()))
        hi = float(max(pred["y_true"].max(), pred["y_pred"].max()))
        pad = 0.05 * max(hi - lo, 1.0)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("Experimental PCE (%)")
        ax.set_ylabel("Predicted PCE (%)")
        ax.set_title(
            f"(a) {label}: parity\n"
            f"r={metrics['r']:.4f}, R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}"
        )

        ax = axes[0, 1]
        ax.scatter(pred["y_true"], pred["residual_pred_minus_true"], s=args.point_size, alpha=0.75, edgecolors="white", linewidths=0.4)
        ax.axhline(0.0, linestyle="--")
        ax.set_xlabel("Experimental PCE (%)")
        ax.set_ylabel("Residual: predicted − experimental (%)")
        ax.set_title(f"(b) Residual pattern\nMean signed error={metrics['mean_signed_error']:.4f}")

        ax = axes[1, 0]
        ax.scatter(pred["y_pred"], pred["y_true"], s=args.point_size, alpha=0.75, edgecolors="white", linewidths=0.4)
        xline = np.linspace(float(pred["y_pred"].min()), float(pred["y_pred"].max()), 100)
        ax.plot(xline, xline, linestyle="--", label="Ideal")
        ax.plot(xline, intercept + slope * xline, label="Observed calibration")
        ax.set_xlabel("Predicted PCE (%)")
        ax.set_ylabel("Experimental PCE (%)")
        ax.set_title(f"(c) Calibration\nintercept={intercept:.4f}, slope={slope:.4f}")
        ax.legend(frameon=False)

        ax = axes[1, 1]
        x = np.arange(len(bins))
        ax.bar(x, bins["mae"].to_numpy())
        ax.set_xticks(x)
        ax.set_xticklabels(bins["pce_bin"].astype(str), rotation=35, ha="right")
        ax.set_ylabel("MAE (%)")
        ax.set_xlabel("Experimental-PCE quantile bin")
        ax.set_title("(d) Error stratified by PCE range")

        for ax in axes.ravel():
            ax.grid(True, alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        save_formats(fig, out / f"{branch}_diagnostic_four_panel")
        plt.close(fig)
        source = pred.copy()
        source["branch"] = branch
        all_sources.append(source)
    pd.concat(all_sources, ignore_index=True).to_csv(
        out / "TextCNN_diagnostic_figure_source_data.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Figures written to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plot_le_b2_fa4_perclass.py

For the CP B2 binary (no-ad vs ad) classifier, plot recall and precision
broken down by the 5 annotation groups vs. number of training labels.
Data read directly from le_b2_fa4_perclass_cp_ds1.csv — no retraining.

Outputs:
  results/le_b2_fa4_recall.png
  results/le_b2_fa4_precision.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_DIR = Path("/net/projects/CLS/lding/data/fa_data_analysis/ae_results/features/eval_results")
OUT_DIR  = Path(__file__).resolve().parents[1] / "results"

CSV = EVAL_DIR / "le_b2_fa4_perclass_cp_ds1.csv"

CLASSES = [
    "No adhesion",
    "focal complex",
    "Nascent Adhesion",
    "focal adhesion",
    "fibrillar adhesion",
]
COLORS = {
    "No adhesion":        "#8B00FF",  # purple
    "Nascent Adhesion":   "#1E64F0",  # blue
    "focal complex":      "#F08200",  # orange
    "focal adhesion":     "#1EB43C",  # green
    "fibrillar adhesion": "#DC1E1E",  # red
}


def load_and_agg() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df = df[df["budget"] != "budget"]
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    rows = []
    for budget, grp in df.groupby("budget"):
        row = {"budget": budget, "n_train_mean": grp["n_train"].mean()}
        for cls in CLASSES:
            acc_col  = f"acc_{cls}"
            prec_col = f"prec_{cls}"
            row[f"recall_{cls}_mean"] = grp[acc_col].mean()
            row[f"recall_{cls}_std"]  = grp[acc_col].std()
            row[f"prec_{cls}_mean"]   = grp[prec_col].mean()
            row[f"prec_{cls}_std"]    = grp[prec_col].std()
        rows.append(row)

    return pd.DataFrame(rows).sort_values("n_train_mean")


def _plot(agg: pd.DataFrame, metric: str, ylabel: str, title: str, out: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = agg["n_train_mean"].values

    for cls in CLASSES:
        y   = agg[f"{metric}_{cls}_mean"].values
        err = agg[f"{metric}_{cls}_std"].values
        c   = COLORS[cls]
        ax.plot(x, y, color=c, linewidth=2, label=cls, marker="o", markersize=4)
        ax.fill_between(x, y - err, y + err, color=c, alpha=0.12)

    for th in [0.85, 0.90, 0.95]:
        ax.axhline(th, color="gray", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.text(x[0] * 1.05, th + 0.006, f"{int(th*100)}%", fontsize=8,
                color="gray", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("N training labels", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ymin = 0.0
    ax.set_ylim(ymin, 1.05)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}", flush=True)


def main():
    print("Loading FA4 perclass data ...", flush=True)
    agg = load_and_agg()

    _plot(agg, "recall",
          "Recall (class-wise)",
          "CP B2 — per-class recall (no-ad vs ad classifier)",
          OUT_DIR / "le_b2_fa4_recall.png")

    _plot(agg, "prec",
          "Precision (class-wise)",
          "CP B2 — per-class precision (no-ad vs ad classifier)",
          OUT_DIR / "le_b2_fa4_precision.png")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

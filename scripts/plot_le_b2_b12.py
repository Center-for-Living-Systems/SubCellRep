#!/usr/bin/env python3
"""
plot_le_b2_b12.py

LE curves (bal_acc + recall_ad + recall_noad + prec_ad) for
CP, Ilastik, and SupCon-AE (z16/proj8) on B2 and B12 sets.

Outputs:
  results/le_b2_cp_il_supcon.png
  results/le_b12_cp_il_supcon.png
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EVAL_DIR  = Path("/net/projects/CLS/lding/data/fa_data_analysis/ae_results/features/eval_results")
REPO_EVAL = Path(__file__).resolve().parents[1] / "eval_results" / "b1b2v2"
OUT_DIR   = Path(__file__).resolve().parents[1] / "results"

COLORS = {"CP": "#2979FF", "Ilastik": "#FF6D00", "SupCon": "#00897B"}
LABELS = {"CP": "CP", "Ilastik": "Ilastik", "SupCon": "SupCon-AE (z16/proj8)"}
METRICS = [
    ("bal_acc",    "Balanced accuracy"),
    ("recall_ad",  "Recall — adhesion"),
    ("recall_noad","Recall — no adhesion"),
    ("prec_ad",    "Precision — adhesion"),
    ("prec_noad",  "Precision — no adhesion"),
]
THRESH = [0.85, 0.90, 0.95]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _derive_b2_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """For B2/B12-SupCon CSVs: derive overall recall/prec from per-subgroup cols."""
    df = df.copy()
    df["recall_ad"]  = (df["acc_ctrl_ad"]  + df["acc_ycomp_ad"])  / 2
    df["recall_noad"]= (df["acc_ctrl_noad"] + df["acc_ycomp_noad"])/ 2
    df["prec_ad"]    = df["prec_ctrl_ad"]  + df["prec_ycomp_ad"]
    df["prec_noad"]  = df["prec_ctrl_noad"] + df["prec_ycomp_noad"]
    return df


def _agg(df: pd.DataFrame, group_col: str = "budget") -> pd.DataFrame:
    """Mean ± std per budget."""
    metrics = ["bal_acc", "recall_ad", "recall_noad", "prec_ad", "prec_noad",
               "n_train"]
    out = df.groupby(group_col)[metrics].agg(["mean", "std"]).reset_index()
    out.columns = [group_col] + [f"{m}_{s}" for m, s in out.columns[1:]]
    return out.sort_values("n_train_mean")


# ── Load B2 data ───────────────────────────────────────────────────────────────

def load_b2() -> dict[str, pd.DataFrame]:
    result = {}
    for feat, fname in [
        ("CP",      "le_b2_ctrlvy_cp_ds1.csv"),
        ("Ilastik", "le_b2_ctrlvy_il_ds1.csv"),
        ("SupCon",  "le_b2_ctrlvy_supcon_ds1.csv"),
    ]:
        df = pd.read_csv(EVAL_DIR / fname)
        df = df[df["budget"] != "budget"]          # drop duplicate header rows
        df["n_train"] = pd.to_numeric(df["n_train"])
        df["bal_acc"] = pd.to_numeric(df["bal_acc"])
        for c in ["acc_ctrl_ad","prec_ctrl_ad","acc_ctrl_noad","prec_ctrl_noad",
                  "acc_ycomp_ad","prec_ycomp_ad","acc_ycomp_noad","prec_ycomp_noad"]:
            df[c] = pd.to_numeric(df[c])
        df = _derive_b2_metrics(df)
        result[feat] = _agg(df)
    return result


# ── Load B12 data ──────────────────────────────────────────────────────────────

def load_b12() -> dict[str, pd.DataFrame]:
    result = {}

    # CP, IL, SupCon — all from B12 union 5-fold CV files (same format as B2)
    for feat, fname in [
        ("CP",      "le_b12_ctrlvy_cp_ds1.csv"),
        ("Ilastik", "le_b12_ctrlvy_il_ds1.csv"),
        ("SupCon",  "le_b12_ctrlvy_supcon_ds1.csv"),
    ]:
        df = pd.read_csv(EVAL_DIR / fname)
        df = df[df["budget"] != "budget"]
        df["n_train"] = pd.to_numeric(df["n_train"])
        df["bal_acc"] = pd.to_numeric(df["bal_acc"])
        for c in ["acc_ctrl_ad","prec_ctrl_ad","acc_ctrl_noad","prec_ctrl_noad",
                  "acc_ycomp_ad","prec_ycomp_ad","acc_ycomp_noad","prec_ycomp_noad"]:
            df[c] = pd.to_numeric(df[c])
        df = _derive_b2_metrics(df)
        result[feat] = _agg(df)

    return result


# ── Plot one dataset ───────────────────────────────────────────────────────────

def plot_dataset(data: dict[str, pd.DataFrame], title: str, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    axes[1, 2].set_visible(False)   # 6th cell unused

    for ax, (metric, ylabel) in zip(axes.flat, METRICS):
        for feat, agg in data.items():
            x   = agg["n_train_mean"].values
            y   = agg[f"{metric}_mean"].values
            err = agg[f"{metric}_std"].values
            c   = COLORS[feat]
            ax.plot(x, y, color=c, linewidth=2, label=LABELS[feat], marker="o",
                    markersize=4)
            ax.fill_between(x, y - err, y + err, color=c, alpha=0.15)

        for th in THRESH:
            ax.axhline(th, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
            ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 1,
                    th + 0.003, f"{int(th*100)}%", fontsize=8, color="gray",
                    va="bottom")

        ymin = 0.0 if metric.startswith("prec") else 0.5
        ax.set_xlabel("N training labels", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xscale("log")
        ax.set_ylim(ymin, 1.02)
        ax.set_yticks(np.arange(ymin, 1.01, 0.1 if ymin == 0.0 else 0.05))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out_path}", flush=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading B2 data ...", flush=True)
    b2 = load_b2()
    print("Loading B12 data ...", flush=True)
    b12 = load_b12()

    plot_dataset(b2,  "Label efficiency — B2 (Annabel, DS1)",
                 OUT_DIR / "le_b2_cp_il_supcon.png")
    plot_dataset(b12, "Label efficiency — B12 (B1+B2 union, DS1, 5-fold CV)",
                 OUT_DIR / "le_b12_cp_il_supcon.png")

    print("Done.", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_pptx_clf_comparison.py

LGBM vs LogReg comparison deck across all b1b2v2 and image-efficiency evals.

Slides
------
  1  Title
  2  Cross-eval: LGBM vs LogReg (grouped bars, both features)
  3  Union eval: LGBM vs LogReg (grouped bars, both features)
  4  Budget curves: LGBM (solid) vs LogReg (dashed), all strategies
  5  Image efficiency b1b2v2: image-budget "all-labels" curves
  6  Image efficiency b1b2v2: LOIO violin
  7  Image efficiency b1_full: image-budget "all-labels" curves
  8  Image efficiency b1_full: LOIO violin
  9  Summary table

Usage:
  python scripts/make_pptx_clf_comparison.py [--out PATH]
"""
from __future__ import annotations
import argparse, io, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pptx_le_benchmark import (
    _blank, _txt, _slide_header, _panel_bg, _rule,
    _setup_col_header, _setup_kv,
    C_HEAD, C_BODY, C_GREY, C_WHITE, C_PANEL,
    SW, SH, PP_ALIGN, RGBColor, Inches, Pt,
)
from pptx import Presentation

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO    = Path(__file__).resolve().parents[1]
LE_DIR  = REPO / "eval_results" / "b1b2v2"
IE_DIR  = REPO / "eval_results"
OUT_DIR = REPO / "results"

# LE paths
_CROSS_CP_LGBM  = LE_DIR / "le_b1b2v2_cross_cp_lgbm.csv"
_CROSS_CP_LR    = LE_DIR / "le_b1b2v2_cross_cp_logreg.csv"
_CROSS_IL_LGBM  = LE_DIR / "le_b1b2v2_cross_ilastik_lgbm.csv"
_CROSS_IL_LR    = LE_DIR / "le_b1b2v2_cross_ilastik_logreg.csv"
_UNION_CP_LGBM  = LE_DIR / "le_b1b2v2_union_cp_lgbm.csv"
_UNION_CP_LR    = LE_DIR / "le_b1b2v2_union_cp_logreg.csv"
_UNION_IL_LGBM  = LE_DIR / "le_b1b2v2_union_ilastik_lgbm.csv"
_UNION_IL_LR    = LE_DIR / "le_b1b2v2_union_ilastik_logreg.csv"
_BUDGET_CP_LGBM = LE_DIR / "le_b1b2v2_budget_cp_lgbm.csv"
_BUDGET_CP_LR   = LE_DIR / "le_b1b2v2_budget_cp_logreg.csv"
_BUDGET_IL_LGBM = LE_DIR / "le_b1b2v2_budget_ilastik_lgbm.csv"
_BUDGET_IL_LR   = LE_DIR / "le_b1b2v2_budget_ilastik_logreg.csv"

# Image efficiency paths
_IE_B1B2_LGBM   = IE_DIR / "image_efficiency_b1b2v2_lgbm.csv"
_IE_B1B2_LR     = IE_DIR / "image_efficiency_b1b2v2_logreg.csv"
_IE_B1F_LGBM    = IE_DIR / "image_efficiency_b1_full_lgbm.csv"
_IE_B1F_LR      = IE_DIR / "image_efficiency_b1_full_logreg.csv"

# ── Visual constants ───────────────────────────────────────────────────────────
CLF_COLOR = {"lgbm": "#1f77b4", "logreg": "#e07b39"}
CLF_LABEL = {"lgbm": "LightGBM", "logreg": "LogReg"}
CLF_LS    = {"lgbm": "-",        "logreg": "--"}
CLF_LW    = {"lgbm": 2.2,        "logreg": 2.0}

STRAT_COLORS = {"b1_only": "#e07b39", "b2v2_only": "#2e86ab", "mixed": "#2ca02c"}
STRAT_LABELS = {"b1_only": "B1-only", "b2v2_only": "B2v2-only", "mixed": "Mixed"}

COMBOS = [("b1","b1"), ("b1","b2v2"), ("b2v2","b1"), ("b2v2","b2v2")]
COMBO_LABELS = {
    ("b1","b1"):     "B1→B1",
    ("b1","b2v2"):   "B1→B2v2",
    ("b2v2","b1"):   "B2v2→B1",
    ("b2v2","b2v2"): "B2v2→B2v2",
}
COMBO_COLORS = ["#1f77b4","#aec7e8","#d62728","#ff9896"]

METRICS = ["bal_acc","recall_ad","recall_noad","prec_ad","prec_noad"]
MLABELS = ["Bal. Acc","Recall\n(ad)","Recall\n(no-ad)","Prec\n(ad)","Prec\n(no-ad)"]

COL_CTRL  = "#e85454"
COL_YCOMP = "#2ca02c"

PLOT_LEFT = Inches(0.35)
PLOT_TOP  = Inches(1.05)
PLOT_W    = Inches(12.6)
PLOT_H    = Inches(6.15)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _prs():
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH
    return prs


def _fig_to_buf(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    return buf


def _add_fig(slide, fig, left, top, width, height):
    buf = _fig_to_buf(fig)
    slide.shapes.add_picture(buf, left, top, width=width, height=height)


def _safe_csv(*paths):
    """Load and concat CSVs that exist."""
    dfs = []
    for p in paths:
        if Path(p).exists():
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# ── Slide 1: Title ─────────────────────────────────────────────────────────────

def _slide_title(prs):
    slide = _blank(prs)
    _txt(slide, "LightGBM vs Logistic Regression — Classifier Comparison",
         Inches(1.0), Inches(2.2), Inches(11.3), Inches(1.1),
         bold=True, size_pt=34, color=C_HEAD, align=PP_ALIGN.CENTER)
    _txt(slide, "B1/B2v2 matched annotators  ·  DS1 vinculin CIO imaging  ·  CP + ilastik features",
         Inches(1.0), Inches(3.4), Inches(11.3), Inches(0.45),
         size_pt=16, color=C_GREY, align=PP_ALIGN.CENTER)
    _rule(slide, Inches(3.7), width=Inches(9), left=Inches(2.17))
    _txt(slide, "Label efficiency: cross-eval · union · budget study",
         Inches(1.0), Inches(4.0), Inches(11.3), Inches(0.4),
         size_pt=13, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide, "Image efficiency: b1b2v2 matched (22 images) · b1_full (89 images)",
         Inches(1.0), Inches(4.45), Inches(11.3), Inches(0.4),
         size_pt=13, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide, "LightGBM  ─────  LogReg  - - - -",
         Inches(1.0), Inches(5.0), Inches(11.3), Inches(0.35),
         size_pt=12, color=RGBColor(0x33, 0x33, 0x99), align=PP_ALIGN.CENTER)


# ── Slide 2: Cross eval LGBM vs LogReg ────────────────────────────────────────

def _slide_cross(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "Cross-Annotator Eval — LGBM vs LogReg",
        "B1→B1, B1→B2v2, B2v2→B1, B2v2→B2v2  ·  5-fold mean ± std  ·  nball budget")

    feat_specs = [
        ("CellProfiler", _CROSS_CP_LGBM, _CROSS_CP_LR),
        ("ilastik",      _CROSS_IL_LGBM, _CROSS_IL_LR),
    ]
    n_met = len(METRICS)
    fig, axes = plt.subplots(2, n_met, figsize=(13.5, 7.5), facecolor="white")

    x = np.arange(len(COMBOS))
    w = 0.35  # bar width; two bars per combo (lgbm / logreg)

    for row, (feat_name, lgbm_csv, lr_csv) in enumerate(feat_specs):
        df_lgbm = pd.read_csv(lgbm_csv) if lgbm_csv.exists() else pd.DataFrame()
        df_lr   = pd.read_csv(lr_csv)   if lr_csv.exists()   else pd.DataFrame()

        for col, (metric, mlabel) in enumerate(zip(METRICS, MLABELS)):
            ax = axes[row, col]

            for ci, (clf, df) in enumerate([("lgbm", df_lgbm), ("logreg", df_lr)]):
                if df.empty:
                    continue
                offset = (ci - 0.5) * w
                vals = [df[(df.train_batch==tr)&(df.test_batch==te)][metric].mean()
                        for tr, te in COMBOS]
                stds = [df[(df.train_batch==tr)&(df.test_batch==te)][metric].std()
                        for tr, te in COMBOS]
                bars = ax.bar(x + offset, vals, w,
                              color=[CLF_COLOR[clf]]*len(COMBOS), alpha=0.80,
                              label=CLF_LABEL[clf], hatch="" if clf=="lgbm" else "///",
                              edgecolor="white", zorder=2)
                ax.errorbar(x + offset, vals, stds, fmt="none",
                            color="#333", capsize=2.5, lw=1.0, zorder=3)
                for bar, val in zip(bars, vals):
                    if not np.isnan(val):
                        ax.text(bar.get_x()+bar.get_width()/2, val+0.005,
                                f"{val:.3f}", ha="center", va="bottom", fontsize=5.5)

            ax.set_xticks(x)
            ax.set_xticklabels([COMBO_LABELS[c].replace("→","\n→\n") for c in COMBOS],
                               fontsize=7)
            ax.set_ylim(0.45, 1.05)
            ax.set_ylabel("Score" if col == 0 else "", fontsize=8)
            if row == 0:
                ax.set_title(mlabel, fontsize=9, fontweight="bold", pad=4)
                if col == n_met - 1:
                    ax.legend(fontsize=7, framealpha=0.9, loc="upper right")
            if col == 0:
                ax.text(-0.25, 0.5, feat_name, transform=ax.transAxes,
                        fontsize=10, fontweight="bold", rotation=90,
                        va="center", ha="right", color="#16213e")
            ax.axhline(0.5, color="#aaa", lw=0.7, ls="--", zorder=0)
            ax.set_facecolor("white")
            ax.grid(axis="y", color="#eee", lw=0.7)
            ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    _add_fig(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, PLOT_H)


# ── Slide 3: Union eval LGBM vs LogReg ────────────────────────────────────────

def _slide_union(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "Union Training Eval — LGBM vs LogReg",
        "Train on B1∪B2v2 (conflict-excluded, ~628/fold) → test on B1 or B2v2  ·  5-fold mean ± std")

    feat_specs = [
        ("CellProfiler", _UNION_CP_LGBM, _UNION_CP_LR),
        ("ilastik",      _UNION_IL_LGBM, _UNION_IL_LR),
    ]
    test_batches = ["b1", "b2v2"]
    tb_labels    = {"b1": "Test: B1", "b2v2": "Test: B2v2"}
    n_met = len(METRICS)
    fig, axes = plt.subplots(2, n_met, figsize=(13.5, 7.5), facecolor="white")

    for row, (feat_name, lgbm_csv, lr_csv) in enumerate(feat_specs):
        df_lgbm = pd.read_csv(lgbm_csv) if lgbm_csv.exists() else pd.DataFrame()
        df_lr   = pd.read_csv(lr_csv)   if lr_csv.exists()   else pd.DataFrame()

        for col, (metric, mlabel) in enumerate(zip(METRICS, MLABELS)):
            ax = axes[row, col]
            x = np.arange(len(test_batches))
            w = 0.30

            for ci, (clf, df) in enumerate([("lgbm", df_lgbm), ("logreg", df_lr)]):
                if df.empty:
                    continue
                offset = (ci - 0.5) * w
                vals = [df[df.test_batch==tb][metric].mean() for tb in test_batches]
                stds = [df[df.test_batch==tb][metric].std()  for tb in test_batches]
                bars = ax.bar(x + offset, vals, w,
                              color=[CLF_COLOR[clf]]*len(test_batches), alpha=0.80,
                              label=CLF_LABEL[clf], hatch="" if clf=="lgbm" else "///",
                              edgecolor="white", zorder=2)
                ax.errorbar(x + offset, vals, stds, fmt="none",
                            color="#333", capsize=2.5, lw=1.0, zorder=3)
                for bar, val in zip(bars, vals):
                    if not np.isnan(val):
                        ax.text(bar.get_x()+bar.get_width()/2, val+0.005,
                                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels([tb_labels[tb] for tb in test_batches], fontsize=9)
            ax.set_ylim(0.45, 1.05)
            ax.set_ylabel("Score" if col == 0 else "", fontsize=8)
            if row == 0:
                ax.set_title(mlabel, fontsize=9, fontweight="bold", pad=4)
                if col == n_met - 1:
                    ax.legend(fontsize=8, framealpha=0.9)
            if col == 0:
                ax.text(-0.3, 0.5, feat_name, transform=ax.transAxes,
                        fontsize=10, fontweight="bold", rotation=90,
                        va="center", ha="right", color="#16213e")
            ax.axhline(0.5, color="#aaa", lw=0.7, ls="--", zorder=0)
            ax.set_facecolor("white")
            ax.grid(axis="y", color="#eee", lw=0.7)
            ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    _add_fig(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, PLOT_H)


# ── Slide 4: Budget curves LGBM vs LogReg ─────────────────────────────────────

def _slide_budget(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "Budget Study — LGBM (─) vs LogReg (- -)",
        "B1-only / B2v2-only / mixed  ·  CP & ilastik  ·  test on B1 (left) and B2v2 (right)  ·  mean (5 folds × 10 repeats)")

    panels = [
        ("CellProfiler → Test B1",   "cp",      "b1"),
        ("CellProfiler → Test B2v2", "cp",      "b2v2"),
        ("ilastik → Test B1",        "ilastik", "b1"),
        ("ilastik → Test B2v2",      "ilastik", "b2v2"),
    ]
    csv_map = {
        ("cp",      "lgbm"):   _BUDGET_CP_LGBM,
        ("cp",      "logreg"): _BUDGET_CP_LR,
        ("ilastik", "lgbm"):   _BUDGET_IL_LGBM,
        ("ilastik", "logreg"): _BUDGET_IL_LR,
    }
    SINGLE_MAX = 340

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 6.5), facecolor="white")
    axes_flat = axes.flatten()

    for ai, (title, feat, test_b) in enumerate(panels):
        ax = axes_flat[ai]
        ax.set_title(title, fontsize=10, fontweight="bold", pad=5)

        for clf in ["lgbm", "logreg"]:
            csv_path = csv_map[(feat, clf)]
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            df_te = df[df.test_batch == test_b]

            num_bs = sorted(b for b in df_te.budget.unique() if b != 0)
            step   = num_bs[-1] - num_bs[-2] if len(num_bs) > 1 else 20
            all_pos = num_bs[-1] + step

            for strat in ["b1_only", "b2v2_only", "mixed"]:
                sub = df_te[df_te.strategy == strat]
                if sub.empty:
                    continue
                g   = sub.groupby("budget")["bal_acc"]
                means = g.mean()
                num_sub = {b: v for b, v in means.items() if b != 0}
                all_sub = {b: v for b, v in means.items() if b == 0}

                c = STRAT_COLORS[strat]
                lbl = f"{STRAT_LABELS[strat]} ({CLF_LABEL[clf]})" if clf == "lgbm" else None

                if num_sub:
                    xs = sorted(num_sub)
                    ys = [num_sub[b] for b in xs]
                    ax.plot(xs, ys, color=c, lw=CLF_LW[clf], ls=CLF_LS[clf],
                            marker="o", markersize=3 if clf=="lgbm" else 2,
                            alpha=0.9, label=lbl, zorder=2)
                if all_sub:
                    av = list(all_sub.values())[0]
                    m = "D" if clf == "lgbm" else "s"
                    ax.scatter([all_pos], [av], marker=m, s=70 if clf=="lgbm" else 40,
                               color=c, zorder=5, alpha=0.9)

        ax.axvline(SINGLE_MAX + step/2, color="#bbb", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("Label budget (N)", fontsize=9)
        ax.set_ylabel("Balanced accuracy", fontsize=9)
        ax.set_ylim(0.42, 1.02)
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#EEEEEE", lw=0.7)
        ax.spines[["top","right"]].set_visible(False)
        if ai == 0:
            strat_patches = [
                mpatches.Patch(color=STRAT_COLORS[s], label=STRAT_LABELS[s])
                for s in ["b1_only","b2v2_only","mixed"]]
            clf_lines = [
                plt.Line2D([0],[0], color="#555", ls="-",  lw=2, label="LightGBM"),
                plt.Line2D([0],[0], color="#555", ls="--", lw=2, label="LogReg"),
            ]
            ax.legend(handles=strat_patches + clf_lines,
                      fontsize=7.5, framealpha=0.9, loc="lower right", ncol=2)

    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    _add_fig(slide, fig, Inches(0.2), Inches(1.0), Inches(13.0), Inches(6.2))


# ── Image efficiency helpers ───────────────────────────────────────────────────

def _load_ie(lgbm_csv, lr_csv) -> pd.DataFrame:
    dfs = []
    for clf, p in [("lgbm", lgbm_csv), ("logreg", lr_csv)]:
        if Path(p).exists():
            df = pd.read_csv(p)
            if "clf" not in df.columns:
                df["clf"] = clf
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _ie_budget_fig(df: pd.DataFrame, group_col: str, group_vals: list,
                   group_labels: dict, feat_col: str, title: str) -> plt.Figure:
    """
    2×2 grid: rows = groups (annotator or single), cols = features (cp, ilastik).
    Each panel: LGBM solid vs LogReg dashed, label_budget='all'.
    """
    feats = ["cp", "ilastik"]
    feat_lbl = {"cp": "CellProfiler", "ilastik": "ilastik"}
    n_rows = len(group_vals)

    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4.5 * n_rows), facecolor="white",
                             squeeze=False)
    fig.subplots_adjust(hspace=0.40, wspace=0.28,
                        left=0.07, right=0.95, top=0.92, bottom=0.08)

    for ri, grp in enumerate(group_vals):
        for ci, feat in enumerate(feats):
            ax = axes[ri, ci]
            for clf in ["lgbm", "logreg"]:
                sub = df[
                    (df[group_col]     == grp) &
                    (df[feat_col]      == feat) &
                    (df["clf"]         == clf) &
                    (df["experiment"]  == "image_budget") &
                    (df["label_budget"]== "all")
                ]
                if sub.empty:
                    continue
                g = sub.groupby("k_images")["bal_acc"].agg(["mean","std"])
                k_vals = sorted(g.index)
                means  = g.loc[k_vals, "mean"].values * 100
                stds   = g.loc[k_vals, "std"].values  * 100
                ax.plot(k_vals, means, color=CLF_COLOR[clf], lw=CLF_LW[clf],
                        ls=CLF_LS[clf], marker="o", markersize=4,
                        label=CLF_LABEL[clf], zorder=3)
                ax.fill_between(k_vals, means - stds, means + stds,
                                color=CLF_COLOR[clf], alpha=0.10, zorder=1)

            for thresh, ls in [(70,"--"),(75,":"),(80,"-."),(85,"--"),(90,":"),(95,"-.")]:
                ax.axhline(thresh, color="#aaa", ls=ls, lw=0.8, alpha=0.7)
                sub_k = df[df["experiment"]=="image_budget"]["k_images"].max() \
                    if not df.empty else 20
                ax.text(sub_k + 0.2, thresh + 0.3, f"{thresh}%",
                        color="#999", fontsize=7, va="bottom")

            ax.set_title(f"{group_labels.get(grp, grp)}  —  {feat_lbl[feat]}",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("Training images (k)", fontsize=9)
            ax.set_ylabel("Balanced accuracy (%)", fontsize=9)
            ax.set_ylim(40, 108)
            ax.set_facecolor("white")
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y", color="#EEEEEE", lw=0.7)
            if ri == 0 and ci == 1:
                ax.legend(fontsize=9, framealpha=0.9, loc="lower right")

    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.97)
    return fig


def _ie_loio_fig(df: pd.DataFrame, group_col: str, group_vals: list,
                 group_labels: dict, feat_col: str, title: str) -> plt.Figure:
    """
    2×2 grouped bar chart for LOIO (all-labels): rows=groups, cols=features.
    Each panel: LGBM vs LogReg side-by-side, colored by condition.
    """
    feats = ["cp", "ilastik"]
    feat_lbl = {"cp": "CellProfiler", "ilastik": "ilastik"}
    n_rows = len(group_vals)

    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4.5 * n_rows), facecolor="white",
                             squeeze=False)
    fig.subplots_adjust(hspace=0.42, wspace=0.30,
                        left=0.07, right=0.95, top=0.92, bottom=0.12)

    for ri, grp in enumerate(group_vals):
        for ci, feat in enumerate(feats):
            ax = axes[ri, ci]
            x_pos = [1, 2]  # control, ycomp
            w = 0.28

            for ki, clf in enumerate(["lgbm", "logreg"]):
                sub = df[
                    (df[group_col]     == grp) &
                    (df[feat_col]      == feat) &
                    (df["clf"]         == clf) &
                    (df["experiment"]  == "loio") &
                    (df["label_budget"]== "all")
                ].copy()
                if sub.empty:
                    continue
                if "held_out_image" in sub.columns:
                    sub["condition"] = sub["held_out_image"].str.split("_f").str[0]
                else:
                    sub["condition"] = sub.get("cond", "?")

                offset = (ki - 0.5) * w
                for xi, cond in zip(x_pos, ["control","ycomp"]):
                    vals = sub[sub["condition"] == cond]["bal_acc"].values * 100
                    if len(vals) == 0:
                        continue
                    c = COL_CTRL if cond == "control" else COL_YCOMP
                    alpha = 0.85 if clf == "lgbm" else 0.5
                    ax.bar(xi + offset, np.mean(vals), w,
                           color=c, alpha=alpha,
                           hatch="" if clf=="lgbm" else "///",
                           edgecolor="white", zorder=2)
                    ax.errorbar(xi + offset, np.mean(vals), np.std(vals),
                                fmt="none", color="#333", capsize=2.5, lw=1.0, zorder=3)

            for thresh, ls in [(70,"--"),(80,"-."),(90,":")]:
                ax.axhline(thresh, color="#aaa", ls=ls, lw=0.8, alpha=0.7)

            ax.set_title(f"{group_labels.get(grp, grp)}  —  {feat_lbl[feat]}",
                         fontsize=10, fontweight="bold")
            ax.set_ylabel("Balanced accuracy (%)", fontsize=9)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(["control","ycomp"], fontsize=9)
            ax.set_ylim(30, 108)
            ax.set_facecolor("white")
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="y", color="#EEEEEE", lw=0.7)

    # Legend
    handles = [
        mpatches.Patch(color=COL_CTRL,  label="control"),
        mpatches.Patch(color=COL_YCOMP, label="ycomp"),
        plt.Line2D([0],[0], color="#555", ls="-",  lw=2, label="LightGBM (solid)"),
        plt.Line2D([0],[0], color="#555", ls="--", lw=2, label="LogReg (hatched)"),
    ]
    fig.legend(handles=handles, fontsize=9, loc="lower center",
               ncol=4, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.97)
    return fig


# ── Slides 5–6: Image efficiency b1b2v2 ───────────────────────────────────────

def _slide_ie_b1b2v2_budget(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "Image Efficiency b1b2v2 — Budget Curves: LGBM (─) vs LogReg (- -)",
        "All-labels budget  ·  B1 (Margaret) + B2v2 (Annabel)  ·  k = 1–19 images  ·  20 repeats")
    df = _load_ie(_IE_B1B2_LGBM, _IE_B1B2_LR)
    if df.empty:
        _txt(slide, "No data", Inches(1), Inches(2), Inches(11), Inches(1),
             size_pt=14, color=C_GREY)
        return
    fig = _ie_budget_fig(df, "annotator", ["b1","b2v2"],
                         {"b1":"B1 (Margaret)","b2v2":"B2v2 (Annabel)"},
                         "feature",
                         "Image Efficiency — b1b2v2 annotators  (all labels, LGBM vs LogReg)")
    _add_fig(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, PLOT_H)


def _slide_ie_b1b2v2_loio(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "Image Efficiency b1b2v2 — LOIO: LGBM vs LogReg",
        "All-labels budget  ·  22 held-out images  ·  per-condition mean ± std")
    df = _load_ie(_IE_B1B2_LGBM, _IE_B1B2_LR)
    if df.empty:
        _txt(slide, "No data", Inches(1), Inches(2), Inches(11), Inches(1),
             size_pt=14, color=C_GREY)
        return
    fig = _ie_loio_fig(df, "annotator", ["b1","b2v2"],
                       {"b1":"B1 (Margaret)","b2v2":"B2v2 (Annabel)"},
                       "feature",
                       "LOIO — b1b2v2 annotators  (all labels, LGBM vs LogReg)")
    _add_fig(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, PLOT_H)


# ── Slides 7–8: Image efficiency b1_full ──────────────────────────────────────

def _slide_ie_b1full_budget(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "Image Efficiency b1_full — Budget Curves: LGBM (─) vs LogReg (- -)",
        "Margaret full label set  ·  89 frames  ·  all-labels budget  ·  k = 1–70 images  ·  20 repeats")
    df = _load_ie(_IE_B1F_LGBM, _IE_B1F_LR)
    if df.empty:
        _txt(slide, "No data", Inches(1), Inches(2), Inches(11), Inches(1),
             size_pt=14, color=C_GREY)
        return
    # b1_full has only one "annotator" — just use feature as grouping
    feats = sorted(df["feature"].unique()) if not df.empty else ["cp","ilastik"]
    feat_lbl = {"cp": "CellProfiler", "ilastik": "ilastik"}
    clfs = [c for c in ["lgbm","logreg"] if c in df["clf"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    fig.subplots_adjust(hspace=0.38, wspace=0.28, left=0.07, right=0.95, top=0.88, bottom=0.12)

    for ci, feat in enumerate(feats):
        ax = axes[ci]
        for clf in clfs:
            sub = df[
                (df["feature"]      == feat) &
                (df["clf"]          == clf) &
                (df["experiment"]   == "image_budget") &
                (df["label_budget"] == "all")
            ]
            if sub.empty:
                continue
            g = sub.groupby("k_images")["bal_acc"].agg(["mean","std"])
            k_vals = sorted(g.index)
            means  = g.loc[k_vals, "mean"].values * 100
            stds   = g.loc[k_vals, "std"].values  * 100
            ax.plot(k_vals, means, color=CLF_COLOR[clf], lw=CLF_LW[clf],
                    ls=CLF_LS[clf], marker="o", markersize=4,
                    label=CLF_LABEL[clf], zorder=3)
            ax.fill_between(k_vals, means - stds, means + stds,
                            color=CLF_COLOR[clf], alpha=0.10, zorder=1)

        for thresh, ls in [(70,"--"),(75,":"),(80,"-."),(85,"--"),(90,":"),(95,"-.")]:
            ax.axhline(thresh, color="#aaa", ls=ls, lw=0.8, alpha=0.7)

        ax.set_title(f"B1 full — {feat_lbl.get(feat, feat)}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Training images (k)", fontsize=10)
        ax.set_ylabel("Balanced accuracy (%)", fontsize=10)
        ax.set_ylim(40, 108)
        ax.set_facecolor("white")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", color="#EEEEEE", lw=0.7)
        ax.legend(fontsize=9, framealpha=0.9, loc="lower right")

    fig.suptitle("Image Efficiency — b1_full  (all labels, LGBM vs LogReg)",
                 fontsize=12, fontweight="bold", y=0.97)
    _add_fig(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, Inches(5.8))


def _slide_ie_b1full_loio(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "Image Efficiency b1_full — LOIO: LGBM vs LogReg",
        "Margaret full label set  ·  89 frames  ·  all-labels budget  ·  per-condition mean ± std")
    df = _load_ie(_IE_B1F_LGBM, _IE_B1F_LR)
    if df.empty:
        _txt(slide, "No data", Inches(1), Inches(2), Inches(11), Inches(1),
             size_pt=14, color=C_GREY)
        return

    feats = sorted(df["feature"].unique()) if not df.empty else ["cp","ilastik"]
    feat_lbl = {"cp": "CellProfiler", "ilastik": "ilastik"}
    clfs = [c for c in ["lgbm","logreg"] if c in df["clf"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), facecolor="white")
    fig.subplots_adjust(hspace=0.38, wspace=0.30, left=0.07, right=0.95, top=0.88, bottom=0.14)

    x_pos = [1, 2]
    w = 0.28
    for ci, feat in enumerate(feats):
        ax = axes[ci]
        for ki, clf in enumerate(clfs):
            sub = df[
                (df["feature"]      == feat) &
                (df["clf"]          == clf) &
                (df["experiment"]   == "loio") &
                (df["label_budget"] == "all")
            ].copy()
            if sub.empty:
                continue
            if "held_out_image" in sub.columns:
                # handles both "control_f01" and "control-f01" formats
                sub["condition"] = sub["held_out_image"].str.extract(r'^([^_-]+)', expand=False)
            elif "image_key" in sub.columns:
                sub["condition"] = sub["image_key"].str.extract(r'^([^_-]+)', expand=False)

            offset = (ki - 0.5) * w
            for xi, cond in zip(x_pos, ["control","ycomp"]):
                vals = sub[sub["condition"] == cond]["bal_acc"].dropna().values * 100
                if len(vals) == 0:
                    continue
                c = COL_CTRL if cond == "control" else COL_YCOMP
                ax.bar(xi + offset, np.mean(vals), w,
                       color=c, alpha=0.85 if clf=="lgbm" else 0.50,
                       hatch="" if clf=="lgbm" else "///", edgecolor="white", zorder=2)
                ax.errorbar(xi + offset, np.mean(vals), np.std(vals),
                            fmt="none", color="#333", capsize=2.5, lw=1.0, zorder=3)

        for thresh, ls in [(70,"--"),(80,"-."),(90,":")]:
            ax.axhline(thresh, color="#aaa", ls=ls, lw=0.8, alpha=0.7)

        ax.set_title(f"B1 full — {feat_lbl.get(feat, feat)}", fontsize=11, fontweight="bold")
        ax.set_ylabel("Balanced accuracy (%)", fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(["control","ycomp"], fontsize=10)
        ax.set_ylim(30, 108)
        ax.set_facecolor("white")
        ax.spines[["top","right"]].set_visible(False)
        ax.grid(axis="y", color="#EEEEEE", lw=0.7)

    handles = [
        mpatches.Patch(color=COL_CTRL,  label="control"),
        mpatches.Patch(color=COL_YCOMP, label="ycomp"),
        plt.Line2D([0],[0], color="#555", ls="-",  lw=2, label="LightGBM (solid)"),
        plt.Line2D([0],[0], color="#555", ls="--", lw=2, label="LogReg (hatched)"),
    ]
    fig.legend(handles=handles, fontsize=9, loc="lower center",
               ncol=4, framealpha=0.9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("LOIO — b1_full  (all labels, LGBM vs LogReg)",
                 fontsize=12, fontweight="bold", y=0.97)
    _add_fig(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, Inches(5.8))


# ── Slide 9: Summary table ─────────────────────────────────────────────────────

def _slide_summary(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "Summary — LGBM vs LogReg Balanced Accuracy",
        "Mean across folds/repeats  ·  all-labels budget  ·  key comparisons")

    rows = []

    # Cross eval
    for feat, lgbm_csv, lr_csv in [
        ("CP",      _CROSS_CP_LGBM, _CROSS_CP_LR),
        ("ilastik", _CROSS_IL_LGBM, _CROSS_IL_LR),
    ]:
        for (tr, te) in COMBOS:
            row = [f"Cross {COMBO_LABELS[(tr,te)]} ({feat})"]
            for csv_path in [lgbm_csv, lr_csv]:
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    sub = df[(df.train_batch==tr)&(df.test_batch==te)]
                    row.append(f"{sub.bal_acc.mean()*100:.1f}" if not sub.empty else "—")
                else:
                    row.append("—")
            rows.append(row)

    # Union eval
    for feat, lgbm_csv, lr_csv in [
        ("CP",      _UNION_CP_LGBM, _UNION_CP_LR),
        ("ilastik", _UNION_IL_LGBM, _UNION_IL_LR),
    ]:
        for tb in ["b1","b2v2"]:
            row = [f"Union→{tb} ({feat})"]
            for csv_path in [lgbm_csv, lr_csv]:
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    sub = df[df.test_batch==tb]
                    row.append(f"{sub.bal_acc.mean()*100:.1f}" if not sub.empty else "—")
                else:
                    row.append("—")
            rows.append(row)

    # Budget "all" mixed
    for feat, lgbm_csv, lr_csv in [
        ("CP",      _BUDGET_CP_LGBM, _BUDGET_CP_LR),
        ("ilastik", _BUDGET_IL_LGBM, _BUDGET_IL_LR),
    ]:
        for tb in ["b1","b2v2"]:
            row = [f"Budget all mixed→{tb} ({feat})"]
            for csv_path in [lgbm_csv, lr_csv]:
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    sub = df[(df.test_batch==tb)&(df.strategy=="mixed")&(df.budget==0)]
                    row.append(f"{sub.bal_acc.mean()*100:.1f}" if not sub.empty else "—")
                else:
                    row.append("—")
            rows.append(row)

    # Image efficiency b1b2v2 LOIO
    df_ie = _load_ie(_IE_B1B2_LGBM, _IE_B1B2_LR)
    if not df_ie.empty:
        for ann in ["b1","b2v2"]:
            for feat in ["cp","ilastik"]:
                row = [f"IE b1b2v2 LOIO {ann} ({feat})"]
                for clf in ["lgbm","logreg"]:
                    sub = df_ie[(df_ie.annotator==ann)&(df_ie.feature==feat)&
                                (df_ie.clf==clf)&(df_ie.experiment=="loio")&
                                (df_ie.label_budget=="all")]
                    row.append(f"{sub.bal_acc.mean()*100:.1f}" if not sub.empty else "—")
                rows.append(row)

    # Image efficiency b1_full LOIO
    df_b1f = _load_ie(_IE_B1F_LGBM, _IE_B1F_LR)
    if not df_b1f.empty:
        for feat in ["cp","ilastik"]:
            row = [f"IE b1_full LOIO ({feat})"]
            for clf in ["lgbm","logreg"]:
                sub = df_b1f[(df_b1f.feature==feat)&(df_b1f.clf==clf)&
                             (df_b1f.experiment=="loio")&(df_b1f.label_budget=="all")]
                row.append(f"{sub.bal_acc.mean()*100:.1f}" if not sub.empty else "—")
            rows.append(row)

    col_headers = ["Experiment", "LGBM (%)", "LogReg (%)"]
    n_rows = len(rows); n_cols = len(col_headers)
    row_h  = 0.30  # inches per data row
    fig_h  = max(5.0, n_rows * row_h + 0.8)

    fig, ax = plt.subplots(figsize=(12.0, fig_h), facecolor="white")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.30)
    for col in range(n_cols):
        cell = tbl[0, col]
        cell.set_facecolor("#162130")
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    for ri in range(n_rows):
        tbl[ri+1, 0].get_text().set_ha("left")
        bg = "#f0f4ff" if ri % 2 == 0 else "#ffffff"
        for ci in range(n_cols):
            tbl[ri+1, ci].set_facecolor(bg)
        try:
            v1 = float(rows[ri][1])
            v2 = float(rows[ri][2])
            winner_ci = 1 if v1 >= v2 else 2
            tbl[ri+1, winner_ci].get_text().set_fontweight("bold")
            tbl[ri+1, winner_ci].set_facecolor("#d4edda" if ri%2==0 else "#c8e6c9")
        except (ValueError, IndexError):
            pass

    # Save without bbox_inches="tight" — tight crops tables incorrectly
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    slide.shapes.add_picture(buf, Inches(0.3), Inches(1.05), width=Inches(12.7), height=Inches(6.1))


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "clf_comparison.pptx"))
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prs = _prs()

    print("  Slide 1 — Title")
    _slide_title(prs)

    print("  Slide 2 — Cross-eval LGBM vs LogReg")
    _slide_cross(prs)

    print("  Slide 3 — Union eval LGBM vs LogReg")
    _slide_union(prs)

    print("  Slide 4 — Budget curves LGBM vs LogReg")
    _slide_budget(prs)

    print("  Slide 5 — Image efficiency b1b2v2: budget curves")
    _slide_ie_b1b2v2_budget(prs)

    print("  Slide 6 — Image efficiency b1b2v2: LOIO")
    _slide_ie_b1b2v2_loio(prs)

    print("  Slide 7 — Image efficiency b1_full: budget curves")
    _slide_ie_b1full_budget(prs)

    print("  Slide 8 — Image efficiency b1_full: LOIO")
    _slide_ie_b1full_loio(prs)

    print("  Slide 9 — Summary table")
    _slide_summary(prs)

    prs.save(str(out_path))
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(prs.slides)} slides → {out_path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()

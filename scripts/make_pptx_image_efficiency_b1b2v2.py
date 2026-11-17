#!/usr/bin/env python3
"""
make_pptx_image_efficiency_b1b2v2.py

Slide deck visualizing image efficiency test results for B1/B2v2 annotators.
Tests how many images are needed to train a good adhesion classifier.

Usage:
  python scripts/make_pptx_image_efficiency_b1b2v2.py
  python scripts/make_pptx_image_efficiency_b1b2v2.py --out results/image_efficiency_b1b2v2.pptx
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as _cm
import matplotlib.colors as _mc
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pptx_le_benchmark import (
    _blank, _txt, _slide_header, _panel_bg, _rule, _setup_col_header, _setup_kv,
    C_HEAD, C_BODY, C_GREY, C_WHITE, C_PANEL, SW, SH, PP_ALIGN, RGBColor, Inches, Pt,
)
from pptx import Presentation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
CSVS = {
    "lgbm":   REPO_ROOT / "eval_results" / "image_efficiency_b1b2v2_lgbm.csv",
    "logreg": REPO_ROOT / "eval_results" / "image_efficiency_b1b2v2_logreg.csv",
}
OUT_DIR = REPO_ROOT / "results"
CLF_LABELS = {"lgbm": "LightGBM", "logreg": "LogReg"}

# ---------------------------------------------------------------------------
# Slide geometry
# ---------------------------------------------------------------------------
PLOT_LEFT  = Inches(0.35)
PLOT_TOP   = Inches(1.05)
PLOT_W     = Inches(12.6)
PLOT_H     = Inches(6.15)

# ---------------------------------------------------------------------------
# Colours (matplotlib hex strings — never RGBColor for mpl)
# ---------------------------------------------------------------------------
BUDGET_ORDER = [5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500,
                600, 700, 800, 900, 1000, 1100, 1200, "all"]
_n = len(BUDGET_ORDER)
_cmap = _cm.get_cmap("Blues")
BUDGET_COLORS = {str(b): _mc.to_hex(_cmap(0.25 + 0.60 * i / (_n - 2)))
                 for i, b in enumerate(BUDGET_ORDER[:-1])}
BUDGET_COLORS["all"] = "#0a2e5a"

BUDGET_ALPHA = {str(b): (0.35 + 0.65 * i / (_n - 2)) for i, b in enumerate(BUDGET_ORDER[:-1])}
BUDGET_ALPHA["all"] = 1.00

BUDGET_LW = {str(b): (1.0 + 1.5 * i / (_n - 2)) for i, b in enumerate(BUDGET_ORDER[:-1])}
BUDGET_LW["all"] = 2.5

# Key budgets shown in legend to avoid clutter
BUDGET_LEGEND_KEYS = {"5", "50", "200", "500", "1200", "all"}

# Condition colors for LOIO
COL_CTRL  = "#e85454"   # red for control
COL_YCOMP = "#2ca02c"   # green for ycomp

C_ACCENT_HEX = "#0F3D79"


# ---------------------------------------------------------------------------
# Core figure helpers
# ---------------------------------------------------------------------------

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


def _prs() -> Presentation:
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH
    return prs


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_data() -> pd.DataFrame:
    dfs = []
    for clf, csv in CSVS.items():
        if not csv.exists():
            print(f"  WARNING: {csv.name} not found — skipping {clf}")
            continue
        part = pd.read_csv(csv)
        if "clf" not in part.columns:
            part["clf"] = clf
        dfs.append(part)
        print(f"  Loaded {len(part):,} rows from {csv.name}")
    return pd.concat(dfs, ignore_index=True)


def _image_budget_summary(df: pd.DataFrame, annotator: str, feature: str,
                           label_budget: str, clf: str = "lgbm") -> pd.Series:
    """Mean and std of bal_acc over repeats for image_budget experiment."""
    sub = df[
        (df["experiment"]  == "image_budget") &
        (df["annotator"]   == annotator) &
        (df["feature"]     == feature) &
        (df["label_budget"]== label_budget) &
        (df["clf"]         == clf)
    ]
    grp = sub.groupby("k_images")["bal_acc"].agg(["mean", "std"])
    return grp


def _loio_data(df: pd.DataFrame, annotator: str, feature: str,
               label_budget: str = "all", clf: str = "lgbm") -> pd.DataFrame:
    """Return per-image LOIO rows for a given (annotator, feature, label_budget, clf)."""
    sub = df[
        (df["experiment"]  == "loio") &
        (df["annotator"]   == annotator) &
        (df["feature"]     == feature) &
        (df["label_budget"]== label_budget) &
        (df["clf"]         == clf)
    ].copy()
    sub["condition"] = sub["held_out_image"].str.split("_f").str[0]
    return sub


# ---------------------------------------------------------------------------
# Slide 1 — Title
# ---------------------------------------------------------------------------

def _slide_title(prs):
    slide = _blank(prs)
    _txt(slide,
         "Image Efficiency Test — B1 / B2v2 Annotators",
         Inches(0.8), Inches(1.8), Inches(11.7), Inches(1.1),
         bold=True, size_pt=34, color=C_HEAD, align=PP_ALIGN.CENTER)
    _txt(slide,
         "How many images are needed to train a good adhesion classifier?  ·  CP + ilastik  ·  22 images per annotator",
         Inches(0.8), Inches(3.0), Inches(11.7), Inches(0.55),
         size_pt=16, color=C_GREY, align=PP_ALIGN.CENTER)
    _rule(slide, Inches(3.7), width=Inches(9), left=Inches(2.17))
    _txt(slide,
         "DS1 vinculin CIO imaging  ·  B1=Margaret (393 patches)  ·  B2v2=Annabel Sep-2026 (393 patches)",
         Inches(0.8), Inches(3.9), Inches(11.7), Inches(0.45),
         size_pt=12, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide,
         "Exp 1: image budget sweep (k ∈ {1–19}, 20 repeats × label budgets)  ·  Exp 2: leave-one-image-out (LOIO)",
         Inches(0.8), Inches(4.4), Inches(11.7), Inches(0.45),
         size_pt=12, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide,
         "Classifiers: LightGBM & LogReg  ·  Metric: balanced accuracy",
         Inches(0.8), Inches(4.85), Inches(11.7), Inches(0.35),
         size_pt=11, color=RGBColor(0x44, 0x44, 0x88), align=PP_ALIGN.CENTER)


# ---------------------------------------------------------------------------
# Slide 2 — Experiment Design
# ---------------------------------------------------------------------------

def _slide_design(prs):
    slide = _blank(prs)
    _slide_header(slide,
                  "Experiment Design — Image Efficiency Test",
                  "How many training images are needed? · B1 (Margaret) + B2v2 (Annabel) · CP + ilastik · LightGBM")

    CONTENT_TOP = Inches(1.05)
    CONTENT_H   = Inches(6.15)
    PAD         = Inches(0.14)

    # Two columns: left=Data & Setup, right=Experiments
    col_w = [Inches(5.8), Inches(7.2)]
    col_l = [Inches(0.18),
             Inches(0.18) + col_w[0] + PAD]

    # ── Col 0: Data & Setup ──────────────────────────────────────────────
    _panel_bg(slide, col_l[0], CONTENT_TOP, col_w[0], CONTENT_H)
    y = _setup_col_header(slide, "Data & Setup", col_l[0], CONTENT_TOP, col_w[0]) + Inches(0.12)

    for label, val in [
        ("Dataset",       "DS1 — vinculin channel (CIO imaging)"),
        ("Images",        "22 images per annotator\n(from DS1 set, 4 sparse images excluded)"),
        ("Conditions",    "10 control + 12 ycomp images"),
        ("B1 annotator",  "Margaret — 393 patches total"),
        ("B2v2 annotator","Annabel (Sep-2026) — 393 patches total"),
        ("Features",      "CellProfiler (50 features)\nilastik pixel filters (80 features)"),
        ("Classifier",    "LightGBM (class-balanced weights)"),
        ("Metric",        "Balanced accuracy = mean(recall_adh, recall_noad)"),
        ("Patch size",    "32 × 32 px, PAX (paxillin) channel"),
    ]:
        y = _setup_kv(slide, label, val, col_l[0], y, col_w[0])
        if y > CONTENT_TOP + CONTENT_H - Inches(0.1):
            break

    # ── Col 1: Experiments ───────────────────────────────────────────────
    _panel_bg(slide, col_l[1], CONTENT_TOP, col_w[1], CONTENT_H)
    y = _setup_col_header(slide, "Two Experiments", col_l[1], CONTENT_TOP, col_w[1]) + Inches(0.12)

    for label, val in [
        ("Exp 1 — Image Budget",
         "Training image count: k ∈ {1, 2, 3, 5, 7, 10, 13, 16, 19}\n"
         "Random selection: 20 repeats per k value\n"
         "Label budget: 5→1200 (19 levels) + all\n"
         "Test set: all remaining images (not in training pool)\n"
         "→ Measures: how accuracy scales with more training images"),
        ("Exp 2 — Leave-One-Image-Out (LOIO)",
         "Train on 21 images, test on 1 held-out image\n"
         "Repeat for each of the 22 images\n"
         "Label budgets: 5→1200 (19 levels) + all\n"
         "→ Measures: per-image generalizability; identifies hard/easy images\n"
         "→ k=21 (all training images), held-out image rotated through all 22"),
        ("Key question",
         "How many images are needed to reach 70%/75%/80%/85%/90%/95% balanced accuracy?\n"
         "Does B2v2 (Annabel) require fewer images than B1 (Margaret)?\n"
         "Which feature set (CP vs ilastik) is more image-efficient?"),
    ]:
        y = _setup_kv(slide, label, val, col_l[1], y, col_w[1], label_pt=10, val_pt=9)
        y += Inches(0.05)
        if y > CONTENT_TOP + CONTENT_H - Inches(0.1):
            break


# ---------------------------------------------------------------------------
# Slide 3 — Image Budget Curves (2×2 matplotlib)
# ---------------------------------------------------------------------------

def _fig_image_budget(df: pd.DataFrame, clf: str) -> plt.Figure:
    """2×2 grid: rows=annotator (b1, b2v2), cols=feature (cp, ilastik)."""
    annotators  = ["b1", "b2v2"]
    features    = ["cp", "ilastik"]
    ann_labels  = {"b1": "B1 (Margaret)", "b2v2": "B2v2 (Annabel)"}
    feat_labels = {"cp": "CellProfiler", "ilastik": "ilastik"}

    budgets_to_show = [str(b) for b in BUDGET_ORDER]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor="white")
    fig.subplots_adjust(hspace=0.38, wspace=0.28, left=0.07, right=0.97,
                        top=0.93, bottom=0.09)

    for row_i, ann in enumerate(annotators):
        for col_i, feat in enumerate(features):
            ax = axes[row_i, col_i]

            sub_all = df[
                (df["experiment"] == "image_budget") &
                (df["annotator"]  == ann) &
                (df["feature"]    == feat) &
                (df["clf"]        == clf)
            ]
            k_vals = sorted(sub_all["k_images"].unique())

            legend_handles = []
            for lb in budgets_to_show:
                grp = _image_budget_summary(df, ann, feat, lb, clf=clf)
                if grp.empty:
                    continue

                x_plot = [k for k in k_vals if k in grp.index]
                means  = np.array([grp.loc[k, "mean"] * 100 for k in x_plot])
                stds   = np.array([grp.loc[k, "std"]  * 100 for k in x_plot])

                valid = ~np.isnan(means)
                xv, mv, sv = np.array(x_plot)[valid], means[valid], stds[valid]

                alpha = BUDGET_ALPHA[lb]
                lw    = BUDGET_LW[lb]
                color = BUDGET_COLORS[lb]
                ms    = 5 if lb == "all" else 3
                in_legend = lb in BUDGET_LEGEND_KEYS
                lbl = (f"N={lb}" if lb != "all" else "all labels") if in_legend else None

                ax.plot(xv, mv, color=color, linewidth=lw,
                        marker="o", markersize=ms, alpha=alpha,
                        label=lbl, zorder=3 if lb == "all" else 2)

                if lb == "all":
                    ax.fill_between(xv, mv - sv, mv + sv,
                                    color=color, alpha=0.13, zorder=1)

                if in_legend:
                    legend_handles.append(
                        mpatches.Patch(color=color, alpha=max(alpha, 0.6), label=lbl)
                    )

            for thresh, ls in [(70, "--"), (75, ":"), (80, "-."), (85, "--"), (90, ":"), (95, "-.")]:
                ax.axhline(thresh, color="#aaaaaa", linestyle=ls, linewidth=0.9,
                           alpha=0.7, zorder=0)
                if k_vals:
                    ax.text(k_vals[-1] + 0.2, thresh + 0.3, f"{thresh}%",
                            color="#999999", fontsize=7, va="bottom", ha="left")

            title = f"{ann_labels[ann]}  —  {feat_labels[feat]}"
            ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
            ax.set_xlabel("Training images (k)", fontsize=10)
            ax.set_ylabel("Balanced accuracy (%)", fontsize=10)
            if k_vals:
                ax.set_xlim(0.5, k_vals[-1] + 1.5)
                ax.set_xticks(k_vals)
                ax.set_xticklabels([str(k) for k in k_vals], fontsize=8)
            ax.set_ylim(42, 105)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_facecolor("white")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#EEEEEE", linewidth=0.7, zorder=0)

            if row_i == 0 and col_i == 1:
                ax.legend(handles=legend_handles, fontsize=7.5, framealpha=0.9,
                          loc="lower right", ncol=2)

    fig.suptitle(f"Image Budget Curves — {CLF_LABELS[clf]}",
                 fontsize=13, fontweight="bold", y=0.975)
    return fig


def _slide_image_budget(prs, df: pd.DataFrame, clf: str):
    slide = _blank(prs)
    _slide_header(slide,
                  f"Exp 1 — Image Budget: Balanced Accuracy vs. Training Images ({CLF_LABELS[clf]})",
                  "B1 (Margaret) vs B2v2 (Annabel) · CP + ilastik · k ∈ {1–19} · 20 repeats · label budgets 5→1200 + all")
    fig = _fig_image_budget(df, clf)
    _add_fig(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, PLOT_H)


# ---------------------------------------------------------------------------
# Slide 4 — LOIO Results (2×2 matplotlib)
# ---------------------------------------------------------------------------

def _fig_loio(df: pd.DataFrame, clf: str) -> plt.Figure:
    """2×2 violin + scatter, rows=annotator, cols=feature, colored by condition."""
    annotators  = ["b1", "b2v2"]
    features    = ["cp", "ilastik"]
    ann_labels  = {"b1": "B1 (Margaret)", "b2v2": "B2v2 (Annabel)"}
    feat_labels = {"cp": "CellProfiler", "ilastik": "ilastik"}
    cond_labels = {"control": "control", "ycomp": "ycomp"}
    cond_colors = {"control": COL_CTRL, "ycomp": COL_YCOMP}

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor="white")
    fig.subplots_adjust(hspace=0.42, wspace=0.3, left=0.08, right=0.93,
                        top=0.97, bottom=0.09)

    for row_i, ann in enumerate(annotators):
        for col_i, feat in enumerate(features):
            ax = axes[row_i, col_i]

            loio = _loio_data(df, ann, feat, label_budget="all", clf=clf)

            if loio.empty:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=12)
                continue

            conditions = ["control", "ycomp"]
            x_pos      = [1, 2]
            all_vals_ctrl  = loio[loio["condition"] == "control"]["bal_acc"].values * 100
            all_vals_ycomp = loio[loio["condition"] == "ycomp"]["bal_acc"].values * 100

            cond_data = {
                "control": all_vals_ctrl,
                "ycomp":   all_vals_ycomp,
            }

            # Violin plot for each condition
            vp_data  = []
            vp_pos   = []
            vp_color = []

            for xi, cond in zip(x_pos, conditions):
                vals = cond_data[cond]
                if len(vals) < 2:
                    continue
                vp_data.append(vals)
                vp_pos.append(xi)
                vp_color.append(cond_colors[cond])

            if vp_data:
                parts = ax.violinplot(vp_data, positions=vp_pos,
                                      showmedians=True, showextrema=True,
                                      widths=0.55)
                for pc, color in zip(parts["bodies"], vp_color):
                    pc.set_facecolor(color)
                    pc.set_alpha(0.30)
                parts["cmedians"].set_color("#333333")
                parts["cmedians"].set_linewidth(2.0)
                parts["cbars"].set_color("#777777")
                parts["cmins"].set_color("#777777")
                parts["cmaxes"].set_color("#777777")

            # Scatter individual image points
            rng = np.random.default_rng(42)
            for xi, cond in zip(x_pos, conditions):
                vals = cond_data[cond]
                if len(vals) == 0:
                    continue
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(xi + jitter, vals, color=cond_colors[cond],
                           s=30, alpha=0.75, zorder=4, edgecolors="white",
                           linewidths=0.4)
                mn = np.mean(vals)
                sd = np.std(vals)
                # Place stats below the x-tick labels (outside axes)
                ax.annotate(
                    f"μ={mn:.1f}%\nσ={sd:.1f}%",
                    xy=(xi, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -32), textcoords="offset points",
                    ha="center", va="top", fontsize=7.5, color="#444444",
                    clip_on=False)

            # Overall stats (all images) — placed outside plot, above top-right
            all_vals = loio["bal_acc"].values * 100
            overall_mean = np.mean(all_vals)
            overall_std  = np.std(all_vals)
            ax.annotate(
                f"All images\nμ = {overall_mean:.1f}%  σ = {overall_std:.1f}%",
                xy=(1.0, 1.0), xycoords="axes fraction",
                xytext=(4, 4), textcoords="offset points",
                ha="left", va="bottom",
                fontsize=9, color="#222222",
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F4FF",
                          edgecolor="#BBC8EE", alpha=0.95))

            # Reference lines
            for thresh, ls in [(70, "--"), (75, ":"), (80, "-."), (85, "--"), (90, ":"), (95, "-.")]:
                ax.axhline(thresh, color="#bbbbbb", linestyle=ls, linewidth=0.9,
                           alpha=0.8, zorder=0)
                ax.text(2.32, thresh + 0.3, f"{thresh}%",
                        color="#999999", fontsize=7, va="bottom")

            title = f"{ann_labels[ann]}  —  {feat_labels[feat]}"
            ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
            ax.set_ylabel("Balanced accuracy (%)", fontsize=10)
            ax.set_ylim(-8, 110)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([cond_labels[c] for c in conditions], fontsize=10)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_facecolor("white")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#EEEEEE", linewidth=0.7, zorder=0)

            # Legend (only one panel)
            if row_i == 0 and col_i == 0:
                legend_patches = [
                    mpatches.Patch(color=COL_CTRL,  alpha=0.7, label="control"),
                    mpatches.Patch(color=COL_YCOMP, alpha=0.7, label="ycomp"),
                ]
                ax.legend(handles=legend_patches, fontsize=9, framealpha=0.9,
                          loc="lower right")

    fig.suptitle(f"LOIO — {CLF_LABELS[clf]}",
                 fontsize=13, fontweight="bold", y=0.975)
    return fig


def _slide_loio(prs, df: pd.DataFrame, clf: str):
    slide = _blank(prs)
    _slide_header(slide,
                  f"Exp 2 — LOIO: Per-Image Generalizability ({CLF_LABELS[clf]})",
                  "B1 vs B2v2 · CP + ilastik · all-label budget · 22 held-out images · violin + scatter colored by condition")
    fig = _fig_loio(df, clf)
    _add_fig(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, PLOT_H)


# ---------------------------------------------------------------------------
# Slide 5 — Summary / Discussion
# ---------------------------------------------------------------------------

def _compute_threshold_k(df: pd.DataFrame, annotator: str, feature: str,
                          threshold: float, clf: str = "lgbm") -> str:
    """Find minimum k to reach threshold at 'all' label budget."""
    grp = _image_budget_summary(df, annotator, feature, "all", clf=clf)
    for k in sorted(grp.index):
        if grp.loc[k, "mean"] >= threshold:
            return str(k)
    return "not reached"


def _slide_summary(prs, df: pd.DataFrame):
    slide = _blank(prs)
    _slide_header(slide,
                  "Summary — Image Efficiency: B1 vs B2v2",
                  "Key numbers, annotator comparison, and LOIO interpretation")

    CONTENT_TOP = Inches(1.05)
    CONTENT_H   = Inches(6.15)
    PAD         = Inches(0.14)

    col_w = [Inches(4.2), Inches(4.2), Inches(4.7)]
    col_l = [Inches(0.18),
             Inches(0.18) + col_w[0] + PAD,
             Inches(0.18) + col_w[0] + col_w[1] + 2 * PAD]

    # Pre-compute threshold numbers
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    def _thresh_row(ann, feat, t):
        k = _compute_threshold_k(df, ann, feat, t)
        return f"{int(t*100)}% → k = {k}"

    # LOIO stats at all-budget
    clf = "lgbm"   # summary uses lgbm as reference
    loio_stats = {}
    for ann in ["b1", "b2v2"]:
        for feat in ["cp", "ilastik"]:
            sub = df[(df["experiment"]=="loio") & (df["annotator"]==ann) &
                     (df["feature"]==feat) & (df["label_budget"]=="all") &
                     (df["clf"]==clf)]
            loio_stats[(ann, feat)] = (sub["bal_acc"].mean() * 100,
                                       sub["bal_acc"].std() * 100)

    # ── Col 0: Image Budget Thresholds ──────────────────────────────────
    _panel_bg(slide, col_l[0], CONTENT_TOP, col_w[0], CONTENT_H)
    y = _setup_col_header(slide, "Image Budget Thresholds (all labels)",
                          col_l[0], CONTENT_TOP, col_w[0]) + Inches(0.12)

    for ann, ann_lbl in [("b1", "B1 (Margaret)"), ("b2v2", "B2v2 (Annabel)")]:
        for feat, feat_lbl in [("cp", "CellProfiler"), ("ilastik", "ilastik")]:
            lines = "\n".join(_thresh_row(ann, feat, t) for t in thresholds)
            y = _setup_kv(slide, f"{ann_lbl} — {feat_lbl}", lines,
                          col_l[0], y, col_w[0])
            y += Inches(0.04)
        if y > CONTENT_TOP + CONTENT_H - Inches(0.1):
            break

    # ── Col 1: B1 vs B2v2 Comparison ────────────────────────────────────
    _panel_bg(slide, col_l[1], CONTENT_TOP, col_w[1], CONTENT_H)
    y = _setup_col_header(slide, "B1 vs B2v2 Annotator Comparison",
                          col_l[1], CONTENT_TOP, col_w[1]) + Inches(0.12)

    comparison_items = [
        ("B2v2 is more image-efficient",
         "B2v2 reaches 80% with k=3 (CP)\n"
         "B1 never reaches 80% across k=1–19\n"
         "B2v2 labels are easier to learn from"),
        ("At k=5 (all labels, LGBM)",
         f"B2v2 CP: {_image_budget_summary(df,'b2v2','cp','all',clf='lgbm').loc[5,'mean']*100:.1f}%\n"
         f"B1  CP: {_image_budget_summary(df,'b1','cp','all',clf='lgbm').loc[5,'mean']*100:.1f}%\n"
         "Gap ≈ 25 percentage points"),
        ("CP vs ilastik",
         "CP slightly better than ilastik for B2v2\n"
         "Similar performance for B1\n"
         "Both features show same annotator pattern"),
        ("Saturation",
         "B2v2 saturates around k=5–7 (CP)\n"
         "B1 keeps slowly improving up to k=13\n"
         "B1 may benefit from even more images"),
    ]
    for label, val in comparison_items:
        y = _setup_kv(slide, label, val, col_l[1], y, col_w[1])
        y += Inches(0.05)
        if y > CONTENT_TOP + CONTENT_H - Inches(0.1):
            break

    # ── Col 2: LOIO Interpretation ──────────────────────────────────────
    _panel_bg(slide, col_l[2], CONTENT_TOP, col_w[2], CONTENT_H)
    y = _setup_col_header(slide, "LOIO Results & Interpretation",
                          col_l[2], CONTENT_TOP, col_w[2]) + Inches(0.12)

    loio_items = []
    for ann, ann_lbl in [("b1", "B1 (Margaret)"), ("b2v2", "B2v2 (Annabel)")]:
        for feat, feat_lbl in [("cp", "CellProfiler"), ("ilastik", "ilastik")]:
            mn, sd = loio_stats[(ann, feat)]
            loio_items.append(
                (f"LOIO — {ann_lbl} / {feat_lbl}",
                 f"Mean: {mn:.1f}%  ±  {sd:.1f}%")
            )

    loio_items += [
        ("B2v2 >> B1 on LOIO",
         f"B2v2 CP: {loio_stats[('b2v2','cp')][0]:.1f}%  vs  "
         f"B1 CP: {loio_stats[('b1','cp')][0]:.1f}%\n"
         "Same pattern as label efficiency test\n"
         "B2v2 patches are more consistently classifiable"),
        ("Control vs ycomp",
         "Both conditions present in LOIO test\n"
         "Individual image difficulty varies widely\n"
         "Some images have very few patches → noisy estimate"),
        ("Interpretation",
         "B2v2 labels are higher quality / more consistent\n"
         "Or B2v2 dataset has less within-annotator variance\n"
         "B1 may have more ambiguous borderline patches\n"
         "Fewer images needed if labels are clean and consistent"),
    ]

    for label, val in loio_items:
        y = _setup_kv(slide, label, val, col_l[2], y, col_w[2])
        y += Inches(0.04)
        if y > CONTENT_TOP + CONTENT_H - Inches(0.1):
            break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DIR / "image_efficiency_b1b2v2.pptx"))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data …")
    df = _load_data()
    print(f"  Total: {len(df):,} rows  |  clfs: {sorted(df['clf'].unique())}")

    print("Building presentation …")
    prs = _prs()

    print("  Slide 1: Title")
    _slide_title(prs)

    print("  Slide 2: Experiment Design")
    _slide_design(prs)

    clfs_present = [c for c in ["lgbm", "logreg"] if c in df["clf"].unique()]
    for clf in clfs_present:
        print(f"  Slide: Image Budget Curves ({clf})")
        _slide_image_budget(prs, df, clf)

    for clf in clfs_present:
        print(f"  Slide: LOIO Results ({clf})")
        _slide_loio(prs, df, clf)

    print("  Slide: Summary / Discussion")
    _slide_summary(prs, df)

    prs.save(str(out_path))
    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved {len(prs.slides)} slides → {out_path}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()

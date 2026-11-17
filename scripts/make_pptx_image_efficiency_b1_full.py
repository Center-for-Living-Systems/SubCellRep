#!/usr/bin/env python3
"""
make_pptx_image_efficiency_b1_full.py

Build image_efficiency_b1_full.pptx from eval_results/image_efficiency_b1_full.csv.

B1 = Margaret's full label set (labels_vinc_20260521.csv):
  ~1310 patches across 89 frames (50 control + 39 ycomp), 2-class.
  CP and ilastik features.

Usage:
  python scripts/make_pptx_image_efficiency_b1_full.py [--out PATH]
"""
from __future__ import annotations
import argparse, io, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as _cm
import matplotlib.colors as _mc
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

REPO    = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results"
CSVS = {
    "lgbm":   REPO / "eval_results" / "image_efficiency_b1_full_lgbm.csv",
    "logreg": REPO / "eval_results" / "image_efficiency_b1_full_logreg.csv",
}

FEAT_LABELS = {"cp": "CellProfiler", "ilastik": "ilastik"}
FEAT_COLORS = {"cp": "#2e86ab",       "ilastik": "#e07b39"}
CLF_LABELS  = {"lgbm": "LightGBM",   "logreg": "LogReg"}

BUDGET_ORDER  = [5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500,
                 600, 700, 800, 900, 1000, 1100, 1200, "all"]

# Generate a blue sequential gradient; "all" gets a near-black anchor.
_n = len(BUDGET_ORDER)
_cmap = _cm.get_cmap("Blues")
BUDGET_COLORS = {b: _mc.to_hex(_cmap(0.25 + 0.60 * i / (_n - 2)))
                 for i, b in enumerate(BUDGET_ORDER[:-1])}
BUDGET_COLORS["all"] = "#08205a"

# Legend: show only these key budgets to avoid clutter on the curves slide.
BUDGET_LEGEND_KEYS = [5, 50, 200, 500, 1200, "all"]


def _prs():
    prs = Presentation()
    prs.slide_width  = SW
    prs.slide_height = SH
    return prs


def _add_fig(slide, fig, left, top, w, h):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    buf.seek(0)
    slide.shapes.add_picture(buf, left, top, width=w, height=h)


# ── slide 1: title ────────────────────────────────────────────────────────────

def _slide_title(prs):
    slide = _blank(prs)
    _txt(slide, "Image Efficiency Test — Margaret (B1 Full Set)",
         Inches(1.0), Inches(2.3), Inches(11.3), Inches(1.2),
         bold=True, size_pt=34, color=C_HEAD, align=PP_ALIGN.CENTER)
    _txt(slide, "How many images are needed to train a good adhesion classifier?",
         Inches(1.0), Inches(3.6), Inches(11.3), Inches(0.5),
         size_pt=18, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide,
         "Margaret's full annotation · ~1310 patches · 89 frames (50 control + 39 ycomp) · "
         "CP + ilastik features · LightGBM & LogReg",
         Inches(1.0), Inches(4.2), Inches(11.3), Inches(0.4),
         size_pt=12, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide,
         "Multi-class labels (focal adhesion / Nascent / fibrillar / focal complex / No adhesion) "
         "converted to binary · Uncertain excluded",
         Inches(1.0), Inches(4.65), Inches(11.3), Inches(0.4),
         size_pt=10, color=C_GREY, align=PP_ALIGN.CENTER)


# ── slide 2: experiment design ────────────────────────────────────────────────

def _slide_design(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "Experiment Design — Image Budget Study & LOIO",
        "Vary training image count (K) · test on held-out images · nested label budget sweep")

    CT = Inches(1.1); CH = Inches(6.0); PAD = Inches(0.15)
    cw = [Inches(4.6), Inches(3.8), Inches(3.8)]
    cl = [Inches(0.2), Inches(0.2)+cw[0]+PAD, Inches(0.2)+cw[0]+cw[1]+2*PAD]

    _panel_bg(slide, cl[0], CT, cw[0], CH)
    y = _setup_col_header(slide, "Data & Setup", cl[0], CT, cw[0]) + Inches(0.1)
    for lbl, val in [
        ("Dataset",    "DS1 — vinculin channel\n(CIO imaging)"),
        ("Annotator",  "Margaret (B1 full set)"),
        ("Label file", "labels_vinc_20260521.csv"),
        ("Raw patches","1340 patches · 89 frames"),
        ("Excluded",   "'Uncertain' class (~30 patches)"),
        ("Usable",     "~1310 patches · 89 images\nadh≈47%  no-ad≈53%"),
        ("Conditions", "50 control images\n39 ycomp images"),
        ("Label mapping",
         "adhesion: focal adhesion, Nascent\n"
         "  Adhesion, fibrillar adhesion,\n  focal complex\n"
         "No adhesion: No adhesion"),
        ("Features",   "CellProfiler morphology\nilastik pixel filters"),
        ("Classifier", "LightGBM & LogReg (balanced weights)"),
        ("Metric",     "Balanced accuracy\n= mean(recall_adh, recall_no-adh)"),
    ]:
        y = _setup_kv(slide, lbl, val, cl[0], y, cw[0], val_pt=8.5)

    _panel_bg(slide, cl[1], CT, cw[1], CH)
    y = _setup_col_header(slide, "Exp 1 — Image Budget", cl[1], CT, cw[1]) + Inches(0.1)
    for lbl, val in [
        ("K (train images)", "1, 2, 3, 5, 7, 10, 15,\n20, 30, 50, 70"),
        ("Repeats",   "20 random subsets per K"),
        ("Test set",  "All images not in training\n(89 − K images)"),
        ("Label budget",
         "Within each training set:\n"
         "N = 5, 10, 20, 50, 100, 150,\n"
         "200, 250, 300, 400, 500, 600,\n"
         "700, 800, 900, 1000, 1100,\n"
         "1200, all"),
        ("Goal", "How many images are\nneeded to approach\nfull-data performance?"),
    ]:
        y = _setup_kv(slide, lbl, val, cl[1], y, cw[1], val_pt=8.0)

    _panel_bg(slide, cl[2], CT, cw[2], CH)
    y = _setup_col_header(slide, "Exp 2 — LOIO", cl[2], CT, cw[2]) + Inches(0.1)
    for lbl, val in [
        ("Train",      "88 images (all − 1)"),
        ("Test",       "1 held-out image"),
        ("Repeats",    "Each image held out once\n= 89 LOIO runs"),
        ("Label budget",
         "N = 5 → 1200 (19 levels) + all\n(within training pool)"),
        ("Goal",
         "Which individual images\nare hard/easy to classify?\nInspect per-image variability."),
    ]:
        y = _setup_kv(slide, lbl, val, cl[2], y, cw[2])


# ── shared plot helper ────────────────────────────────────────────────────────

def _plot_budget_ax(ax, sub_fc, n_images, first_col: bool):
    """Draw budget curves on ax for one (feat, clf) panel."""
    for bud in BUDGET_ORDER:
        sub_b = sub_fc[sub_fc.label_budget.astype(str) == str(bud)]
        if sub_b.empty:
            continue
        g = sub_b.groupby("k_images")["bal_acc"]
        means = g.mean(); stds = g.std()
        xs = sorted(means.index)
        ys = [means[x] for x in xs]
        es = [stds.get(x, 0) for x in xs]
        lw = 2.2 if bud == "all" else 1.0
        alpha = 1.0 if bud == "all" else 0.65
        ls = "-" if bud == "all" else "--"
        color = BUDGET_COLORS[bud]
        in_legend = bud in BUDGET_LEGEND_KEYS
        label = (f"N={bud}" if bud != "all" else "all labels") if in_legend else None
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, ls=ls,
                marker="o", markersize=2.5, label=label, zorder=3)
        if bud == "all":
            ax.fill_between(xs, [y-e for y,e in zip(ys,es)],
                                [y+e for y,e in zip(ys,es)],
                            color=color, alpha=0.12, zorder=2)
    ax.axhline(0.5, color="#aaa", lw=0.7, ls=":", zorder=0)
    ax.set_xlabel("# Training images (K)", fontsize=9)
    if first_col:
        ax.set_ylabel("Balanced accuracy", fontsize=9)
    ax.set_ylim(0.42, 1.02)
    ax.set_xticks(n_images)
    ax.set_xticklabels([str(k) for k in n_images], fontsize=7, rotation=30)
    ax.legend(fontsize=7.5, framealpha=0.9, loc="lower right")
    ax.set_facecolor("white"); ax.grid(axis="y", color="#eee", lw=0.7)
    ax.spines[["top","right"]].set_visible(False)


# ── slide 3: image budget curves (2×2: rows=feat, cols=clf) ──────────────────

def _slide_budget(prs, df: pd.DataFrame):
    slide = _blank(prs)
    _slide_header(slide,
        "Image Budget Curves — LightGBM vs LogReg",
        "rows: feature · cols: classifier · lines = label budget · mean ± std (20 repeats)")

    sub = df[df.experiment == "image_budget"].copy()
    feats = ["cp", "ilastik"]
    clfs  = ["lgbm", "logreg"]
    n_images = sorted(sub.k_images.unique())

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 5.6), facecolor="white")

    for row, feat in enumerate(feats):
        for col, clf in enumerate(clfs):
            ax = axes[row][col]
            sub_fc = sub[(sub.feature == feat) & (sub.clf == clf)]
            _plot_budget_ax(ax, sub_fc, n_images, first_col=(col == 0))
            title = f"{FEAT_LABELS[feat]} — {CLF_LABELS[clf]}"
            ax.set_title(title, fontsize=10, fontweight="bold", pad=4)

    fig.tight_layout(h_pad=0.6, w_pad=0.8)
    _add_fig(slide, fig, Inches(0.2), Inches(1.05), Inches(13.0), Inches(6.1))


# ── slide 4: LOIO results (2×2: rows=feat, cols=clf) ─────────────────────────

def _slide_loio(prs, df: pd.DataFrame):
    from matplotlib.patches import Patch
    slide = _blank(prs)
    _slide_header(slide,
        "LOIO Results — Per-Image Balanced Accuracy",
        "Train on 88 images → test on 1 held-out image · all-label budget · sorted by performance")

    sub = df[(df.experiment == "loio") & (df.label_budget.astype(str) == "all")].copy()
    feats = ["cp", "ilastik"]
    clfs  = ["lgbm", "logreg"]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 5.6), facecolor="white")

    for row, feat in enumerate(feats):
        for col, clf in enumerate(clfs):
            ax = axes[row][col]
            sub_fc = sub[(sub.feature == feat) & (sub.clf == clf)].copy()
            if sub_fc.empty:
                ax.text(0.5, 0.5, "pending", ha="center", va="center",
                        transform=ax.transAxes, color="grey")
                ax.axis("off"); continue

            by_img = sub_fc.groupby("held_out_image")["bal_acc"].mean().sort_values()
            xs = range(len(by_img))
            bar_colors = ["#1f77b4" if "control" in img else "#d62728"
                          for img in by_img.index]
            ax.bar(xs, list(by_img.values), color=bar_colors, alpha=0.8, width=0.7, zorder=2)
            ax.axhline(sub_fc.bal_acc.mean(), color="#333", lw=1.2, ls="--")
            ax.axhline(0.5, color="#aaa", lw=0.7, ls=":", zorder=0)
            ax.set_ylim(0.0, 1.05)
            if col == 0:
                ax.set_ylabel("Balanced accuracy", fontsize=8)
            ax.set_title(f"{FEAT_LABELS[feat]} — {CLF_LABELS[clf]}", fontsize=9, fontweight="bold")
            ax.set_xticks(list(xs))
            ax.set_xticklabels(list(by_img.index), fontsize=5.5, rotation=45, ha="right")
            ax.legend(handles=[
                Patch(color="#1f77b4", label="control"),
                Patch(color="#d62728", label="ycomp"),
                plt.Line2D([0],[0], color="#333", ls="--", lw=1.2,
                           label=f"mean={sub_fc.bal_acc.mean():.3f} ± {sub_fc.bal_acc.std():.3f}"),
            ], fontsize=7, loc="lower right")
            ax.set_facecolor("white"); ax.grid(axis="y", color="#eee", lw=0.7)
            ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    _add_fig(slide, fig, Inches(0.2), Inches(1.05), Inches(13.0), Inches(6.1))


# ── slide 5: LOIO label budget comparison (2×2: rows=feat, cols=clf) ─────────

def _slide_loio_budget(prs, df: pd.DataFrame):
    slide = _blank(prs)
    _slide_header(slide,
        "LOIO — Label Budget Effect",
        "How many labels per image are needed? · mean ± std (89 LOIO runs)")

    sub = df[df.experiment == "loio"].copy()
    budgets = [b for b in BUDGET_ORDER if sub.label_budget.astype(str).str.contains(str(b)).any()]
    feats = ["cp", "ilastik"]
    clfs  = ["lgbm", "logreg"]
    xs = list(range(len(budgets)))
    colors = [BUDGET_COLORS[b] for b in budgets]
    xlabels = [f"N={b}" if b != "all" else "all" for b in budgets]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 5.6), facecolor="white")

    for row, feat in enumerate(feats):
        for col, clf in enumerate(clfs):
            ax = axes[row][col]
            sub_fc = sub[(sub.feature == feat) & (sub.clf == clf)]
            means, stds = [], []
            for bud in budgets:
                sb = sub_fc[sub_fc.label_budget.astype(str) == str(bud)]
                means.append(sb.bal_acc.mean() if not sb.empty else float("nan"))
                stds.append(sb.bal_acc.std() if not sb.empty else 0.0)

            bars = ax.bar(xs, means, color=colors, alpha=0.85, width=0.65, zorder=2)
            ax.errorbar(xs, means, stds, fmt="none", color="#333", capsize=3, lw=1.0, zorder=3)
            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, fontsize=6.5, rotation=45, ha="right")
            ax.set_ylim(0.0, 1.05)
            if col == 0:
                ax.set_ylabel("Balanced accuracy", fontsize=8)
            ax.set_title(f"{FEAT_LABELS[feat]} — {CLF_LABELS[clf]}", fontsize=9, fontweight="bold")
            ax.axhline(0.5, color="#aaa", lw=0.7, ls=":", zorder=0)
            ax.set_facecolor("white"); ax.grid(axis="y", color="#eee", lw=0.7)
            ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    _add_fig(slide, fig, Inches(0.2), Inches(1.05), Inches(13.0), Inches(6.1))


# ── build ─────────────────────────────────────────────────────────────────────

def build(out_path: Path):
    print("Building B1 full set image efficiency PPT (lgbm + logreg) …")
    dfs = []
    for clf, csv in CSVS.items():
        if not csv.exists():
            print(f"  WARNING: {csv.name} not found — skipping {clf}")
            continue
        part = pd.read_csv(csv)
        if "clf" not in part.columns:
            part["clf"] = clf
        dfs.append(part)
        print(f"  Loaded {len(part)} rows from {csv.name}")
    if not dfs:
        print("ERROR: no CSV files found"); return
    df = pd.concat(dfs, ignore_index=True)

    prs = _prs()
    print("  Slide 1 — Title")
    _slide_title(prs)
    print("  Slide 2 — Experiment design")
    _slide_design(prs)
    print("  Slide 3 — Image budget curves")
    _slide_budget(prs, df)
    print("  Slide 4 — LOIO per-image results")
    _slide_loio(prs, df)
    print("  Slide 5 — LOIO label budget comparison")
    _slide_loio_budget(prs, df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    print(f"\nSaved → {out_path}  ({out_path.stat().st_size // 1024} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "image_efficiency_b1_full.pptx"))
    build(Path(ap.parse_args().out))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_pptx_le_b1b2v2_matched.py

Build le_b1b2v2_matched.pptx — B1/B2v2 matched-annotator comparison deck.
CP and ilastik only (no SupCon).

B1   = Margaret  (~393 patches after sparse-image exclusion)
B2v2 = Annabel Sep-2026 (~393 patches)

Usage:
  python scripts/make_pptx_le_b1b2v2_matched.py [--out PATH]
"""
from __future__ import annotations
import argparse, io, json, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── import infrastructure from main PPT script ────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pptx_le_benchmark import (
    _blank, _txt, _slide_header, _add_fig, _rule, _panel_bg,
    _bullet_box, _setup_col_header, _setup_kv,
    _load_error_patches_grid, _load_tif,
    C_HEAD, C_BODY, C_GREY, C_WHITE, C_PANEL, C_ACCENT,
    C_ROW_A, C_ROW_B, SW, SH,
    PLOT_LEFT, PLOT_TOP, PLOT_W, PLOT_H,
    PP_ALIGN, RGBColor, Inches, Pt,
)
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

# ── paths ─────────────────────────────────────────────────────────────────────
DATA       = Path("/net/projects/CLS/lding/data/fa_data_analysis")
EVAL_DIR   = DATA / "ae_results" / "features" / "eval_results" / "b1b2v2"
FS_DIR     = DATA / "labelling" / "le_b1b2v2_matched"
PATCH_DIR  = DATA / "ae_results" / "patches" / "cio" / "vinc"
OUT_DIR    = Path(__file__).resolve().parents[1] / "results"

_CROSS_CP   = EVAL_DIR / "le_b1b2v2_cross_cp.csv"
_CROSS_IL   = EVAL_DIR / "le_b1b2v2_cross_ilastik.csv"
_CROSS_ERR_CP = EVAL_DIR / "le_b1b2v2_cross_errors_cp.json"
_CROSS_ERR_IL = EVAL_DIR / "le_b1b2v2_cross_errors_ilastik.json"
_UNION_CP   = EVAL_DIR / "le_b1b2v2_union_cp.csv"
_UNION_IL   = EVAL_DIR / "le_b1b2v2_union_ilastik.csv"
_UNION_ERR_CP = EVAL_DIR / "le_b1b2v2_union_errors_cp.json"
_UNION_ERR_IL = EVAL_DIR / "le_b1b2v2_union_errors_ilastik.json"
_BUDGET_CP_LGBM = EVAL_DIR / "le_b1b2v2_budget_cp_lgbm.csv"
_BUDGET_IL_LGBM = EVAL_DIR / "le_b1b2v2_budget_ilastik_lgbm.csv"
_BUDGET_CP_LR   = EVAL_DIR / "le_b1b2v2_budget_cp_logreg.csv"
_BUDGET_IL_LR   = EVAL_DIR / "le_b1b2v2_budget_ilastik_logreg.csv"

BATCHES = ["b1", "b2v2"]
COMBOS  = [("b1","b1"), ("b1","b2v2"), ("b2v2","b1"), ("b2v2","b2v2")]
COMBO_LABELS = {
    ("b1","b1"):       "B1 → B1",
    ("b1","b2v2"):     "B1 → B2v2",
    ("b2v2","b1"):     "B2v2 → B1",
    ("b2v2","b2v2"):   "B2v2 → B2v2",
    ("union","b1"):    "Union → B1",
    ("union","b2v2"):  "Union → B2v2",
}
COMBO_COLORS = {
    ("b1","b1"):       "#1f77b4",
    ("b1","b2v2"):     "#aec7e8",
    ("b2v2","b1"):     "#d62728",
    ("b2v2","b2v2"):   "#ff9896",
    ("union","b1"):    "#2ca02c",
    ("union","b2v2"):  "#98df8a",
}

METRICS  = ["bal_acc","recall_ad","recall_noad","prec_ad","prec_noad"]
MLABELS  = ["Bal. Acc","Recall\n(adhesion)","Recall\n(no-ad)","Prec\n(adhesion)","Prec\n(no-ad)"]
FOLDS    = list(range(5))


# ── helpers ───────────────────────────────────────────────────────────────────

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


def _add_fig_local(slide, fig, left, top, width, height):
    buf = _fig_to_buf(fig)
    slide.shapes.add_picture(buf, left, top, width=width, height=height)


def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def _load_hc(feat: str) -> pd.DataFrame | None:
    csv = DATA / "ae_results" / "features" / ("cellprofiler" if feat=="cp" else "ilastik") / "ds1.csv"
    if not csv.exists(): return None
    df = pd.read_csv(csv)
    return df.set_index("filename")[[c for c in df.columns if c != "filename"]]


# ── slide 1: title ────────────────────────────────────────────────────────────

def _slide_title(prs):
    slide = _blank(prs)
    _txt(slide, "B1 / B2v2 Matched Annotator Comparison",
         Inches(1.0), Inches(2.5), Inches(11.3), Inches(1.2),
         bold=True, size_pt=36, color=C_HEAD, align=PP_ALIGN.CENTER)
    _txt(slide, "Label Efficiency Benchmark  ·  DS1 (vinculin channel, CIO imaging)",
         Inches(1.0), Inches(3.7), Inches(11.3), Inches(0.5),
         size_pt=18, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide, "B1 = Margaret  ·  B2v2 = Annabel (Sep-2026)  ·  CP + ilastik features  ·  LightGBM",
         Inches(1.0), Inches(4.3), Inches(11.3), Inches(0.4),
         size_pt=13, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide, "4 sparse images excluded (≤3 patches each)  ·  393 patches per annotator",
         Inches(1.0), Inches(4.75), Inches(11.3), Inches(0.35),
         size_pt=11, color=C_GREY, align=PP_ALIGN.CENTER)


# ── slide 2: cross-eval experiment design (native PPTX) ──────────────────────

def _slide_cross_eval_setup(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "Cross-Annotator Evaluation — Experiment Design",
        "Two experiments: within-annotator baseline (Exp 1) · cross-annotator generalization (Exp 2)")

    CT = Inches(1.05); CH = Inches(6.15); PAD = Inches(0.12)
    cw = [Inches(3.3), Inches(5.6), Inches(4.2)]
    cl = [Inches(0.18), Inches(0.18)+cw[0]+PAD, Inches(0.18)+cw[0]+cw[1]+2*PAD]

    _panel_bg(slide, cl[0], CT, cw[0], CH)
    y = _setup_col_header(slide, "Data & Setup", cl[0], CT, cw[0]) + Inches(0.1)
    for label, val in [
        ("Dataset",    "DS1 — vinculin channel\n(CIO imaging, 16 FOVs)"),
        ("Patch size", "32 × 32 px"),
        ("B1 labels",  "393 patches  |  adhesion ≈ 43%\n(4 sparse images excluded)"),
        ("B2v2 labels","393 patches  |  adhesion ≈ 37%\n(Annabel, Sep-2026)"),
        ("Features",   "CellProfiler morphology\nilastik pixel filters"),
        ("Classifier", "LightGBM (balanced weights)"),
        ("CV",         "5-fold stratified\nAll labels per fold"),
        ("Metric",     "Balanced accuracy\n= mean(recall_adh, recall_no-adh)"),
    ]:
        y = _setup_kv(slide, label, val, cl[0], y, cw[0])

    _panel_bg(slide, cl[1], CT, cw[1], CH)
    y = _setup_col_header(slide, "Train → Test Configurations", cl[1], CT, cw[1]) + Inches(0.12)
    lw, tw, aw, ew = Inches(1.1), Inches(1.95), Inches(0.35), Inches(1.9)
    sx = cl[1]+Inches(0.08)
    _txt(slide, "Train set", sx+lw, y, tw, Inches(0.22), bold=True, size_pt=9, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide, "Test set",  sx+lw+tw+aw, y, ew, Inches(0.22), bold=True, size_pt=9, color=C_GREY, align=PP_ALIGN.CENTER)
    y += Inches(0.24)
    configs = [
        ("Exp 1a — Within B1",   "B1 train folds (~314)", "B1 test fold (~79)",    RGBColor(0x1f,0x77,0xb4), RGBColor(0xae,0xc7,0xe8)),
        ("Exp 1b — Within B2v2", "B2v2 train folds (~314)","B2v2 test fold (~79)",  RGBColor(0xd6,0x27,0x28), RGBColor(0xff,0x98,0x96)),
        ("Exp 2a — B1→B2v2",     "B1 train folds (~314)", "B2v2 test fold (~79)",  RGBColor(0x1f,0x77,0xb4), RGBColor(0xff,0x98,0x96)),
        ("Exp 2b — B2v2→B1",     "B2v2 train folds (~314)","B1 test fold (~79)",   RGBColor(0xd6,0x27,0x28), RGBColor(0xae,0xc7,0xe8)),
    ]
    rh = Inches(0.95)
    for lbl, tr, te, c_tr, c_te in configs:
        _txt(slide, lbl, sx, y+Inches(0.05), lw, rh-Inches(0.1), bold=True, size_pt=8.5, color=C_HEAD)
        sh = slide.shapes.add_shape(1, sx+lw, y+Inches(0.05), tw, rh-Inches(0.1))
        sh.fill.solid(); sh.fill.fore_color.rgb = c_tr; sh.line.fill.background()
        _txt(slide, tr, sx+lw+Inches(0.05), y+Inches(0.22), tw-Inches(0.1), rh-Inches(0.3), size_pt=8.5, color=C_WHITE, align=PP_ALIGN.CENTER)
        _txt(slide, "→", sx+lw+tw, y+Inches(0.3), aw, Inches(0.35), size_pt=14, color=C_GREY, align=PP_ALIGN.CENTER)
        sh2 = slide.shapes.add_shape(1, sx+lw+tw+aw, y+Inches(0.05), ew, rh-Inches(0.1))
        sh2.fill.solid(); sh2.fill.fore_color.rgb = c_te; sh2.line.fill.background()
        _txt(slide, te, sx+lw+tw+aw+Inches(0.05), y+Inches(0.22), ew-Inches(0.1), rh-Inches(0.3), size_pt=8.5, color=C_HEAD, align=PP_ALIGN.CENTER)
        y += rh + Inches(0.04)

    _panel_bg(slide, cl[2], CT, cw[2], CH)
    y = _setup_col_header(slide, "Rationale & Controls", cl[2], CT, cw[2]) + Inches(0.1)
    for label, val in [
        ("Why cross-evaluate?",
         "If B1→B2v2 > B1→B1, the model\ngeneralizes to easier patches —\n"
         "pointing to patch difficulty,\nnot annotation quality."),
        ("Leakage prevention",
         "Shared patches removed from\ntest set when in other annotator's\ntrain fold. Overlap = 0 verified."),
        ("What Exp 2 answers",
         "B1→B2v2 > B1→B1\n→ B2v2 patches are easier\n\n"
         "B2v2→B1 < B1→B1\n→ B1 patches are harder\n   regardless of model"),
    ]:
        y = _setup_kv(slide, label, val, cl[2], y, cw[2])


# ── slide 3: cross-eval bar charts ───────────────────────────────────────────

def _slide_cross_eval(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "B1 / B2v2 Cross-Evaluation — CellProfiler & ilastik (nball budget)",
        "Train on annotator's own labels → test on own or other annotator's patches (5-fold avg)")

    feat_specs = [("CP", _CROSS_CP), ("ilastik", _CROSS_IL)]
    n_feat = len(feat_specs)
    fig, axes = plt.subplots(n_feat, len(METRICS), figsize=(13.5, 7.5), facecolor="white")

    x = np.arange(len(COMBOS)); w = 0.6
    combo_colors = [COMBO_COLORS[c] for c in COMBOS]
    combo_labels = [COMBO_LABELS[c] for c in COMBOS]

    for row, (feat_name, csv_path) in enumerate(feat_specs):
        if not csv_path.exists():
            for ax in axes[row]:
                ax.text(0.5, 0.5, "pending", ha="center", va="center", transform=ax.transAxes, color="grey")
                ax.axis("off")
            continue
        df = pd.read_csv(csv_path)
        for col, (metric, mlabel) in enumerate(zip(METRICS, MLABELS)):
            ax = axes[row, col]
            vals = [df[(df.train_batch==tr)&(df.test_batch==te)][metric].mean()
                    for tr, te in COMBOS]
            stds = [df[(df.train_batch==tr)&(df.test_batch==te)][metric].std()
                    for tr, te in COMBOS]
            bars = ax.bar(x, vals, w, color=combo_colors, alpha=0.85, zorder=2)
            ax.errorbar(x, vals, stds, fmt="none", color="#333", capsize=3, lw=1.2, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([l.replace(" → ","\n→\n") for l in combo_labels],
                               fontsize=7, rotation=0)
            ax.set_ylim(0.45, 1.02)
            ax.set_ylabel("Score" if col == 0 else "", fontsize=8)
            if row == 0: ax.set_title(mlabel, fontsize=9, fontweight="bold", pad=4)
            if col == 0: ax.text(-0.22, 0.5, feat_name, transform=ax.transAxes,
                                 fontsize=10, fontweight="bold", rotation=90,
                                 va="center", ha="right", color="#16213e")
            ax.axhline(0.5, color="#aaa", lw=0.7, ls="--", zorder=0)
            ax.set_facecolor("white"); ax.grid(axis="y", color="#eee", lw=0.7)
            ax.spines[["top","right"]].set_visible(False)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.3f}",
                        ha="center", va="bottom", fontsize=6.5)

    fig.suptitle("B1→B2v2 > B1→B1 confirms patch difficulty drives the gap", fontsize=10, y=1.01)
    fig.tight_layout(h_pad=0.5, w_pad=0.4)
    _add_fig_local(slide, fig, PLOT_LEFT, Inches(1.05), PLOT_W, Inches(6.1))


# ── slides 4–5: error patches ────────────────────────────────────────────────

def _slide_error_patches(prs, cross: bool = False):
    if cross:
        combos_show = [("b1","b2v2"), ("b2v2","b1")]
        title    = "Cross-Evaluation Error Patches (B1→B2v2 test, B2v2→B1 test)"
        subtitle = "CellProfiler features, nball budget, fold 0 representative — top: ad→no-ad, bottom: no-ad→ad"
        err_file = _CROSS_ERR_CP
    else:
        combos_show = [("b1","b1"), ("b2v2","b2v2")]
        title    = "Within-Annotator Error Patches (B1→B1 test, B2v2→B2v2 test)"
        subtitle = "CellProfiler features, nball budget, fold 0 representative — top: ad→no-ad, bottom: no-ad→ad"
        err_file = _CROSS_ERR_CP

    slide = _blank(prs)
    _slide_header(slide, title, subtitle)

    if not err_file.exists():
        _txt(slide, "Error JSON not found — run eval_le_b1b2v2_cross.py",
             Inches(1), Inches(2), Inches(11), Inches(1), size_pt=14, color=C_GREY)
        return

    with open(err_file) as f:
        errs = json.load(f)

    n_cols = len(combos_show); col_w = Inches(13.0 / n_cols)
    top_base = Inches(1.15); row_h = Inches(2.9)

    for col_i, (tb, te) in enumerate(combos_show):
        key = f"{tb}_to_{te}"
        fold_data = errs.get(key, {}).get("0", errs.get(key, {}).get(0, {}))
        ad2noad = fold_data.get("ad2noad", [])
        noad2ad = fold_data.get("noad2ad", [])
        left = Inches(0.2) + col_i * col_w
        _txt(slide, COMBO_LABELS[(tb,te)], left, top_base, col_w, Inches(0.3),
             bold=True, size_pt=12, color=C_HEAD)

        for row_i, (patch_fns, err_label, hex_c) in enumerate([
            (ad2noad, f"ad → no-ad  ({len(ad2noad)} errors)", "#d62728"),
            (noad2ad, f"no-ad → ad  ({len(noad2ad)} errors)", "#1f77b4"),
        ]):
            row_top = top_base + Inches(0.35) + row_i * row_h
            _txt(slide, err_label, left, row_top, col_w, Inches(0.25),
                 size_pt=10, color=RGBColor(int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)))
            grid = _load_error_patches_grid(patch_fns, PATCH_DIR, max_patches=20, ncols=10)
            if grid is None:
                _txt(slide, "(no errors in fold 0)", left, row_top+Inches(0.3),
                     col_w, Inches(0.3), size_pt=10, color=C_GREY)
                continue
            fig, ax = plt.subplots(figsize=(5.5, 1.5), facecolor="white")
            ax.imshow(grid, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off"); fig.tight_layout(pad=0)
            _add_fig_local(slide, fig, left, row_top+Inches(0.28), col_w-Inches(0.1), row_h-Inches(0.35))


# ── slide 6: union/budget experiment design (native PPTX) ────────────────────

def _slide_union_budget_setup(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "Experiment Design — Union Training & Label Budget Study",
        "Exp 3: union training (B1∪B2v2) · Exp 4: mixed-label budget study (CP + ilastik)")

    CT = Inches(1.05); CH = Inches(6.15); PAD = Inches(0.12)
    cw = [Inches(3.1), Inches(5.8), Inches(4.1)]
    cl = [Inches(0.18), Inches(0.18)+cw[0]+PAD, Inches(0.18)+cw[0]+cw[1]+2*PAD]

    _panel_bg(slide, cl[0], CT, cw[0], CH)
    y = _setup_col_header(slide, "Data & Setup", cl[0], CT, cw[0]) + Inches(0.1)
    for label, val in [
        ("Dataset",    "DS1 — vinculin channel"),
        ("B1 labels",  "393 patches  |  adhesion ≈ 43%"),
        ("B2v2 labels","393 patches  |  adhesion ≈ 37%"),
        ("Shared",     "Overlap excluded from test if\nit appears in other's train fold"),
        ("Features",   "CellProfiler morphology\nilastik pixel filters"),
        ("Classifier", "LightGBM (balanced weights)"),
        ("CV",         "5-fold stratified\nBudget study: 10 repeats/fold"),
        ("Metric",     "Balanced accuracy\n= mean(recall_adh, recall_no-adh)"),
    ]:
        y = _setup_kv(slide, label, val, cl[0], y, cw[0])

    _panel_bg(slide, cl[1], CT, cw[1], CH)
    y = _setup_col_header(slide, "Train → Test Matrix", cl[1], CT, cw[1]) + Inches(0.12)
    lw, tw, aw, ew = Inches(1.3), Inches(2.0), Inches(0.35), Inches(1.75)
    sx = cl[1]+Inches(0.08)
    _txt(slide, "Train set", sx+lw, y, tw, Inches(0.22), bold=True, size_pt=9, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide, "Test set",  sx+lw+tw+aw, y, ew, Inches(0.22), bold=True, size_pt=9, color=C_GREY, align=PP_ALIGN.CENTER)
    y += Inches(0.24)
    configs = [
        ("Exp 3 — Union\ntraining",
         "B1∪B2v2 train folds\n(~628, 2 conflicts out)",
         "B1 test\n+ B2v2 test",
         RGBColor(0x2c,0xa0,0x2c), RGBColor(0x98,0xdf,0x8a)),
        ("Exp 4a — B1-only\nbudget",
         "B1 labels only\nN = 20, 40, …, 340",
         "B1 test\n+ B2v2 test",
         RGBColor(0x1f,0x77,0xb4), RGBColor(0xae,0xc7,0xe8)),
        ("Exp 4b — B2v2-only\nbudget",
         "B2v2 labels only\nN = 20, 40, …, 340",
         "B1 test\n+ B2v2 test",
         RGBColor(0xd6,0x27,0x28), RGBColor(0xff,0x98,0x96)),
        ("Exp 4c — Mixed\nbudget (50:50)",
         "B1+B2v2 labels\nN = 20, …, 800, all",
         "B1 test\n+ B2v2 test",
         RGBColor(0xe0,0x7b,0x39), RGBColor(0xff,0xd8,0xb0)),
    ]
    rh = Inches(1.18)
    for lbl, tr, te, c_tr, c_te in configs:
        _txt(slide, lbl, sx, y+Inches(0.05), lw, rh-Inches(0.1), bold=True, size_pt=8.5, color=C_HEAD)
        sh = slide.shapes.add_shape(1, sx+lw, y+Inches(0.05), tw, rh-Inches(0.1))
        sh.fill.solid(); sh.fill.fore_color.rgb = c_tr; sh.line.fill.background()
        _txt(slide, tr, sx+lw+Inches(0.05), y+Inches(0.28), tw-Inches(0.1), rh-Inches(0.4), size_pt=8.5, color=C_WHITE, align=PP_ALIGN.CENTER)
        _txt(slide, "→", sx+lw+tw, y+Inches(0.42), aw, Inches(0.35), size_pt=14, color=C_GREY, align=PP_ALIGN.CENTER)
        sh2 = slide.shapes.add_shape(1, sx+lw+tw+aw, y+Inches(0.05), ew, rh-Inches(0.1))
        sh2.fill.solid(); sh2.fill.fore_color.rgb = c_te; sh2.line.fill.background()
        _txt(slide, te, sx+lw+tw+aw+Inches(0.05), y+Inches(0.28), ew-Inches(0.1), rh-Inches(0.4), size_pt=8.5, color=C_HEAD, align=PP_ALIGN.CENTER)
        y += rh + Inches(0.05)

    _panel_bg(slide, cl[2], CT, cw[2], CH)
    y = _setup_col_header(slide, "Controls & Budget Grid", cl[2], CT, cw[2]) + Inches(0.1)
    for label, val in [
        ("Leakage prevention",
         "Test patches in other annotator's\ntrain fold removed before scoring.\nVerified: 0 overlap per fold."),
        ("Union training budget",
         "nball: ~314 B1 + ~314 B2v2\n≈ 628 patches per fold.\n~2 conflict patches excluded."),
        ("Budget study grid",
         "B1-only / B2v2-only:\n  N = 20, 40, …, 340\nMixed (50:50):\n  N = 20, 40, …, 340,\n  380, …, 800, all (~628)"),
        ("Sampling",
         "Stratified random: class ratio\nmaintained, ≥1 per class.\n10 repeats per fold."),
    ]:
        y = _setup_kv(slide, label, val, cl[2], y, cw[2])


# ── slide 7: union eval ───────────────────────────────────────────────────────

def _slide_union_eval(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "B1 / B2v2 Union Training Evaluation — CellProfiler & ilastik",
        "Train on B1∪B2v2 (conflict-excluded, ~628 patches/fold) → test on B1 or B2v2 test patches (5-fold avg)")

    all_combos = COMBOS + [("union","b1"), ("union","b2v2")]
    all_colors  = [COMBO_COLORS[c] for c in all_combos]
    all_labels  = [COMBO_LABELS[c] for c in all_combos]
    feat_specs  = [("CP", _CROSS_CP, _UNION_CP), ("ilastik", _CROSS_IL, _UNION_IL)]

    x = np.arange(len(all_combos)); w = 0.6
    fig, axes = plt.subplots(len(feat_specs), len(METRICS), figsize=(13.5, 7.5), facecolor="white")

    for row, (feat_name, cross_csv, union_csv) in enumerate(feat_specs):
        df_cross = pd.read_csv(cross_csv) if cross_csv.exists() else pd.DataFrame()
        df_union = pd.read_csv(union_csv) if union_csv.exists() else pd.DataFrame()

        for col, (metric, mlabel) in enumerate(zip(METRICS, MLABELS)):
            ax = axes[row, col]
            vals, stds = [], []
            for tb, te in all_combos:
                if tb == "union":
                    sub = df_union[df_union.test_batch == te] if not df_union.empty else pd.DataFrame()
                else:
                    sub = df_cross[(df_cross.train_batch==tb)&(df_cross.test_batch==te)] if not df_cross.empty else pd.DataFrame()
                vals.append(sub[metric].mean() if not sub.empty else np.nan)
                stds.append(sub[metric].std()  if not sub.empty else np.nan)

            bars = ax.bar(x, vals, w, color=all_colors, alpha=0.85, zorder=2)
            ax.errorbar(x, vals, stds, fmt="none", color="#333", capsize=3, lw=1.2, zorder=3)
            ax.set_xticks(x)
            ax.set_xticklabels([l.replace(" → ","\n→\n") for l in all_labels], fontsize=6.5)
            ax.set_ylim(0.45, 1.05)
            ax.set_ylabel("Score" if col == 0 else "", fontsize=8)
            if row == 0: ax.set_title(mlabel, fontsize=9, fontweight="bold", pad=4)
            if col == 0: ax.text(-0.22, 0.5, feat_name, transform=ax.transAxes,
                                 fontsize=10, fontweight="bold", rotation=90,
                                 va="center", ha="right", color="#16213e")
            ax.axhline(0.5, color="#aaa", lw=0.7, ls="--", zorder=0)
            ax.set_facecolor("white"); ax.grid(axis="y", color="#eee", lw=0.7)
            ax.spines[["top","right"]].set_visible(False)
            for bar, val in zip(bars, vals):
                if not np.isnan(val):
                    ax.text(bar.get_x()+bar.get_width()/2, val+0.01, f"{val:.3f}",
                            ha="center", va="bottom", fontsize=6.0)

    fig.suptitle("Union training improves B1 test and matches/exceeds B2v2-only on B2v2 test", fontsize=10, y=1.01)
    fig.tight_layout(h_pad=0.5, w_pad=0.4)
    _add_fig_local(slide, fig, PLOT_LEFT, Inches(1.05), PLOT_W, Inches(6.1))


# ── slide 8: union error patches ─────────────────────────────────────────────

def _slide_union_error_patches(prs):
    slide = _blank(prs)
    _slide_header(slide,
        "Union Training Error Patches (Union→B1 test, Union→B2v2 test)",
        "CellProfiler features, fold 0 representative — top: ad→no-ad, bottom: no-ad→ad")

    combos_show = [("union","b1"), ("union","b2v2")]
    err_file    = _UNION_ERR_CP

    if not err_file.exists():
        _txt(slide, "Union error JSON not found — run eval_le_b1b2v2_union.py",
             Inches(1), Inches(2), Inches(11), Inches(1), size_pt=14, color=C_GREY)
        return

    with open(err_file) as f:
        errs = json.load(f)

    n_cols = 2; col_w = Inches(13.0 / n_cols)
    top_base = Inches(1.15); row_h = Inches(2.9)

    for col_i, (tb, te) in enumerate(combos_show):
        fold_data = errs.get(te, {}).get("0", errs.get(te, {}).get(0, {}))
        ad2noad = fold_data.get("ad2noad", [])
        noad2ad = fold_data.get("noad2ad", [])
        left = Inches(0.2) + col_i * col_w
        _txt(slide, COMBO_LABELS[(tb,te)], left, top_base, col_w, Inches(0.3),
             bold=True, size_pt=12, color=C_HEAD)

        for row_i, (patch_fns, err_label, hex_c) in enumerate([
            (ad2noad, f"ad → no-ad  ({len(ad2noad)} errors)", "#d62728"),
            (noad2ad, f"no-ad → ad  ({len(noad2ad)} errors)", "#1f77b4"),
        ]):
            row_top = top_base + Inches(0.35) + row_i * row_h
            _txt(slide, err_label, left, row_top, col_w, Inches(0.25),
                 size_pt=10, color=RGBColor(int(hex_c[1:3],16), int(hex_c[3:5],16), int(hex_c[5:7],16)))
            grid = _load_error_patches_grid(patch_fns, PATCH_DIR, max_patches=20, ncols=10)
            if grid is None:
                _txt(slide, "(no errors in fold 0)", left, row_top+Inches(0.3),
                     col_w, Inches(0.3), size_pt=10, color=C_GREY)
                continue
            fig, ax = plt.subplots(figsize=(5.5, 1.5), facecolor="white")
            ax.imshow(grid, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.axis("off"); fig.tight_layout(pad=0)
            _add_fig_local(slide, fig, left, row_top+Inches(0.28), col_w-Inches(0.1), row_h-Inches(0.35))


# ── slide 9: budget curves ────────────────────────────────────────────────────

def _slide_budget(prs, clf: str = "lgbm"):
    clf_label = "LightGBM" if clf == "lgbm" else "Logistic Regression"
    slide = _blank(prs)
    _slide_header(slide,
        f"Mixed-Label Budget Study — CellProfiler & ilastik  ({clf_label})",
        "B1-only vs B2v2-only vs 50:50 mixed · tested on B1 test and B2v2 test · mean ± std (5 folds × 10 repeats)")

    STRAT_COLORS = {"b1_only": "#e07b39", "b2v2_only": "#2e86ab", "mixed": "#2ca02c"}
    STRAT_LABELS = {"b1_only": "B1-only", "b2v2_only": "B2v2-only", "mixed": "Mixed"}
    SINGLE_MAX   = 340

    csv_map = {
        ("cp",      "lgbm"): _BUDGET_CP_LGBM,
        ("ilastik", "lgbm"): _BUDGET_IL_LGBM,
        ("cp",      "logreg"): _BUDGET_CP_LR,
        ("ilastik", "logreg"): _BUDGET_IL_LR,
    }

    panels = [("CellProfiler", "b1"), ("CellProfiler", "b2v2"),
              ("ilastik",      "b1"), ("ilastik",      "b2v2")]
    feat_key = {"CellProfiler": "cp", "ilastik": "ilastik"}

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 6.5), facecolor="white")
    axes_flat = axes.flatten()

    for ax_i, (feat_name, test_b) in enumerate(panels):
        ax = axes_flat[ax_i]
        feat = feat_key[feat_name]
        csv_path = csv_map[(feat, clf)]

        ax.set_title(f"{feat_name}  →  Test on {'B1' if test_b=='b1' else 'B2v2'}",
                     fontsize=10, fontweight="bold", pad=5)

        if not csv_path.exists():
            ax.text(0.5, 0.5, "pending", ha="center", va="center",
                    transform=ax.transAxes, color="grey")
            ax.axis("off"); continue

        df = pd.read_csv(csv_path)
        df_te = df[df.test_batch == test_b]

        # numeric budgets
        num_bs = sorted(b for b in df_te.budget.unique() if b != 0)
        step   = num_bs[-1] - num_bs[-2] if len(num_bs) > 1 else 20
        all_pos = num_bs[-1] + step  # x position for "all"

        for strat in ["b1_only", "b2v2_only", "mixed"]:
            sub = df_te[df_te.strategy == strat]
            if sub.empty: continue
            g = sub.groupby("budget")["bal_acc"]
            means = g.mean(); stds = g.std()
            num_sub = {b: v for b, v in means.items() if b != 0}
            all_sub = {b: v for b, v in means.items() if b == 0}
            num_std = {b: v for b, v in stds.items()  if b != 0}

            if num_sub:
                xs = sorted(num_sub)
                ys = [num_sub[b] for b in xs]
                es = [num_std.get(b, 0) for b in xs]
                ax.plot(xs, ys, color=STRAT_COLORS[strat], lw=2.0,
                        marker="o", markersize=4, label=STRAT_LABELS[strat])
                ax.fill_between(xs, [y-e for y,e in zip(ys,es)],
                                    [y+e for y,e in zip(ys,es)],
                                color=STRAT_COLORS[strat], alpha=0.12)

            if all_sub:
                av = list(all_sub.values())[0]
                ae = list({b: v for b, v in stds.items() if b == 0}.values())[0] if 0 in stds else 0
                ax.scatter([all_pos], [av], marker="*", s=110,
                           color=STRAT_COLORS[strat], zorder=5)
                ax.errorbar([all_pos], [av], [[ae],[ae]], fmt="none",
                            color=STRAT_COLORS[strat], capsize=3, lw=1.2)

        ax.axvline(SINGLE_MAX + step/2, color="#aaa", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("Label budget (N)", fontsize=9)
        ax.set_ylabel("Balanced accuracy", fontsize=9)
        ax.set_xticks(sorted(num_bs[::2]) + [all_pos])
        ax.set_xticklabels([str(b) for b in num_bs[::2]] + ["all"], fontsize=8, rotation=30)
        ax.set_ylim(0.42, 1.02)
        ax.legend(fontsize=8, framealpha=0.9, loc="lower right")
        ax.set_facecolor("white"); ax.grid(axis="y", color="#EEEEEE", lw=0.7)
        ax.spines[["top","right"]].set_visible(False)

    fig.tight_layout(h_pad=0.8, w_pad=0.8)
    _add_fig_local(slide, fig, Inches(0.2), Inches(1.0), Inches(13.0), Inches(6.2))


# ── slides 10–11: budget tables ───────────────────────────────────────────────

def _slide_budget_table(prs, feat: str, clf: str = "lgbm"):
    feat_label = "CellProfiler" if feat == "cp" else "ilastik"
    clf_label  = "LightGBM" if clf == "lgbm" else "Logistic Regression"
    slide = _blank(prs)
    _slide_header(slide,
        f"B1/B2v2 Budget Study — {feat_label} ({clf_label}) — Balanced Accuracy",
        "mean ± std (%) across 5 folds × 10 repeats  ·  'all' = full pool (~314/annotator, ~628 mixed)  ·  '—' = exceeds single-annotator pool")

    csv_path = (_BUDGET_CP_LGBM if feat=="cp" else _BUDGET_IL_LGBM) if clf=="lgbm" else \
               (_BUDGET_CP_LR   if feat=="cp" else _BUDGET_IL_LR)

    if not csv_path.exists():
        _txt(slide, "Results pending", Inches(1), Inches(2), Inches(11), Inches(1),
             size_pt=14, color=C_GREY)
        return

    df = pd.read_csv(csv_path)
    SINGLE_MAX = 340

    all_bs_raw = sorted(df.budget.unique())
    num_bs = sorted(b for b in all_bs_raw if b != 0)
    budgets_ordered = num_bs + ([0] if 0 in all_bs_raw else [])

    def fmt(m, s):
        return "—" if np.isnan(m) else f"{m*100:.1f}±{s*100:.1f}"

    def get_stats(tb, strat, b):
        sub = df[(df.test_batch==tb)&(df.strategy==strat)&(df.budget==b)]
        return (sub.bal_acc.mean(), sub.bal_acc.std()) if not sub.empty else (np.nan, np.nan)

    col_headers = (["Budget"] +
                   [f"B1-test\n{s}" for s in ["B1-only","B2v2-only","Mixed"]] +
                   [f"B2v2-test\n{s}" for s in ["B1-only","B2v2-only","Mixed"]])
    rows_data = []
    for b in budgets_ordered:
        row = ["all" if b==0 else str(b)]
        for tb in ["b1","b2v2"]:
            for strat in ["b1_only","b2v2_only","mixed"]:
                if strat != "mixed" and b != 0 and b > SINGLE_MAX:
                    row.append("—")
                else:
                    m, s = get_stats(tb, strat, b)
                    row.append(fmt(m, s))
        rows_data.append(row)

    n_rows = len(rows_data); n_cols = len(col_headers)
    fig, ax = plt.subplots(figsize=(13.0, max(4.5, n_rows*0.30+0.8)), facecolor="white")
    ax.axis("off")
    tbl = ax.table(cellText=rows_data, colLabels=col_headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1, 1.35)
    for col in range(n_cols):
        cell = tbl[0, col]
        cell.set_facecolor("#162130"); cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold"); cell.get_text().set_fontsize(8)
    col_bg = {1:"#dce8f5", 2:"#dce8f5", 3:"#d4f0d4", 4:"#fde8e8", 5:"#fde8e8", 6:"#d4f0d4"}
    for ri in range(n_rows):
        for ci in range(n_cols):
            cell = tbl[ri+1, ci]
            base = col_bg.get(ci, "#f5f5f5")
            if ri % 2 == 1:
                import colorsys
                r, g, b = int(base[1:3],16)/255, int(base[3:5],16)/255, int(base[5:7],16)/255
                h, s_h, v = colorsys.rgb_to_hsv(r, g, b)
                r2, g2, b2 = colorsys.hsv_to_rgb(h, s_h, v*0.93)
                base = f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"
            cell.set_facecolor(base)
            if rows_data[ri][0] == "all":
                cell.get_text().set_fontweight("bold")
        tbl[ri+1, 0].set_facecolor("#eeeeee")
    fig.tight_layout(pad=0.2)
    _add_fig_local(slide, fig, Inches(0.15), Inches(1.0), Inches(13.0), Inches(6.3))


# ── build ─────────────────────────────────────────────────────────────────────

def build_pptx_b1b2v2(out_path: Path):
    print("Building B1/B2v2 matched PPT …")
    prs = _prs()

    print("  Slide 1 — Title")
    _slide_title(prs)
    print("  Slide 2 — Cross-annotator evaluation: experiment design")
    _slide_cross_eval_setup(prs)
    print("  Slide 3 — Cross-evaluation metrics (CP + ilastik)")
    _slide_cross_eval(prs)
    print("  Slide 4 — Error patches (within: B1→B1, B2v2→B2v2)")
    _slide_error_patches(prs, cross=False)
    print("  Slide 5 — Error patches (cross: B1→B2v2, B2v2→B1)")
    _slide_error_patches(prs, cross=True)
    print("  Slide 6 — Union/budget experiment design")
    _slide_union_budget_setup(prs)
    print("  Slide 7 — Union training evaluation")
    _slide_union_eval(prs)
    print("  Slide 8 — Union error patches")
    _slide_union_error_patches(prs)
    print("  Slide 9 — Budget study curves (LightGBM)")
    _slide_budget(prs, clf="lgbm")
    print("  Slide 10 — Budget accuracy table (LightGBM, CellProfiler)")
    _slide_budget_table(prs, feat="cp", clf="lgbm")
    print("  Slide 11 — Budget accuracy table (LightGBM, ilastik)")
    _slide_budget_table(prs, feat="ilastik", clf="lgbm")
    if _BUDGET_CP_LR.exists() or _BUDGET_IL_LR.exists():
        print("  Slide 12 — Budget study curves (Logistic Regression)")
        _slide_budget(prs, clf="logreg")
        print("  Slide 13 — Budget accuracy table (LogReg, CellProfiler)")
        _slide_budget_table(prs, feat="cp", clf="logreg")
        print("  Slide 14 — Budget accuracy table (LogReg, ilastik)")
        _slide_budget_table(prs, feat="ilastik", clf="logreg")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    print(f"\nSaved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "le_b1b2v2_matched.pptx"))
    args = ap.parse_args()
    build_pptx_b1b2v2(Path(args.out))


if __name__ == "__main__":
    main()

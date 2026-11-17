#!/usr/bin/env python3
"""
make_pptx_cp_combo.py

Comprehensive feature-combination study deck for DS1 focal adhesion classification.
22 combos tested across Label-Efficiency (LE) and Image-Efficiency (IE) experiments.

Feature groups:
  Baselines      : cp_s32, cp_s128, dist, il_s32, supcon_z
  Scale (ch1)    : cp_s16+cp_s32, cp_s32+s128, cp_s32+cp_s64
  Zyxin (ch2)    : cp_s32 + cp_s{16,32,64,128}_ch2  (4 combos)
  Actin (ch3)    : cp_s32 + cp_s{16,32,64,128}_ch3  (4 combos)
  Multi-source   : cp_s32+supcon_z, cp_s32+il_s32, supcon_z+il_s32,
                   cp_s32+supcon_z+il_s32, cp_s32+dist, cp_s32+s128+dist

Slides:
  1  — Title
  2  — Feature sources overview
  3  — LE experiment design
  4  — IE experiment design
  5  — LE: ranked bar chart (all 22, B1 & B2v2)
  6  — LE: budget curves — baseline + scale combos (lgbm)
  7  — LE: budget curves — cross-channel + multi-source combos (lgbm)
  8  — LE: all-budget summary table (lgbm)
  9  — LE: all-budget summary table (logreg)
  10 — IE: LOIO ranking bar chart (all 22, lgbm)
  11 — IE: LOIO per held-out image (top 8 combos, lgbm)
  12 — IE: image budget curves (top 6 combos, lgbm)
  13 — IE: LOIO label-budget effect (key combos, lgbm)
  14 — Summary: top combos & key findings

Usage:
  python scripts/make_pptx_cp_combo.py [--out PATH]
"""
from __future__ import annotations
import argparse, io, sys, warnings, colorsys
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_pptx_le_benchmark import (
    _blank, _txt, _slide_header, _rule, _panel_bg,
    _setup_col_header, _setup_kv,
    C_HEAD, C_BODY, C_GREY, C_WHITE, C_PANEL,
    SW, SH, PP_ALIGN, RGBColor, Inches, Pt,
)
from pptx import Presentation

# ---------------------------------------------------------------------------
# Paths

REPO       = Path(__file__).resolve().parents[1]
EVAL_DIR   = REPO / "eval_results"
B1B2V2_DIR = EVAL_DIR / "b1b2v2"
LE_CSVS    = {
    "lgbm":   B1B2V2_DIR / "le_b1b2v2_budget_cp_combo_lgbm.csv",
    "logreg": B1B2V2_DIR / "le_b1b2v2_budget_cp_combo_logreg.csv",
}
IE_CSVS    = {
    "lgbm":   EVAL_DIR / "image_efficiency_cp_combo_lgbm.csv",
    "logreg": EVAL_DIR / "image_efficiency_cp_combo_logreg.csv",
}
OUT_DIR    = REPO / "results"

# ---------------------------------------------------------------------------
# Feature combo groups & colours

GROUPS = {
    "Baselines":       ["cp_s32", "cp_s128", "dist", "il_s32", "supcon_z"],
    "Scale ch1":       ["cp_s16+cp_s32", "cp_s32+s128", "cp_s32+cp_s64"],
    "Zyxin ch2":       ["cp_s32+cp_s16_ch2", "cp_s32+cp_s32_ch2",
                        "cp_s32+cp_s64_ch2", "cp_s32+cp_s128_ch2"],
    "Actin ch3":       ["cp_s32+cp_s16_ch3", "cp_s32+cp32ch3",
                        "cp_s32+cp_s64_ch3", "cp_s32+cp_s128_ch3"],
    "Multi-source":    ["cp_s32+dist", "cp_s32+s128+dist",
                        "cp_s32+supcon_z", "cp_s32+il_s32",
                        "supcon_z+il_s32", "cp_s32+supcon_z+il_s32"],
}

# Colour per group (matplotlib hex)
GROUP_PALETTE = {
    "Baselines":    ["#555555", "#aaaaaa", "#d62728", "#17becf", "#9467bd"],
    "Scale ch1":    ["#1f77b4", "#0a4fa0", "#6baed6"],
    "Zyxin ch2":    ["#ff7f0e", "#e05c00", "#ffab6b", "#c44d00"],
    "Actin ch3":    ["#2ca02c", "#1a6b1a", "#74c476", "#105910"],
    "Multi-source": ["#e377c2", "#8c564b", "#9467bd", "#17becf",
                     "#bcbd22", "#7f7f7f"],
}

def _combo_color(feat: str) -> str:
    for grp, combos in GROUPS.items():
        if feat in combos:
            idx = combos.index(feat)
            return GROUP_PALETTE[grp][idx % len(GROUP_PALETTE[grp])]
    return "#333333"

ALL_COMBOS = [f for combos in GROUPS.values() for f in combos]

CLF_LABELS = {"lgbm": "LightGBM", "logreg": "Logistic Regression"}
ANN_LABELS = {"b1": "B1 (Margaret)", "b2v2": "B2v2 (Annabel)"}
STRAT_COLORS = {"b1_only": "#1f77b4", "b2v2_only": "#d62728", "mixed": "#2ca02c"}
STRAT_LABELS = {"b1_only": "B1-only", "b2v2_only": "B2v2-only", "mixed": "Mixed"}
SINGLE_MAX = 340

PLOT_LEFT = Inches(0.2)
PLOT_TOP  = Inches(1.05)
PLOT_W    = Inches(12.9)
PLOT_H    = Inches(6.15)

# ---------------------------------------------------------------------------
# Helpers

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
    slide.shapes.add_picture(_fig_to_buf(fig), left, top, width=width, height=height)

def _load_le(clf):
    p = LE_CSVS[clf]
    return pd.read_csv(p) if p.exists() else None

def _load_ie(clf):
    p = IE_CSVS[clf]
    return pd.read_csv(p) if p.exists() else None

def _shade(base_hex, factor):
    r, g, b = int(base_hex[1:3],16)/255, int(base_hex[3:5],16)/255, int(base_hex[5:7],16)/255
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    r2,g2,b2 = colorsys.hsv_to_rgb(h, s, v*factor)
    return f"#{int(r2*255):02x}{int(g2*255):02x}{int(b2*255):02x}"

# ---------------------------------------------------------------------------
# Slide 1: Title

def _slide_title(prs):
    slide = _blank(prs)
    _txt(slide, "Feature Combination Study",
         Inches(0.8), Inches(1.6), Inches(11.7), Inches(1.0),
         bold=True, size_pt=36, color=C_HEAD, align=PP_ALIGN.CENTER)
    _txt(slide, "CP Scale · Cross-Channel · ilastik · SupCon Embeddings",
         Inches(0.8), Inches(2.7), Inches(11.7), Inches(0.55),
         size_pt=20, color=C_GREY, align=PP_ALIGN.CENTER)
    _rule(slide, Inches(3.85), width=Inches(6.6), left=Inches(3.45))
    _txt(slide, "DS1 · Vinculin CIO imaging · 22 feature combos",
         Inches(0.8), Inches(3.4), Inches(11.7), Inches(0.4),
         size_pt=14, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide, "B1 (Margaret) · B2v2 (Annabel Sep-2026) · LightGBM & LogReg",
         Inches(0.8), Inches(3.85), Inches(11.7), Inches(0.38),
         size_pt=13, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide,
         "Exp A: Label-efficiency budget study (5-fold CV, 3 strategies, budget 20→all)\n"
         "Exp B: Image-efficiency test (LOIO + image budget sweep k=1→19)",
         Inches(0.8), Inches(4.3), Inches(11.7), Inches(0.65),
         size_pt=12, color=RGBColor(0x44,0x44,0x88), align=PP_ALIGN.CENTER)


# ---------------------------------------------------------------------------
# Slide 2: Feature sources overview

def _slide_features(prs):
    slide = _blank(prs)
    _slide_header(slide, "Feature Sources — 22 Combinations Tested",
                  "All combos use cp_s32 (paxillin 32px) as base or standalone · DS1 only (vinculin CIO)")

    CT = Inches(1.05); CH = Inches(6.15); PAD = Inches(0.10)
    cw = [Inches(2.45)] * 5
    cl = [Inches(0.18) + i*(cw[0]+PAD) for i in range(5)]

    groups_info = [
        ("Baselines", GROUP_PALETTE["Baselines"][0],
         [("cp_s32",    "CellProfiler\n32×32 px paxillin\n50 features"),
          ("cp_s128",   "CellProfiler\n128×128 px paxillin\n50 features"),
          ("dist",      "Distance to\ncell boundary\n8-ray, 9 features"),
          ("il_s32",    "ilastik filters\n32×32 px\n80 features"),
          ("supcon_z",  "SupCon AE\nencoder latent\n32-dim (avg 3 seeds)")]),
        ("Scale ch1\n(paxillin)", GROUP_PALETTE["Scale ch1"][0],
         [("cp_s16+cp_s32",  "16px + 32px\npaxillin\n100 features"),
          ("cp_s32+s128",    "32px + 128px\npaxillin\n100 features"),
          ("cp_s32+cp_s64",  "32px + 64px\npaxillin\n100 features")]),
        ("Zyxin ch2\n(cross-ch)", GROUP_PALETTE["Zyxin ch2"][0],
         [("cp_s32+cp_s16_ch2",  "pax-32 +\nzyx-16\n100 features"),
          ("cp_s32+cp_s32_ch2",  "pax-32 +\nzyx-32\n100 features"),
          ("cp_s32+cp_s64_ch2",  "pax-32 +\nzyx-64\n100 features"),
          ("cp_s32+cp_s128_ch2", "pax-32 +\nzyx-128\n100 features")]),
        ("Actin ch3\n(cross-ch)", GROUP_PALETTE["Actin ch3"][0],
         [("cp_s32+cp_s16_ch3",  "pax-32 +\nact-16\n100 features"),
          ("cp_s32+cp32ch3",     "pax-32 +\nact-32\n100 features"),
          ("cp_s32+cp_s64_ch3",  "pax-32 +\nact-64\n100 features"),
          ("cp_s32+cp_s128_ch3", "pax-32 +\nact-128\n100 features")]),
        ("Multi-source", GROUP_PALETTE["Multi-source"][0],
         [("cp_s32+dist",           "pax-32 + dist\n59 features"),
          ("cp_s32+s128+dist",      "pax-32+128 + dist\n109 features"),
          ("cp_s32+supcon_z",       "pax-32 +\nSupCon-z\n82 features"),
          ("cp_s32+il_s32",         "pax-32 +\nilastik\n130 features"),
          ("supcon_z+il_s32",       "SupCon-z +\nilastik\n112 features"),
          ("cp_s32+supcon_z+il_s32","pax-32 + SupCon\n+ ilastik\n162 features")]),
    ]

    for col_i, (grp_name, grp_color, items) in enumerate(groups_info):
        rgb = RGBColor(int(grp_color[1:3],16), int(grp_color[3:5],16), int(grp_color[5:7],16))
        _panel_bg(slide, cl[col_i], CT, cw[col_i], CH)
        y = _setup_col_header(slide, grp_name, cl[col_i], CT, cw[col_i]) + Inches(0.08)
        for feat_name, desc in items:
            color_hex = _combo_color(feat_name)
            sh = slide.shapes.add_shape(1, cl[col_i]+Inches(0.06), y, Inches(0.08), Inches(0.18))
            sh.fill.solid()
            sh.fill.fore_color.rgb = RGBColor(int(color_hex[1:3],16), int(color_hex[3:5],16), int(color_hex[5:7],16))
            sh.line.fill.background()
            _txt(slide, feat_name, cl[col_i]+Inches(0.18), y, cw[col_i]-Inches(0.22), Inches(0.2),
                 bold=True, size_pt=7.5, color=C_HEAD)
            y += Inches(0.2)
            _txt(slide, desc, cl[col_i]+Inches(0.18), y, cw[col_i]-Inches(0.22), Inches(0.38),
                 size_pt=7, color=C_BODY)
            y += Inches(0.42)


# ---------------------------------------------------------------------------
# Slide 3: LE experiment design

def _slide_le_design(prs):
    slide = _blank(prs)
    _slide_header(slide, "Exp A — Label-Efficiency Budget Study",
                  "How many labels are needed? · 3 annotation strategies · B1 & B2v2 test sets")

    CT = Inches(1.05); CH = Inches(6.15); PAD = Inches(0.12)
    cw = [Inches(3.2), Inches(5.5), Inches(4.2)]
    cl = [Inches(0.18), Inches(0.18)+cw[0]+PAD, Inches(0.18)+cw[0]+cw[1]+2*PAD]

    _panel_bg(slide, cl[0], CT, cw[0], CH)
    y = _setup_col_header(slide, "Data & Setup", cl[0], CT, cw[0]) + Inches(0.1)
    for label, val in [
        ("Dataset",    "DS1 — vinculin, CIO imaging"),
        ("Patches",    "B1: 393  |  B2v2: 393\n(4 sparse images excluded)"),
        ("Labels",     "B1 adhesion ≈ 43%\nB2v2 adhesion ≈ 37%"),
        ("Features",   "22 combos (see slide 2)"),
        ("Classifier", "LightGBM (balanced)\nLogistic Regression"),
        ("CV",         "5-fold stratified\n10 repeats per budget"),
        ("Metric",     "Balanced accuracy\n= mean(recall_adh, recall_noad)"),
    ]:
        y = _setup_kv(slide, label, val, cl[0], y, cw[0])

    _panel_bg(slide, cl[1], CT, cw[1], CH)
    y = _setup_col_header(slide, "Training Strategies", cl[1], CT, cw[1]) + Inches(0.12)
    lw, tw, aw, ew = Inches(1.2), Inches(2.0), Inches(0.35), Inches(1.7)
    sx = cl[1]+Inches(0.08)
    _txt(slide, "Train set", sx+lw, y, tw, Inches(0.22), bold=True, size_pt=9, color=C_GREY, align=PP_ALIGN.CENTER)
    _txt(slide, "Test sets", sx+lw+tw+aw, y, ew, Inches(0.22), bold=True, size_pt=9, color=C_GREY, align=PP_ALIGN.CENTER)
    y += Inches(0.24)
    configs = [
        ("B1-only\n(budget 20–340)",
         "B1 labels only\nN = 20, 40, …, 340",
         "B1 test\n+ B2v2 test",
         RGBColor(0x1f,0x77,0xb4), RGBColor(0xae,0xc7,0xe8)),
        ("B2v2-only\n(budget 20–340)",
         "B2v2 labels only\nN = 20, 40, …, 340",
         "B1 test\n+ B2v2 test",
         RGBColor(0xd6,0x27,0x28), RGBColor(0xff,0x98,0x96)),
        ("Mixed 50:50\n(budget 20–628)",
         "B1+B2v2 labels\nN = 20, …, 340, …, 628, all",
         "B1 test\n+ B2v2 test",
         RGBColor(0x2c,0xa0,0x2c), RGBColor(0x98,0xdf,0x8a)),
    ]
    rh = Inches(1.5)
    for lbl, tr, te, c_tr, c_te in configs:
        _txt(slide, lbl, sx, y+Inches(0.1), lw, rh-Inches(0.1), bold=True, size_pt=8.5, color=C_HEAD)
        sh = slide.shapes.add_shape(1, sx+lw, y+Inches(0.08), tw, rh-Inches(0.16))
        sh.fill.solid(); sh.fill.fore_color.rgb = c_tr; sh.line.fill.background()
        _txt(slide, tr, sx+lw+Inches(0.05), y+Inches(0.32), tw-Inches(0.1), rh-Inches(0.5), size_pt=8.5, color=C_WHITE, align=PP_ALIGN.CENTER)
        _txt(slide, "→", sx+lw+tw, y+Inches(0.55), aw, Inches(0.35), size_pt=14, color=C_GREY, align=PP_ALIGN.CENTER)
        sh2 = slide.shapes.add_shape(1, sx+lw+tw+aw, y+Inches(0.08), ew, rh-Inches(0.16))
        sh2.fill.solid(); sh2.fill.fore_color.rgb = c_te; sh2.line.fill.background()
        _txt(slide, te, sx+lw+tw+aw+Inches(0.05), y+Inches(0.32), ew-Inches(0.1), rh-Inches(0.5), size_pt=8.5, color=C_HEAD, align=PP_ALIGN.CENTER)
        y += rh + Inches(0.06)

    _panel_bg(slide, cl[2], CT, cw[2], CH)
    y = _setup_col_header(slide, "Key Questions", cl[2], CT, cw[2]) + Inches(0.1)
    for label, val in [
        ("Does cross-channel\ndata help?",
         "Adding zyxin (ch2) or actin (ch3)\nCP features on top of paxillin (ch1).\nDoes it improve beyond pax-only?"),
        ("Do learned\nrepresentations help?",
         "SupCon AE (32-dim latent) vs.\nhand-crafted CP & ilastik features.\nCan deep embeddings break 82% B1?"),
        ("Label budget\nefficiency",
         "At low budgets (20–100 labels),\nwhich feature set reaches\nthe best accuracy fastest?"),
        ("Annotation\ngeneralization",
         "Mixed training (B1+B2v2) helps\nboth test sets → reduces dependence\non any single annotator's style."),
    ]:
        y = _setup_kv(slide, label, val, cl[2], y, cw[2])


# ---------------------------------------------------------------------------
# Slide 4: IE experiment design

def _slide_ie_design(prs):
    slide = _blank(prs)
    _slide_header(slide, "Exp B — Image-Efficiency Test",
                  "How many training images are needed? · LOIO + image budget sweep · same 22 feature combos")

    CT = Inches(1.05); CH = Inches(6.15); PAD = Inches(0.12)
    cw = [Inches(3.2), Inches(5.5), Inches(4.2)]
    cl = [Inches(0.18), Inches(0.18)+cw[0]+PAD, Inches(0.18)+cw[0]+cw[1]+2*PAD]

    _panel_bg(slide, cl[0], CT, cw[0], CH)
    y = _setup_col_header(slide, "Data & Setup", cl[0], CT, cw[0]) + Inches(0.1)
    for label, val in [
        ("Dataset",       "DS1 — vinculin, CIO imaging"),
        ("Images",        "22 images per annotator\n(4 sparse excluded from 26)"),
        ("Conditions",    "10 control + 12 ycomp"),
        ("B1 patches",    "393 total (≈18 per image)"),
        ("B2v2 patches",  "393 total (≈18 per image)"),
        ("Features",      "22 combos (see slide 2)"),
        ("Classifier",    "LightGBM (balanced)\nLogistic Regression"),
        ("Metric",        "Balanced accuracy"),
    ]:
        y = _setup_kv(slide, label, val, cl[0], y, cw[0])

    _panel_bg(slide, cl[1], CT, cw[1], CH)
    y = _setup_col_header(slide, "Two Sub-Experiments", cl[1], CT, cw[1]) + Inches(0.12)
    for label, val in [
        ("Exp B1 — LOIO\n(Leave-One-Image-Out)",
         "Train on 21 images → test on 1 held-out\n"
         "Repeat for all 22 images (rotation)\n"
         "Label budgets: 5 → 1200 → all\n"
         "→ Measures per-image generalizability\n"
         "→ Identifies hard/easy images"),
        ("Exp B2 — Image Budget Sweep",
         "k ∈ {1, 2, 3, 5, 7, 10, 13, 16, 19} training images\n"
         "20 random repeats per k\n"
         "Label budgets: 5 → 1200 → all\n"
         "Test: all remaining images\n"
         "→ Measures how acc scales with more images"),
        ("Key question",
         "Which feature combo is most image-efficient?\n"
         "Can SupCon embeddings reduce the number\n"
         "of training images needed?\n"
         "Do cross-channel features help when\n"
         "training images are scarce?"),
    ]:
        y = _setup_kv(slide, label, val, cl[1], y, cw[1], label_pt=10, val_pt=9)
        y += Inches(0.05)

    _panel_bg(slide, cl[2], CT, cw[2], CH)
    y = _setup_col_header(slide, "Excluded Images", cl[2], CT, cw[2]) + Inches(0.1)
    for label, val in [
        ("4 excluded", "(control, f6), (control, f7)\n(control, f19), (ycomp, f0)\nReason: ≤3 patches per image"),
        ("Annotators",  "Both B1 and B2v2 evaluated\nseparately on same image set"),
        ("Label budget","For LOIO results shown here:\nlabel_budget = all (full pool)\nof 21 training images"),
        ("Repeats",     "LOIO: 1 per held-out image\nImage budget: 20 random repeats\n× each k value"),
    ]:
        y = _setup_kv(slide, label, val, cl[2], y, cw[2])


# ---------------------------------------------------------------------------
# Slide 5: LE ranked bar chart

def _slide_le_ranking(prs, clf="lgbm"):
    clf_label = CLF_LABELS[clf]
    slide = _blank(prs)
    _slide_header(slide,
        f"LE All-Budget Ranking — {clf_label}",
        "budget=all, mixed strategy · sorted by B1 test score · bars = B1 (left) and B2v2 (right)")

    df = _load_le(clf)
    if df is None:
        _txt(slide, "Data pending", Inches(1), Inches(3), Inches(11), Inches(1), size_pt=18, color=C_GREY)
        return

    sub = df[(df["budget"]==0) & (df["strategy"]=="mixed")]
    means = sub.groupby(["feat","test_batch"])["bal_acc"].mean().unstack("test_batch")

    combos_sorted = means.sort_values("b1", ascending=True).index.tolist()
    x = np.arange(len(combos_sorted))
    w = 0.38

    fig, ax = plt.subplots(figsize=(13.5, 7.0), facecolor="white")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.92, bottom=0.08)

    colors = [_combo_color(f) for f in combos_sorted]

    for i, feat in enumerate(combos_sorted):
        b1_v  = means.loc[feat, "b1"]  if "b1"  in means.columns else np.nan
        b2v2_v= means.loc[feat, "b2v2"] if "b2v2" in means.columns else np.nan
        ax.barh(i - w/2, b1_v,  w, color=colors[i], alpha=0.90, zorder=2)
        ax.barh(i + w/2, b2v2_v, w, color=colors[i], alpha=0.45, zorder=2)
        ax.text(max(b1_v, 0.01)+0.002,  i - w/2, f"{b1_v:.3f}",  va="center", fontsize=7)
        ax.text(max(b2v2_v,0.01)+0.002, i + w/2, f"{b2v2_v:.3f}", va="center", fontsize=7)

    ax.set_yticks(x)
    ax.set_yticklabels(combos_sorted, fontsize=8)
    ax.set_xlabel("Balanced accuracy", fontsize=9)
    ax.axvline(0.82, color="#333", lw=0.9, ls="--", alpha=0.5, label="0.82 (prev best)")
    ax.set_xlim(0.40, 1.02)
    ax.set_facecolor("white"); ax.grid(axis="x", color="#EEEEEE", lw=0.7)
    ax.spines[["top","right"]].set_visible(False)

    b1_patch   = mpatches.Patch(color="#555", alpha=0.90, label="B1 test (solid)")
    b2v2_patch = mpatches.Patch(color="#555", alpha=0.45, label="B2v2 test (faded)")
    ax.legend(handles=[b1_patch, b2v2_patch], fontsize=8, loc="lower right")

    # Group dividers
    y_pos = -0.5
    for grp, combos in GROUPS.items():
        in_sorted = [f for f in combos_sorted if f in combos]
        if not in_sorted:
            continue
        positions = [combos_sorted.index(f) for f in in_sorted]
        grp_bot = min(positions) - 0.5
        grp_top = max(positions) + 0.5
        ax.axhspan(grp_bot, grp_top, color=_combo_color(in_sorted[0]), alpha=0.04, zorder=0)
        ax.text(0.405, (grp_bot+grp_top)/2, grp, fontsize=7, color=_combo_color(in_sorted[0]),
                va="center", fontweight="bold", alpha=0.7)

    fig.suptitle(f"Label Efficiency — All-Budget Balanced Accuracy ({clf_label}, mixed strategy)",
                 fontsize=11, fontweight="bold", y=0.97)
    _add_fig_local(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, PLOT_H)


# ---------------------------------------------------------------------------
# Slides 6–7: LE budget curves by group

def _slide_le_budget_group(prs, clf, groups_to_show, title_suffix):
    clf_label = CLF_LABELS[clf]
    slide = _blank(prs)
    _slide_header(slide,
        f"LE Budget Curves — {title_suffix} ({clf_label})",
        "mixed strategy · mean ± std across 5 folds × 10 repeats · ★ = all-budget")

    df = _load_le(clf)
    combos_here = [f for grp in groups_to_show for f in GROUPS.get(grp, [])]
    test_batches = ["b1", "b2v2"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), facecolor="white")
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.97, top=0.90, bottom=0.11)

    for col_i, tb in enumerate(test_batches):
        ax = axes[col_i]
        ax.set_title(f"Test: {'B1 (Margaret)' if tb=='b1' else 'B2v2 (Annabel)'}",
                     fontsize=10.5, fontweight="bold", pad=5)

        if df is None:
            ax.text(0.5,0.5,"pending",ha="center",va="center",transform=ax.transAxes,color="grey"); ax.axis("off"); continue

        sub = df[(df["strategy"]=="mixed") & (df["test_batch"]==tb)]
        num_bs = sorted(b for b in sub["budget"].unique() if b != 0)
        step   = (num_bs[-1]-num_bs[-2]) if len(num_bs)>1 else 20
        all_pos = num_bs[-1]+step

        for feat in combos_here:
            sf = sub[sub["feat"]==feat]
            if sf.empty: continue
            g = sf.groupby("budget")["bal_acc"]
            means = g.mean(); stds = g.std()
            num_m = {b:v for b,v in means.items() if b!=0}
            all_m = {b:v for b,v in means.items() if b==0}
            num_s = {b:v for b,v in stds.items()  if b!=0}
            c = _combo_color(feat)
            if num_m:
                xs = sorted(num_m); ys = [num_m[b] for b in xs]; es = [num_s.get(b,0) for b in xs]
                ax.plot(xs, ys, color=c, lw=1.8, marker="o", markersize=3, label=feat)
                ax.fill_between(xs, [y-e for y,e in zip(ys,es)], [y+e for y,e in zip(ys,es)], color=c, alpha=0.10)
            if all_m:
                av = list(all_m.values())[0]
                ax.scatter([all_pos], [av], marker="*", s=90, color=c, zorder=5)

        ax.axvline(SINGLE_MAX+step/2, color="#aaa", lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("Label budget (N)", fontsize=9)
        ax.set_ylabel("Balanced accuracy" if col_i==0 else "", fontsize=9)
        xtick_vals = sorted(num_bs[::2])+[all_pos]
        ax.set_xticks(xtick_vals)
        ax.set_xticklabels([str(b) for b in num_bs[::2]]+["all"], fontsize=7.5, rotation=30)
        ax.set_ylim(0.40, 1.02)
        ax.set_facecolor("white"); ax.grid(axis="y", color="#EEEEEE", lw=0.7)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=7, framealpha=0.9, loc="lower right", ncol=1)

    fig.suptitle(f"LE Budget Curves — {clf_label} · {title_suffix} (★ = all-budget)",
                 fontsize=11, fontweight="bold", y=0.97)
    _add_fig_local(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, Inches(5.9))


# ---------------------------------------------------------------------------
# Slide 8/9: LE all-budget summary table

def _slide_le_table(prs, clf):
    clf_label = CLF_LABELS[clf]
    slide = _blank(prs)
    _slide_header(slide,
        f"LE All-Budget Summary Table — {clf_label}",
        "bal_acc × 100 at budget=all · mixed strategy · mean over 5 folds · sorted by B1 test")

    df = _load_le(clf)
    strategies = ["b1_only", "b2v2_only", "mixed"]

    col_headers = ["Feature combo",
                   "B1-test\nB1-only", "B1-test\nB2v2-only", "B1-test\nMixed",
                   "B2v2-test\nB1-only","B2v2-test\nB2v2-only","B2v2-test\nMixed"]

    # Sort by mixed/b1
    if df is not None:
        ref = (df[(df["budget"]==0)&(df["strategy"]=="mixed")&(df["test_batch"]=="b1")]
               .groupby("feat")["bal_acc"].mean().sort_values(ascending=False))
        combos_ordered = [f for f in ref.index if f in ALL_COMBOS] + \
                         [f for f in ALL_COMBOS if f not in ref.index]
    else:
        combos_ordered = ALL_COMBOS

    rows_data = []
    for feat in combos_ordered:
        row = [feat]
        for tb in ["b1","b2v2"]:
            for strat in strategies:
                if df is None: row.append("—"); continue
                sub = df[(df["feat"]==feat)&(df["strategy"]==strat)&(df["test_batch"]==tb)&(df["budget"]==0)]
                row.append(f"{sub['bal_acc'].mean()*100:.1f}" if not sub.empty else "—")
        rows_data.append(row)

    n_rows=len(rows_data); n_cols=len(col_headers)
    fig, ax = plt.subplots(figsize=(13.5, max(5.5, n_rows*0.27+1.0)), facecolor="white")
    ax.axis("off")
    tbl = ax.table(cellText=rows_data, colLabels=col_headers, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5); tbl.scale(1, 1.35)

    col_bg = {1:"#dce8f5",2:"#dce8f5",3:"#d4f0d4",4:"#fde8e8",5:"#fde8e8",6:"#d4f0d4"}
    for ci in range(n_cols):
        cell=tbl[0,ci]; cell.set_facecolor("#162130"); cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold"); cell.get_text().set_fontsize(7)
    for ri in range(n_rows):
        feat = rows_data[ri][0]
        for ci in range(n_cols):
            cell=tbl[ri+1,ci]; base=col_bg.get(ci,"#f5f5f5")
            if ri%2==1: base=_shade(base,0.93)
            cell.set_facecolor(base)
        tbl[ri+1,0].set_facecolor("#eeeeee"); tbl[ri+1,0].get_text().set_fontweight("bold")
        # colour the combo name by group
        c_hex = _combo_color(feat)
        tbl[ri+1,0].get_text().set_color(c_hex)

    fig.tight_layout(pad=0.3)
    _add_fig_local(slide, fig, Inches(0.15), Inches(1.0), Inches(13.1), Inches(6.15))


# ---------------------------------------------------------------------------
# Slide 10: IE LOIO ranking bar chart

def _slide_ie_loio_ranking(prs, clf="lgbm"):
    clf_label = CLF_LABELS[clf]
    slide = _blank(prs)
    _slide_header(slide,
        f"IE LOIO Ranking — {clf_label}",
        "label_budget=all · mean over 22 held-out images · sorted by B1 score")

    df = _load_ie(clf)
    if df is None:
        _txt(slide,"Data pending",Inches(1),Inches(3),Inches(11),Inches(1),size_pt=18,color=C_GREY); return

    loio = df[(df["experiment"]=="loio")&(df["label_budget"]=="all")]
    means = loio.groupby(["feature","annotator"])["bal_acc"].mean().unstack("annotator")
    combos_sorted = means.sort_values("b1", ascending=True).index.tolist()
    x = np.arange(len(combos_sorted)); w = 0.38

    fig, ax = plt.subplots(figsize=(13.5, 7.0), facecolor="white")
    fig.subplots_adjust(left=0.30, right=0.97, top=0.92, bottom=0.08)
    colors = [_combo_color(f) for f in combos_sorted]

    for i, feat in enumerate(combos_sorted):
        b1_v  = means.loc[feat,"b1"]  if "b1"  in means.columns else np.nan
        b2v2_v= means.loc[feat,"b2v2"] if "b2v2" in means.columns else np.nan
        ax.barh(i-w/2, b1_v,  w, color=colors[i], alpha=0.90, zorder=2)
        ax.barh(i+w/2, b2v2_v, w, color=colors[i], alpha=0.45, zorder=2)
        ax.text(max(b1_v,0.01)+0.005,  i-w/2, f"{b1_v:.3f}",  va="center", fontsize=7)
        ax.text(max(b2v2_v,0.01)+0.005, i+w/2, f"{b2v2_v:.3f}", va="center", fontsize=7)

    ax.set_yticks(x); ax.set_yticklabels(combos_sorted, fontsize=8)
    ax.set_xlabel("Balanced accuracy", fontsize=9)
    ax.axvline(0.806, color="#333", lw=0.9, ls="--", alpha=0.5, label="0.806 best B1")
    ax.set_xlim(0.35, 1.05)
    ax.set_facecolor("white"); ax.grid(axis="x", color="#EEEEEE", lw=0.7)
    ax.spines[["top","right"]].set_visible(False)
    b1p  = mpatches.Patch(color="#555", alpha=0.90, label="B1 test (solid)")
    b2p  = mpatches.Patch(color="#555", alpha=0.45, label="B2v2 test (faded)")
    ax.legend(handles=[b1p,b2p], fontsize=8, loc="lower right")
    for grp, combos in GROUPS.items():
        in_s = [f for f in combos_sorted if f in combos]
        if not in_s: continue
        positions = [combos_sorted.index(f) for f in in_s]
        ax.axhspan(min(positions)-0.5, max(positions)+0.5, color=_combo_color(in_s[0]), alpha=0.04)
        ax.text(0.36,(min(positions)+max(positions))/2, grp, fontsize=7,
                color=_combo_color(in_s[0]), va="center", fontweight="bold", alpha=0.7)

    fig.suptitle(f"Image Efficiency LOIO — Balanced Accuracy ({clf_label}, label_budget=all)",
                 fontsize=11, fontweight="bold", y=0.97)
    _add_fig_local(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, PLOT_H)


# ---------------------------------------------------------------------------
# Slide 11: IE LOIO per image (top combos)

def _slide_ie_loio_per_image(prs, clf="lgbm"):
    clf_label = CLF_LABELS[clf]
    slide = _blank(prs)
    _slide_header(slide,
        f"IE LOIO Per Held-Out Image — {clf_label}",
        "Top 8 combos by B1 score · label_budget=all · red bg=control, green bg=ycomp")

    df = _load_ie(clf)
    if df is None:
        _txt(slide,"Data pending",Inches(1),Inches(3),Inches(11),Inches(1),size_pt=18,color=C_GREY); return

    loio_all = df[(df["experiment"]=="loio")&(df["label_budget"]=="all")]
    means_b1 = loio_all[loio_all["annotator"]=="b1"].groupby("feature")["bal_acc"].mean()
    top8 = means_b1.sort_values(ascending=False).head(8).index.tolist()

    annotators = ["b1","b2v2"]
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.5), facecolor="white")
    fig.subplots_adjust(hspace=0.45, left=0.05, right=0.98, top=0.92, bottom=0.07)

    for row_i, ann in enumerate(annotators):
        ax = axes[row_i]
        ax.set_title(f"{ANN_LABELS[ann]}", fontsize=10, fontweight="bold", pad=4)
        sub = loio_all[loio_all["annotator"]==ann]
        if sub.empty: ax.axis("off"); continue

        all_images = sorted(sub["held_out_image"].unique(),
                            key=lambda x: (x.split("_f")[0], int(x.split("_f")[1])))
        x = np.arange(len(all_images))
        n = len(top8); w = 0.10
        offsets = np.linspace(-(n-1)/2*w, (n-1)/2*w, n)

        for fi, feat in enumerate(top8):
            sf = sub[sub["feature"]==feat]
            vals = [sf[sf["held_out_image"]==img]["bal_acc"].mean() if not sf[sf["held_out_image"]==img].empty else np.nan
                    for img in all_images]
            ax.bar(x+offsets[fi], vals, w, color=_combo_color(feat), alpha=0.85, label=feat, zorder=2)

        for xi, img in enumerate(all_images):
            cond = img.split("_f")[0]
            ax.axvspan(xi-0.5, xi+0.5, color="#ffe0e0" if cond=="control" else "#e0ffe0", alpha=0.25, zorder=0)

        ax.set_xticks(x)
        ax.set_xticklabels([img.replace("control_f","c").replace("ycomp_f","y")
                            for img in all_images], fontsize=7, rotation=45)
        ax.set_ylabel("Balanced accuracy", fontsize=8)
        ax.set_ylim(0.25, 1.05)
        ax.axhline(0.5, color="#aaa", lw=0.7, ls="--")
        ax.set_facecolor("white"); ax.grid(axis="y", color="#EEEEEE", lw=0.7)
        ax.spines[["top","right"]].set_visible(False)
        if row_i==0:
            ax.legend(fontsize=6.5, framealpha=0.9, loc="lower right", ncol=4)

    fig.suptitle(f"LOIO Balanced Accuracy per Held-Out Image — Top 8 combos by B1 ({clf_label})",
                 fontsize=10, fontweight="bold", y=0.98)
    _add_fig_local(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, PLOT_H)


# ---------------------------------------------------------------------------
# Slide 12: IE image budget curves

def _slide_ie_image_budget(prs, clf="lgbm"):
    clf_label = CLF_LABELS[clf]
    slide = _blank(prs)
    _slide_header(slide,
        f"IE Image Budget Curves — {clf_label}",
        "k training images, 20 repeats, label_budget=all · top 8 combos by B1 LOIO score")

    df = _load_ie(clf)
    if df is None:
        _txt(slide,"Data pending",Inches(1),Inches(3),Inches(11),Inches(1),size_pt=18,color=C_GREY); return

    loio_all = df[(df["experiment"]=="loio")&(df["label_budget"]=="all")]
    means_b1 = loio_all[loio_all["annotator"]=="b1"].groupby("feature")["bal_acc"].mean()
    top8 = means_b1.sort_values(ascending=False).head(8).index.tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), facecolor="white")
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.97, top=0.90, bottom=0.10)

    for col_i, ann in enumerate(["b1","b2v2"]):
        ax = axes[col_i]
        ax.set_title(f"{ANN_LABELS[ann]}", fontsize=10.5, fontweight="bold", pad=5)
        sub = df[(df["experiment"]=="image_budget")&(df["annotator"]==ann)&
                 (df["label_budget"]=="all")&(df["clf"]==clf)]
        if sub.empty: ax.text(0.5,0.5,"no data",ha="center",va="center",transform=ax.transAxes); continue

        k_vals = sorted(sub["k_images"].unique())
        for feat in top8:
            sf = sub[sub["feature"]==feat]
            if sf.empty: continue
            g = sf.groupby("k_images")["bal_acc"]
            means = g.mean(); stds = g.std()
            xs = sorted(means.index)
            ys = [means[k]*100 for k in xs]; es = [stds[k]*100 for k in xs]
            ax.plot(xs, ys, color=_combo_color(feat), lw=2.0, marker="o", markersize=4, label=feat)
            ax.fill_between(xs, [y-e for y,e in zip(ys,es)], [y+e for y,e in zip(ys,es)],
                            color=_combo_color(feat), alpha=0.10)

        ax.set_xlabel("Number of training images (k)", fontsize=9)
        ax.set_ylabel("Balanced accuracy (%)" if col_i==0 else "", fontsize=9)
        ax.set_xticks(k_vals); ax.set_xticklabels([str(k) for k in k_vals], fontsize=8)
        ax.set_ylim(30, 102)
        ax.set_facecolor("white"); ax.grid(axis="y", color="#EEEEEE", lw=0.7)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=7, framealpha=0.9, loc="lower right")

    fig.suptitle(f"Image Budget Sweep — {clf_label} (label_budget=all, top 8 combos by B1 LOIO)",
                 fontsize=11, fontweight="bold", y=0.97)
    _add_fig_local(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, Inches(5.7))


# ---------------------------------------------------------------------------
# Slide 13: IE LOIO label-budget effect

def _slide_ie_loio_budget(prs, clf="lgbm"):
    clf_label = CLF_LABELS[clf]
    slide = _blank(prs)
    _slide_header(slide,
        f"IE LOIO — Effect of Label Budget ({clf_label})",
        "Mean LOIO balanced accuracy vs. label budget · top 6 combos by B1 LOIO score")

    df = _load_ie(clf)
    if df is None:
        _txt(slide,"Data pending",Inches(1),Inches(3),Inches(11),Inches(1),size_pt=18,color=C_GREY); return

    loio_all = df[(df["experiment"]=="loio")&(df["label_budget"]=="all")]
    means_b1 = loio_all[loio_all["annotator"]=="b1"].groupby("feature")["bal_acc"].mean()
    top6 = means_b1.sort_values(ascending=False).head(6).index.tolist()

    BUDGET_ORDER = [5,10,20,50,100,150,200,250,300,400,500,600,700,800,900,1000,1100,1200,"all"]
    loio_df = df[df["experiment"]=="loio"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0), facecolor="white")
    fig.subplots_adjust(wspace=0.28, left=0.07, right=0.97, top=0.90, bottom=0.13)

    for col_i, ann in enumerate(["b1","b2v2"]):
        ax = axes[col_i]
        ax.set_title(f"{ANN_LABELS[ann]}", fontsize=10.5, fontweight="bold", pad=5)
        sub = loio_df[(loio_df["annotator"]==ann)&(loio_df["clf"]==clf)]
        if sub.empty: ax.text(0.5,0.5,"no data",ha="center",va="center",transform=ax.transAxes); continue

        xs_labels = []
        for feat in top6:
            sf = sub[sub["feature"]==feat]
            if sf.empty: continue
            xs, ys, es = [], [], []
            for lb in BUDGET_ORDER:
                lb_str = str(lb)
                sub_lb = sf[sf["label_budget"].astype(str)==lb_str]
                if sub_lb.empty: continue
                xs.append(lb_str); ys.append(sub_lb["bal_acc"].mean()*100)
                es.append(sub_lb["bal_acc"].std()*100)
            if xs:
                xs_labels = xs
                xi = range(len(xs))
                ax.plot(xi, ys, color=_combo_color(feat), lw=1.8, marker="o", markersize=3.5, label=feat)
                ax.fill_between(xi, [y-e for y,e in zip(ys,es)], [y+e for y,e in zip(ys,es)],
                                color=_combo_color(feat), alpha=0.10)

        if xs_labels:
            ax.set_xticks(range(len(xs_labels)))
            ax.set_xticklabels(xs_labels, fontsize=7, rotation=45)
        ax.set_xlabel("Label budget", fontsize=8)
        ax.set_ylabel("Mean LOIO bal. acc. (%)" if col_i==0 else "", fontsize=8)
        ax.set_ylim(30, 102)
        ax.set_facecolor("white"); ax.grid(axis="y", color="#EEEEEE", lw=0.7)
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(fontsize=7.5, framealpha=0.9, loc="lower right")

    fig.suptitle(f"LOIO — Mean Balanced Accuracy vs. Label Budget ({clf_label})",
                 fontsize=11, fontweight="bold", y=0.97)
    _add_fig_local(slide, fig, PLOT_LEFT, PLOT_TOP, PLOT_W, Inches(5.7))


# ---------------------------------------------------------------------------
# Slide 14: Summary

def _slide_summary(prs):
    slide = _blank(prs)
    _slide_header(slide, "Summary — Key Findings",
                  "22 feature combos · DS1 vinculin CIO · LightGBM · B1 & B2v2 annotators")

    CT = Inches(1.05); CH = Inches(6.15); PAD = Inches(0.12)
    cw = [Inches(4.1), Inches(4.1), Inches(4.9)]
    cl = [Inches(0.18), Inches(0.18)+cw[0]+PAD, Inches(0.18)+cw[0]+cw[1]+2*PAD]

    _panel_bg(slide, cl[0], CT, cw[0], CH)
    y = _setup_col_header(slide, "Top 5 — LE B1 Score", cl[0], CT, cw[0]) + Inches(0.1)
    le_top = [
        ("cp_s32+s128",            "0.820", "B2v2: 0.961"),
        ("cp_s32+supcon_z+il_s32", "0.816", "B2v2: 0.930"),
        ("supcon_z+il_s32",        "0.812", "B2v2: 0.943"),
        ("cp_s32+cp_s64_ch3",      "0.810", "B2v2: 0.966"),
        ("cp_s32+s128+dist",       "0.808", "B2v2: 0.961"),
    ]
    for feat, b1, b2v2 in le_top:
        c_hex = _combo_color(feat)
        rgb = RGBColor(int(c_hex[1:3],16), int(c_hex[3:5],16), int(c_hex[5:7],16))
        y = _setup_kv(slide, feat, f"B1: {b1}  ·  {b2v2}", cl[0], y, cw[0])

    _panel_bg(slide, cl[1], CT, cw[1], CH)
    y = _setup_col_header(slide, "Top 5 — IE LOIO B1 Score", cl[1], CT, cw[1]) + Inches(0.1)
    ie_top = [
        ("cp_s32+supcon_z",        "0.806", "B2v2: 0.990 ★"),
        ("cp_s32+supcon_z+il_s32", "0.798", "B2v2: 0.994"),
        ("cp_s16+cp_s32",          "0.794", "B2v2: 0.987"),
        ("supcon_z+il_s32",        "0.774", "B2v2: 0.958"),
        ("cp_s32+cp_s64",          "0.771", "B2v2: 0.993"),
    ]
    for feat, b1, b2v2 in ie_top:
        y = _setup_kv(slide, feat, f"B1: {b1}  ·  {b2v2}", cl[1], y, cw[1])
    _txt(slide, "★ cp_s32+supcon_z is the first combo\nto break 0.80 B1 barrier in LOIO",
         cl[1]+Inches(0.08), y+Inches(0.05), cw[1]-Inches(0.16), Inches(0.5),
         size_pt=8.5, color=RGBColor(0x9,0x46,0x7b))

    _panel_bg(slide, cl[2], CT, cw[2], CH)
    y = _setup_col_header(slide, "Key Findings", cl[2], CT, cw[2]) + Inches(0.1)
    for label, val in [
        ("B1 LE ceiling ~0.82",
         "No single feature pair broke this wall.\nCP features are saturated for B1 LE.\n~18% may be irreducible annotation noise."),
        ("SupCon helps in LOIO",
         "cp_s32+supcon_z achieves B1 LOIO=0.806,\nbeating all CP-only combos.\nLearned embedding adds info beyond hand-crafted."),
        ("Cross-channel helps B2v2",
         "Zyxin-128 gives B2v2 LE=0.971 (best).\nActin-64 gives B2v2 LE=0.966.\nChannel info matters more for easier patches."),
        ("Scale diversity helps B1 IE",
         "cp_s16+cp_s32 gives B1 IE=0.794.\nFine (s16) and coarse (s64) both help.\nMulti-scale > cross-channel for hard images."),
        ("ilastik alone < CP alone",
         "il_s32 LE B1=0.766, IE B1=0.750.\nCP s32 LE B1=0.800, IE B1=0.750.\nCP morphology is more discriminative than\nilastik pixel filters at same patch size."),
        ("Best all-round combo",
         "cp_s32+supcon_z: top IE B1, good LE B1.\ncp_s32+s128: top LE B1, good IE B1.\n→ Use supcon_z for image-limited settings,\n   s128 for label-limited settings."),
    ]:
        y = _setup_kv(slide, label, val, cl[2], y, cw[2])


# ---------------------------------------------------------------------------
# Build

def build(out_path: Path):
    print("Building comprehensive CP combo PPT …")
    prs = _prs()

    print("  Slide 1 — Title")
    _slide_title(prs)
    print("  Slide 2 — Feature sources overview")
    _slide_features(prs)
    print("  Slide 3 — LE experiment design")
    _slide_le_design(prs)
    print("  Slide 4 — IE experiment design")
    _slide_ie_design(prs)
    print("  Slide 5 — LE ranking bar chart (lgbm)")
    _slide_le_ranking(prs, "lgbm")
    print("  Slide 6 — LE budget curves: Baselines + Scale ch1 (lgbm)")
    _slide_le_budget_group(prs, "lgbm", ["Baselines","Scale ch1"], "Baselines + Scale ch1")
    print("  Slide 7 — LE budget curves: Cross-channel + Multi-source (lgbm)")
    _slide_le_budget_group(prs, "lgbm", ["Zyxin ch2","Actin ch3","Multi-source"], "Cross-channel + Multi-source")
    print("  Slide 8 — LE all-budget table (lgbm)")
    _slide_le_table(prs, "lgbm")
    print("  Slide 9 — LE all-budget table (logreg)")
    _slide_le_table(prs, "logreg")
    print("  Slide 10 — IE LOIO ranking (lgbm)")
    _slide_ie_loio_ranking(prs, "lgbm")
    print("  Slide 11 — IE LOIO per image, top 8 combos (lgbm)")
    _slide_ie_loio_per_image(prs, "lgbm")
    print("  Slide 12 — IE image budget curves, top 8 combos (lgbm)")
    _slide_ie_image_budget(prs, "lgbm")
    print("  Slide 13 — IE LOIO label-budget effect (lgbm)")
    _slide_ie_loio_budget(prs, "lgbm")
    print("  Slide 14 — Summary & key findings")
    _slide_summary(prs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out_path))
    print(f"\nSaved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_DIR / "cp_combo_comparison.pptx"))
    args = ap.parse_args()
    build(Path(args.out))


if __name__ == "__main__":
    main()

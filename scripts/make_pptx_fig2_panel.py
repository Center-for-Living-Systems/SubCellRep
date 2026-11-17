#!/usr/bin/env python3
"""
make_pptx_fig2_panel.py

All-panels draft for Figure 2 (representation comparison, label efficiency,
feature importance and interpretation, texture across FA states).

Slides
------
 1  Title / panel index
 2  A  Pipeline workflow schematic
 3  B1 LE curve — CP + SupCon (B1 held-out test; bal_acc + per-class recall)
 4  B2 LE curve — CP + IL (B2 easy setting; b2v2_only → b2v2 test; wide budget)
 5  D  Feature importance — LGBM gain vs Spearman ρ (side-by-side)
 6  E  GLCM feature interpretation cards (native PPTX shapes)
 7  E2 Patch examples — low / high values for top 2 features
 8  F1 FA4 per-class recall vs budget (from le_b2_fa4_perclass_cp_ds1.csv)
 9  F2 Feature distributions per FA class (violin plots, top features)

Usage:
  python scripts/make_pptx_fig2_panel.py

Output: results/fig2_panel.pptx
"""
from __future__ import annotations
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scipy.stats as stats
import tifffile
from PIL import Image
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO     = Path(__file__).resolve().parents[1]
RES      = REPO / "results"
EVDIR    = REPO / "eval_results" / "b1b2v2"
PRED_DIR = REPO / "eval_results" / "patch_preds"
DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
CP_CSV   = DATA / "ae_results" / "features" / "cellprofiler" / "ds1.csv"
B2_CSV   = DATA / "labelling" / "vinc_combined_label_Annabel_20260816.csv"
FA4_CSV  = DATA / "ae_results" / "features" / "eval_results" / "le_b2_fa4_perclass_cp_ds1.csv"
PATCH_ROOT = DATA / "ae_results" / "patches" / "cio" / "vinc"
OUT      = RES / "fig2_panel.pptx"

SLIDE_W, SLIDE_H = 13.33, 7.5

# ── Colours ────────────────────────────────────────────────────────────────────
C_CP    = "#1f77b4"
C_IL    = "#ff7f0e"
C_SUP   = "#2ca02c"
C_DARK  = "#1a2a3a"
C_MID   = "#2c3e50"
C_GREY  = "#888888"
C_BG    = "#f7f9fc"

FA4_COLORS = {
    "No adhesion":        "#6baed6",
    "Nascent Adhesion":   "#fd8d3c",
    "focal complex":      "#74c476",
    "focal adhesion":     "#9e9ac8",
    "fibrillar adhesion": "#e6550d",
}
FA4_ORDER = ["No adhesion", "Nascent Adhesion", "focal complex",
             "focal adhesion", "fibrillar adhesion"]

# ── Top GLCM features (from Spearman ρ analysis) ───────────────────────────────
GLCM_FEATURES = [
    # (csv_key,                    display_name,     rho,    positive, description)
    ("inverse_difference_moment_d1", "IDM d1",        "+0.75", True,
     "Local pixel homogeneity at d=1.\nHigh = smooth, uniform intensity region."),
    ("info_meas1_d1",                "IM-1 d1",       "+0.74", True,
     "Correlation between pixel pairs (entropy-based).\nHigh = structured co-occurrence pattern."),
    ("inverse_difference_moment_d2", "IDM d2",        "+0.74", True,
     "Local homogeneity at d=2.\nHigh = consistent texture over wider neighbourhood."),
    ("difference_entropy_d1",        "Diff.Entropy d1", "−0.72", False,
     "Entropy of intensity differences at d=1.\nHigh = disordered, irregular texture."),
    ("angular_second_moment_d1",     "ASM d1",        "+0.71", True,
     "Energy / uniformity of GLCM.\nHigh = repetitive, ordered texture."),
    ("correlation_d1",               "Correlation d1","+0.69", True,
     "Linear dependency of intensity between pixel pairs.\nHigh = neighbouring pixels co-vary smoothly."),
    ("contrast_d1",                  "Contrast d1",   "−0.69", False,
     "Intensity difference between pixel pairs.\nHigh = high local contrast, sharp edges."),
    ("entropy_d1",                   "Entropy d1",    "−0.67", False,
     "Overall randomness of the GLCM.\nHigh = disordered, noisy texture."),
]

# ── PPTX Helpers ───────────────────────────────────────────────────────────────

def _hex2rgb(h: str) -> RGBColor:
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _txt(slide, text, left, top, width, height,
         size=10, bold=False, color="#222222", align=PP_ALIGN.LEFT,
         italic=False, wrap=True):
    tb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = _hex2rgb(color)
    return tb


def _rect(slide, left, top, width, height, fill_hex,
          line_hex=None, line_pt=None):
    shape = slide.shapes.add_shape(
        1, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = _hex2rgb(fill_hex)
    if line_hex:
        shape.line.fill.solid()
        shape.line.fill.fore_color.rgb = _hex2rgb(line_hex)
        if line_pt:
            shape.line.width = Pt(line_pt)
    else:
        shape.line.fill.background()
    return shape


def _box_txt(slide, text, left, top, width, height,
             fill="#ffffff", line="#aaaaaa", line_pt=0.75,
             font_size=8, font_color="#333333", bold=False,
             align=PP_ALIGN.CENTER, radius=False):
    sid = 5 if radius else 1
    shape = slide.shapes.add_shape(
        sid, Inches(left), Inches(top), Inches(width), Inches(height))
    if radius:
        shape.adjustments[0] = 0.04
    shape.fill.solid()
    shape.fill.fore_color.rgb = _hex2rgb(fill)
    shape.line.fill.solid()
    shape.line.fill.fore_color.rgb = _hex2rgb(line)
    shape.line.width = Pt(line_pt)
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = _hex2rgb(font_color)
    return shape


def _arrow(slide, x1, y1, x2, y2, color="#555555", line_pt=1.0):
    conn = slide.shapes.add_connector(
        1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    conn.line.color.rgb = _hex2rgb(color)
    conn.line.width = Pt(line_pt)
    return conn


def _paste_fig(slide, fig, left, top, width, height):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    slide.shapes.add_picture(
        buf, Inches(left), Inches(top), Inches(width), Inches(height))


def _paste_pil(slide, img: Image.Image, left, top, width, height):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    slide.shapes.add_picture(
        buf, Inches(left), Inches(top), Inches(width), Inches(height))


def _panel_label(slide, letter, left, top):
    _txt(slide, letter, left, top, 0.30, 0.32, size=15, bold=True, color=C_DARK)


def _slide_header(prs, title: str, subtitle: str = "") -> object:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(slide, 0, 0, SLIDE_W, SLIDE_H, "#ffffff")
    _txt(slide, title, 0.30, 0.06, SLIDE_W - 0.60, 0.40,
         size=13, bold=True, color=C_DARK)
    if subtitle:
        _txt(slide, subtitle, 0.30, 0.44, SLIDE_W - 0.60, 0.28,
             size=8.5, italic=True, color="#666666")
    return slide


BODY_TOP = 0.52   # top of panel content area

# ── Slide 1: Title / index ─────────────────────────────────────────────────────

def _slide_title(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(slide, 0, 0, SLIDE_W, SLIDE_H, C_DARK)
    _txt(slide, "Figure 2 — Draft panel deck",
         1.0, 1.5, SLIDE_W - 2.0, 0.80,
         size=22, bold=True, color="#ffffff", align=PP_ALIGN.CENTER)
    _txt(slide,
         "Representation choice, label efficiency, and feature interpretation\n"
         "DS1 • PS32 • paxillin ch1 • U2OS cells",
         1.0, 2.40, SLIDE_W - 2.0, 0.55,
         size=12, color="#aec6e8", align=PP_ALIGN.CENTER)
    panels = [
        ("Slide 2", "A — Workflow: CP / ilastik / SupCon-AE pipelines"),
        ("Slide 3", "B1 — Label efficiency: CP + SupCon, B1 held-out test, bal_acc + per-class recall"),
        ("Slide 4", "B2 — Label efficiency: CP + IL, B2 easy setting, wide budget range (mark 95%@100)"),
        ("Slide 5", "D — Feature importance: LGBM gain vs Spearman ρ correlation"),
        ("Slide 6", "E — GLCM feature interpretation cards"),
        ("Slide 7", "E2 — Patch examples: low / high feature values (IDM d1, Correlation d1)"),
        ("Slide 8", "F1 — FA4 per-class recall vs label budget"),
        ("Slide 9", "F2 — Feature distributions per FA class (violin plots)"),
    ]
    for i, (slide_no, label) in enumerate(panels):
        y = 3.20 + i * 0.42
        _txt(slide, slide_no, 1.0, y, 1.0, 0.38,
             size=9, bold=True, color="#7fb3d3")
        _txt(slide, label, 2.1, y, SLIDE_W - 3.0, 0.38,
             size=9, color="#e0e8f0")


# ── Slide 2: A — Workflow schematic ────────────────────────────────────────────

def _draw_workflow(slide, left, top, width, height):
    C_ARROW = "#4a6278"
    BOX_W   = (width - 0.22) / 2 - 0.06
    BOX_H   = 0.44
    TRACK_L = left
    TRACK_R = left + BOX_W + 0.32
    cx_l = TRACK_L + BOX_W / 2
    cx_r = TRACK_R + BOX_W / 2

    shared_w = width * 0.55
    shared_x = left + (width - shared_w) / 2
    _box_txt(slide, "32 × 32 px patches  (paxillin, ch1, DS1)",
             shared_x, top, shared_w, BOX_H,
             fill="#e8f0fe", line="#2c5f8a", line_pt=1.2,
             font_size=8.5, font_color="#1a2a3a", bold=True, radius=True)

    fork_y  = top + BOX_H
    mid_y   = fork_y + 0.14
    y_after = fork_y + 0.30
    shared_cx = left + width / 2

    _arrow(slide, shared_cx, fork_y, shared_cx, mid_y, C_ARROW, 1.1)
    _rect(slide, cx_l, mid_y - 0.012, cx_r - cx_l, 0.024, C_ARROW)
    _arrow(slide, cx_l, mid_y, cx_l, y_after, C_ARROW, 1.1)
    _arrow(slide, cx_r, mid_y, cx_r, y_after, C_ARROW, 1.1)

    y = y_after
    _txt(slide, "Handcrafted features", TRACK_L, y - 0.26, BOX_W, 0.24,
         size=8.5, bold=True, color=C_CP, align=PP_ALIGN.CENTER)
    _txt(slide, "Learned representation", TRACK_R, y - 0.26, BOX_W, 0.24,
         size=8.5, bold=True, color=C_SUP, align=PP_ALIGN.CENTER)

    cp_sub_h = 0.58
    for si, (label, fill, lc) in enumerate([
        ("CellProfiler\n(intensity, GLCM texture,\nshape, neighbourhood)",  "#dde8f5", C_CP),
        ("Ilastik\n(random-forest pixel\nprobability maps)",                "#fff3e0", C_IL),
    ]):
        _box_txt(slide, label, TRACK_L, y + si * (cp_sub_h + 0.08),
                 BOX_W, cp_sub_h, fill=fill, line=lc, line_pt=0.9,
                 font_size=7.5, font_color="#333333", radius=True)
    cp_bot = y + 2 * (cp_sub_h + 0.08) - 0.08
    _arrow(slide, cx_l, cp_bot, cx_l, cp_bot + 0.22, C_ARROW, 1.1)
    feat_y = cp_bot + 0.22
    _box_txt(slide, "Feature vector  (~50-500 dim)",
             TRACK_L, feat_y, BOX_W, BOX_H,
             fill="#f0f4f8", line="#7a9bbf", line_pt=0.8,
             font_size=8, font_color="#333333", radius=True)
    hc_bot = feat_y + BOX_H

    sup_sub_h = 0.54
    sy = y
    for si, (label, fill, lc) in enumerate([
        ("Conv encoder\n(3 blocks → 12-dim latent)", "#d4edda", C_SUP),
        ("Projector head\n(NT-Xent contrastive loss)", "#d4edda", C_SUP),
        ("12-dim latent\n(ℓ²-normalised)",            "#f0f4f8", "#7a9bbf"),
    ]):
        _box_txt(slide, label, TRACK_R, sy, BOX_W, sup_sub_h,
                 fill=fill, line=lc, line_pt=0.9,
                 font_size=7.5, font_color="#333333", radius=True)
        if si < 2:
            _arrow(slide, cx_r, sy + sup_sub_h, cx_r, sy + sup_sub_h + 0.22, C_ARROW, 1.1)
        sy += sup_sub_h + 0.22
    sup_bot = sy - 0.22

    merge_y  = max(hc_bot, sup_bot) + 0.20
    merge_cx = left + width / 2
    merge_x  = left + (width - shared_w) / 2
    _arrow(slide, cx_l, hc_bot, merge_cx, merge_y, C_ARROW, 1.0)
    _arrow(slide, cx_r, sup_bot, merge_cx, merge_y, C_ARROW, 1.0)
    _box_txt(slide, "LightGBM or logistic regression classifier\n→ patch-level adhesion probability",
             merge_x, merge_y, shared_w, BOX_H,
             fill="#e8f0fe", line="#2c5f8a", line_pt=1.2,
             font_size=8, font_color="#1a2a3a", bold=False, radius=True)
    _txt(slide,
         "Training: supervised on labeled patches only. "
         "SupCon-AE encoder pre-trained self-supervised; classifier fitted on fixed latents.",
         left, merge_y + BOX_H + 0.07, width, 0.38,
         size=7, italic=True, color="#888888")


def slide_A_workflow(prs):
    slide = _slide_header(prs, "A — Feature extraction and representation-learning pipelines")
    _panel_label(slide, "A", 0.25, BODY_TOP)
    _draw_workflow(slide, 1.6, BODY_TOP + 0.10, SLIDE_W - 2.2, SLIDE_H - BODY_TOP - 0.20)


# ── Slide 3: B1 — LE matched test (CP + SupCon) ────────────────────────────────

def _fig_le_matched(df_cp: pd.DataFrame, df_sup: pd.DataFrame) -> plt.Figure:
    npis = [10, 25, 50, 75, 100]
    npi_s = [str(n) for n in npis]

    def _agg(df, acc_col, npi_vals):
        xs, ys, es = [], [], []
        for n in npi_vals:
            sub = df[df["npi"] == str(n)]
            if len(sub) == 0:
                continue
            n_tr = sub["n_train"].mean()
            xs.append(n_tr)
            ys.append(sub[acc_col].mean())
            es.append(sub[acc_col].std(ddof=0))
        return np.array(xs), np.array(ys), np.array(es)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=False)
    fig.patch.set_facecolor("white")

    # Panel 1: bal_acc
    ax = axes[0]
    for df, col, c, lbl, ls in [
        (df_cp,  "balanced_acc", C_CP,  "CellProfiler (B1 test)", "-"),
        (df_sup, "balanced_acc", C_SUP, "SupCon-AE 12d (B1 test)", "--"),
    ]:
        xs, ys, es = _agg(df, col, npis)
        ax.errorbar(xs, ys, yerr=es, color=c, marker="o", markersize=5,
                    linewidth=1.6, capsize=3, linestyle=ls, label=lbl)
        # "all" point
        sub_all = df[df["npi"] == "all"]
        if len(sub_all):
            xall = sub_all["n_train"].mean()
            yall = sub_all[col].mean()
            ax.plot(xall, yall, marker="o", ms=8, color=c,
                    markerfacecolor="white", markeredgewidth=2, zorder=5)
            ax.annotate(f"{yall:.3f}", (xall, yall),
                        xytext=(4, -9), textcoords="offset points",
                        fontsize=6.5, color=c)
    ax.axvline(145, color="#aaaaaa", lw=0.7, ls="--")
    ax.text(147, 0.49, "all\n(145)", fontsize=6, color="#999999")
    ax.set_xlabel("Training labels", fontsize=9)
    ax.set_ylabel("Balanced accuracy", fontsize=9)
    ax.set_title("Balanced accuracy", fontsize=9)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlim(left=0)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: adhesion recall
    ax2 = axes[1]
    for df, c, lbl in [
        (df_cp, C_CP, "CP"), (df_sup, C_SUP, "SupCon"),
    ]:
        rc = "adh_recall" if "adh_recall" in df.columns else None
        if rc is None:
            continue
        xs, ys, es = _agg(df, rc, npis)
        ax2.errorbar(xs, ys, yerr=es, color=c, marker="^", markersize=5,
                     linewidth=1.5, capsize=3, label=f"{lbl} ad recall")
    ax2.set_xlabel("Training labels", fontsize=9)
    ax2.set_ylabel("Recall (adhesion)", fontsize=9)
    ax2.set_title("Adhesion recall", fontsize=9)
    ax2.set_ylim(0.45, 1.05)
    ax2.legend(fontsize=7.5, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    # Panel 3: no-adhesion recall
    ax3 = axes[2]
    for df, c, lbl in [
        (df_cp, C_CP, "CP"), (df_sup, C_SUP, "SupCon"),
    ]:
        rc = "noad_recall" if "noad_recall" in df.columns else None
        if rc is None:
            continue
        xs, ys, es = _agg(df, rc, npis)
        ax3.errorbar(xs, ys, yerr=es, color=c, marker="s", markersize=5,
                     linewidth=1.5, capsize=3, label=f"{lbl} noad recall")
    ax3.set_xlabel("Training labels", fontsize=9)
    ax3.set_ylabel("Recall (no adhesion)", fontsize=9)
    ax3.set_title("No-adhesion recall", fontsize=9)
    ax3.set_ylim(0.45, 1.05)
    ax3.legend(fontsize=7.5, frameon=False)
    ax3.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "DS1 · PS32 · ch1 · B1+B2 matched pool (5-fold CV, 3 rep) · "
        "n_test = 394 (B1 held-out)  ·  cfg=0",
        fontsize=7.5, color="#555555", y=1.01)
    fig.tight_layout()
    return fig


def slide_B1_matched(prs, df_cp, df_sup):
    slide = _slide_header(prs,
        "B1 — Label efficiency: CP + SupCon (B1 held-out test)",
        "Same 5-fold CV and held-out test (n=394). 3 repeats per budget. cfg=0.")
    _panel_label(slide, "B", 0.25, BODY_TOP)
    fig = _fig_le_matched(df_cp, df_sup)
    _paste_fig(slide, fig, 0.50, BODY_TOP + 0.30, SLIDE_W - 0.90,
               SLIDE_H - BODY_TOP - 0.55)


# ── Slide 4: B2 — LE B2 easy setting (CP + IL, wide budget) ───────────────────

def _fig_le_b2set(df_cp_b: pd.DataFrame, df_il_b: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    fig.patch.set_facecolor("white")

    for df, c, lbl, ls, mk in [
        (df_cp_b, C_CP, "CellProfiler (b2v2 train → b2v2 test)", "-",  "o"),
        (df_il_b, C_IL, "Ilastik (b2v2 train → b2v2 test)",        ":", "^"),
    ]:
        agg = (df.groupby("budget")["bal_acc"]
               .agg(["mean", "std"]).reset_index()
               .sort_values("budget"))
        xs = agg["budget"].values
        ys = agg["mean"].values
        es = agg["std"].values
        ax.errorbar(xs, ys, yerr=es, color=c, marker=mk, markersize=5,
                    linewidth=1.8, capsize=3, linestyle=ls, label=lbl)

    # Annotations
    cp100 = df_cp_b[df_cp_b["budget"] == 100]["bal_acc"]
    if len(cp100):
        y95 = cp100.mean()
        ax.annotate(f"CP @ 100 labels\n{y95:.3f}",
                    (100, y95), xytext=(140, y95 - 0.06),
                    arrowprops=dict(arrowstyle="->", color=C_CP, lw=0.9),
                    fontsize=7.5, color=C_CP,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#e8f4fd", ec=C_CP, lw=0.7))

    ax.axhline(0.93, color="#bbbbbb", lw=0.8, ls="--")
    ax.text(820, 0.928, "93%", fontsize=7, color="#888888", va="top")

    ax.set_xlabel("Training label budget", fontsize=10)
    ax.set_ylabel("Balanced accuracy", fontsize=10)
    ax.set_ylim(0.55, 1.02)
    ax.set_xlim(left=0)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    ax.set_title("B2 easy setting — same-distribution train/test  (strategy: b2v2_only)",
                 fontsize=8.5, color="#555555", pad=5)
    fig.tight_layout()
    return fig


def slide_B2_b2set(prs, df_cp_b, df_il_b):
    slide = _slide_header(prs,
        "B2 — Label efficiency: CP + IL (B2 easy setting, b2v2_only → b2v2 test)",
        "Wide budget range. SupCon not shown (no budget CSV beyond npi=100). "
        "Mark: CP reaches 95% at 100 labels; IL needs ~750 to reach 93%.")
    _panel_label(slide, "B", 0.25, BODY_TOP)
    fig = _fig_le_b2set(df_cp_b, df_il_b)
    _paste_fig(slide, fig, 0.50, BODY_TOP + 0.28, SLIDE_W - 0.80,
               SLIDE_H - BODY_TOP - 0.50)


# ── Slide 5: D — Feature importance ────────────────────────────────────────────

def _load_feature_data():
    """Join B2 annotations with CP features to get labeled + featured dataset."""
    b2 = pd.read_csv(B2_CSV)
    b2 = b2.rename(columns={"filename": "fn"})
    b2["y"] = (b2["label"] != "No adhesion").astype(int)

    cp = pd.read_csv(CP_CSV)
    cp = cp.rename(columns={"filename": "fn"})

    df = b2.merge(cp, on="fn", how="inner")
    print(f"  Feature data: {len(df)} patches (ad={df['y'].sum()}, "
          f"no-ad={(df['y']==0).sum()})", flush=True)
    return df


def _fig_feature_importance(df: pd.DataFrame) -> plt.Figure:
    feat_cols = [c for c in df.columns if c not in ("fn", "y", "label", "annotator")]
    X = df[feat_cols].values.astype(np.float32)
    y = df["y"].values

    # Scale + train LGBM
    X_sc = StandardScaler().fit_transform(X)
    clf = LGBMClassifier(n_estimators=200, min_child_samples=1,
                         class_weight="balanced", random_state=42, verbose=-1)
    clf.fit(X_sc, y)
    prob = clf.predict_proba(X_sc)[:, 1]

    gain = pd.Series(clf.feature_importances_, index=feat_cols)
    spearman = pd.Series(
        {c: stats.spearmanr(df[c].values, prob).correlation for c in feat_cols})

    top15_gain = gain.nlargest(15)
    top15_spr  = spearman.abs().nlargest(15)

    overlap = set(top15_gain.index) & set(top15_spr.index)

    def _shorten(n):
        n = n.replace("inverse_difference_moment", "IDM")
        n = n.replace("angular_second_moment", "ASM")
        n = n.replace("difference_entropy", "diff.ent")
        n = n.replace("difference_variance", "diff.var")
        n = n.replace("info_meas1", "IM-1")
        n = n.replace("info_meas2", "IM-2")
        n = n.replace("sum_entropy", "sum.ent")
        n = n.replace("sum_average", "sum.avg")
        n = n.replace("sum_variance", "sum.var")
        n = n.replace("intensity_", "int.")
        n = n.replace("_d", " d")
        return n

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    fig.patch.set_facecolor("white")

    # LGBM gain
    names_g = [_shorten(n) for n in top15_gain.index[::-1]]
    vals_g  = top15_gain.values[::-1]
    cols_g  = ["#e8562a" if n in overlap else C_CP for n in top15_gain.index[::-1]]
    ax1.barh(names_g, vals_g, color=cols_g, edgecolor="white", height=0.7)
    ax1.set_xlabel("LGBM gain importance", fontsize=9)
    ax1.set_title("LGBM built-in (gain)", fontsize=10, fontweight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(axis="y", labelsize=7.5)

    # Spearman ρ
    spr_sorted = spearman[top15_spr.index].sort_values()
    names_s = [_shorten(n) for n in spr_sorted.index]
    vals_s  = spr_sorted.values
    cols_s  = ["#e8562a" if n in overlap else
               (C_CP if v > 0 else "#d62728") for n, v in
               zip(spr_sorted.index, spr_sorted.values)]
    ax2.barh(names_s, vals_s, color=cols_s, edgecolor="white", height=0.7)
    ax2.axvline(0, color="#aaaaaa", lw=0.8)
    ax2.set_xlabel("Spearman ρ (feature vs P(adhesion))", fontsize=9)
    ax2.set_title("Spearman ρ with P(adhesion)", fontsize=10, fontweight="bold")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="y", labelsize=7.5)

    overlap_patch = mpatches.Patch(color="#e8562a",
                                   label=f"In both top-15 ({len(overlap)})")
    cp_patch = mpatches.Patch(color=C_CP, label="Gain-only")
    rd_patch = mpatches.Patch(color="#d62728", label="ρ-only (neg)")
    fig.legend(handles=[overlap_patch, cp_patch, rd_patch],
               fontsize=7.5, frameon=False, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"LightGBM trained on all B2 annotations (n={len(df)} patches, "
        f"ad={df['y'].sum()}, no-ad={(df['y']==0).sum()})  ·  top 15 of {len(feat_cols)} features",
        fontsize=8, color="#555555", y=1.01)
    fig.tight_layout()
    return fig


def slide_D_importance(prs, df_feat):
    slide = _slide_header(prs,
        "D — Feature importance: LGBM gain vs Spearman ρ  (CP features, B2 annotations)",
        "Orange-red = feature in BOTH top-15. LGBM trained on all B2 labeled patches.")
    _panel_label(slide, "D", 0.25, BODY_TOP)
    print("  Building feature importance figure ...", flush=True)
    fig = _fig_feature_importance(df_feat)
    _paste_fig(slide, fig, 0.35, BODY_TOP + 0.28, SLIDE_W - 0.65,
               SLIDE_H - BODY_TOP - 0.52)


# ── Slide 6: E — GLCM interpretation cards ─────────────────────────────────────

def slide_E_glcm_cards(prs):
    slide = _slide_header(prs,
        "E — What the top features measure  (GLCM texture at d = 1, 2 px)",
        "CellProfiler CP features, DS1 paxillin. ρ = Spearman correlation with P(adhesion).")

    # 4 columns × 2 rows
    N_COLS, N_ROWS = 4, 2
    CARD_W = (SLIDE_W - 0.50) / N_COLS - 0.12
    CARD_H = (SLIDE_H - BODY_TOP - 0.52) / N_ROWS - 0.12
    PAD_X, PAD_Y = 0.25, BODY_TOP + 0.30

    for idx, (key, name, rho, is_pos, desc) in enumerate(GLCM_FEATURES):
        col = idx % N_COLS
        row = idx // N_COLS
        x = PAD_X + col * (CARD_W + 0.12)
        y = PAD_Y + row * (CARD_H + 0.12)

        header_col = "#1a5a8a" if is_pos else "#8a2020"
        _rect(slide, x, y, CARD_W, 0.38, header_col)
        _txt(slide, name,
             x + 0.06, y + 0.04, CARD_W - 0.12, 0.22,
             size=9.5, bold=True, color="#ffffff")
        rho_color = "#7ec8e3" if is_pos else "#f4a0a0"
        _txt(slide, f"ρ = {rho}",
             x + 0.06, y + 0.22, CARD_W - 0.12, 0.18,
             size=8.5, color=rho_color)

        _rect(slide, x, y + 0.38, CARD_W, CARD_H - 0.38, "#f7f9fc", "#dddddd", 0.5)
        _txt(slide, desc,
             x + 0.08, y + 0.44, CARD_W - 0.16, CARD_H - 0.52,
             size=8, color="#333333", wrap=True)

    # GLCM note at bottom
    _txt(slide,
         "GLCM = Gray-Level Co-occurrence Matrix. "
         "d = pixel distance. "
         "Features computed in 4 directions (0°, 45°, 90°, 135°) and averaged. "
         "All features sensitive to local paxillin intensity texture at PS32 (1 px ≈ 0.3 µm).",
         PAD_X, SLIDE_H - 0.45, SLIDE_W - 0.50, 0.40,
         size=7, italic=True, color="#888888", wrap=True)


# ── Slide 7: E2 — Patch examples (low/high feature values) ────────────────────

def _load_patch(fn: str) -> np.ndarray | None:
    cond = "ycomp" if fn.startswith("ycomp") else "control"
    path = PATCH_ROOT / cond / "tiff_patches32_mr10" / fn
    if not path.exists():
        return None
    raw = tifffile.imread(str(path)).astype(np.float32)
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)
    if hi > lo:
        raw = (raw - lo) / (hi - lo)
    return np.clip(raw, 0, 1)


def _montage_row(fns: list[str], n: int = 10) -> np.ndarray | None:
    imgs = []
    for fn in fns:
        p = _load_patch(fn)
        if p is not None:
            imgs.append(p)
        if len(imgs) >= n:
            break
    if not imgs:
        return None
    row = np.concatenate([np.pad(im, 1) for im in imgs], axis=1)
    return (row * 255).clip(0, 255).astype(np.uint8)


def _fig_patch_examples(df_feat: pd.DataFrame) -> plt.Figure:
    TOP_FEATS = [
        ("inverse_difference_moment_d1", "IDM d1  (homogeneity, ρ = +0.75)",
         "Low IDM → heterogeneous, noisy",    "High IDM → smooth, uniform"),
        ("correlation_d1",               "Correlation d1  (ρ = +0.69)",
         "Low correlation → disordered",       "High correlation → structured"),
    ]

    n_feat = len(TOP_FEATS)
    fig, axes = plt.subplots(n_feat * 2, 1,
                             figsize=(11, n_feat * 2.2 + 0.4))
    fig.patch.set_facecolor("white")
    ax_idx = 0

    for feat_col, title, low_label, high_label in TOP_FEATS:
        if feat_col not in df_feat.columns:
            ax_idx += 2
            continue
        sorted_df = df_feat.sort_values(feat_col)
        low_fns  = sorted_df["fn"].head(40).tolist()
        high_fns = sorted_df["fn"].tail(40).tolist()

        for fns, lbl in [(low_fns, low_label), (high_fns, high_label)]:
            row = _montage_row(fns, n=10)
            ax = axes[ax_idx]
            if row is not None:
                ax.imshow(row, cmap="gray", interpolation="nearest",
                          vmin=0, vmax=255)
            ax.set_ylabel(lbl, fontsize=8, rotation=0,
                          ha="right", va="center", labelpad=5)
            ax.set_xticks([])
            ax.set_yticks([])
            if ax_idx % 2 == 0:
                ax.set_title(title, fontsize=9, fontweight="bold",
                             loc="left", pad=3)
            ax_idx += 1

    fig.suptitle(
        "10 patches closest to the 5th and 95th percentile of each feature  "
        "(B2 annotations, all patches, 1-99% normalised)",
        fontsize=8, color="#555555")
    fig.tight_layout(h_pad=0.4)
    return fig


def slide_E2_patches(prs, df_feat):
    slide = _slide_header(prs,
        "E2 — Patch examples: low vs high feature values  (IDM d1, Correlation d1)",
        "10 patches at the lowest (5th pctile) and highest (95th pctile) of each feature.")
    _panel_label(slide, "E", 0.25, BODY_TOP)
    print("  Building patch example figure ...", flush=True)
    fig = _fig_patch_examples(df_feat)
    _paste_fig(slide, fig, 0.35, BODY_TOP + 0.28, SLIDE_W - 0.65,
               SLIDE_H - BODY_TOP - 0.52)


# ── Slide 8: F1 — FA4 per-class accuracy vs budget ─────────────────────────────

def _fig_fa4_accuracy(df_fa4: pd.DataFrame) -> plt.Figure:
    acc_cols = ["acc_No adhesion", "acc_focal adhesion",
                "acc_Nascent Adhesion", "acc_focal complex",
                "acc_fibrillar adhesion"]
    labels   = ["No adhesion", "focal adhesion", "Nascent Adhesion",
                "focal complex", "fibrillar adhesion"]
    colors   = [FA4_COLORS[l] for l in labels]

    # Budget as numeric; drop "all"
    df = df_fa4.copy()
    df["budget_n"] = pd.to_numeric(df["budget"], errors="coerce")
    df = df.dropna(subset=["budget_n"])
    df["budget_n"] = df["budget_n"].astype(int)

    budgets = sorted(df["budget_n"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.patch.set_facecolor("white")

    # Panel 1: per-class accuracy
    ax = axes[0]
    for col, lbl, c in zip(acc_cols, labels, colors):
        agg = (df.groupby("budget_n")[col]
               .agg(["mean", "std"]).reset_index())
        ax.errorbar(agg["budget_n"], agg["mean"], yerr=agg["std"],
                    color=c, marker="o", markersize=4,
                    linewidth=1.5, capsize=2, label=lbl)
    ax.set_xlabel("Training label budget", fontsize=9)
    ax.set_ylabel("Class accuracy (binary clf)", fontsize=9)
    ax.set_title("Per-FA-class accuracy vs budget", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0.0, 1.05)

    # Panel 2: overall bal_acc
    ax2 = axes[1]
    agg2 = (df.groupby("budget_n")["bal_acc"]
            .agg(["mean", "std"]).reset_index())
    ax2.errorbar(agg2["budget_n"], agg2["mean"], yerr=agg2["std"],
                 color=C_CP, marker="o", markersize=5,
                 linewidth=1.8, capsize=3, label="bal. acc")
    ax2.set_xlabel("Training label budget", fontsize=9)
    ax2.set_ylabel("Balanced accuracy (binary clf)", fontsize=9)
    ax2.set_title("Overall balanced accuracy", fontsize=10, fontweight="bold")
    ax2.set_ylim(0.45, 1.02)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(fontsize=7.5, frameon=False)

    fig.suptitle(
        "CP–LightGBM binary classifier · B2 FA4 labels as test set · "
        f"{df['n_test'].iloc[0] if 'n_test' in df.columns else '?'} test patches",
        fontsize=8, color="#555555", y=1.01)
    fig.tight_layout()
    return fig


def slide_F1_fa4_accuracy(prs, df_fa4):
    slide = _slide_header(prs,
        "F1 — FA4 per-class recall vs label budget  (CP–LightGBM, B2 test set)",
        "Classifier trained binary (ad / no-ad). Accuracy = recall within each subtype. "
        "From le_b2_fa4_perclass_cp_ds1.csv.")
    _panel_label(slide, "F", 0.25, BODY_TOP)
    print("  Building FA4 accuracy figure ...", flush=True)
    fig = _fig_fa4_accuracy(df_fa4)
    _paste_fig(slide, fig, 0.35, BODY_TOP + 0.28, SLIDE_W - 0.65,
               SLIDE_H - BODY_TOP - 0.52)


# ── Slide 9: F2 — Feature distributions per FA class ──────────────────────────

def _fig_fa4_distributions(df_feat_b2: pd.DataFrame) -> plt.Figure:
    TOP4 = ["correlation_d1", "inverse_difference_moment_d1",
            "difference_entropy_d1", "contrast_d1"]
    top4_labels = ["Correlation d1\n(ρ=+0.69)", "IDM d1\n(ρ=+0.75)",
                   "Diff.Entropy d1\n(ρ=−0.72)", "Contrast d1\n(ρ=−0.69)"]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    fig.patch.set_facecolor("white")

    positions = list(range(len(FA4_ORDER)))
    for ai, (feat, ftitle) in enumerate(zip(TOP4, top4_labels)):
        ax = axes[ai]
        if feat not in df_feat_b2.columns:
            ax.text(0.5, 0.5, "feature\nnot found", transform=ax.transAxes,
                    ha="center", va="center", fontsize=8, color="#999999")
            continue
        data_by_class = [
            df_feat_b2[df_feat_b2["label"] == cls][feat].dropna().values
            for cls in FA4_ORDER
        ]
        # Standardise for comparability
        all_vals = np.concatenate([d for d in data_by_class if len(d)])
        mu, sigma = all_vals.mean(), all_vals.std() + 1e-8
        data_sc = [(d - mu) / sigma for d in data_by_class]

        parts = ax.violinplot(
            [d if len(d) > 1 else np.array([0.0]) for d in data_sc],
            positions=positions,
            showmedians=True, widths=0.75)
        for pc, cls in zip(parts["bodies"], FA4_ORDER):
            pc.set_facecolor(FA4_COLORS[cls])
            pc.set_alpha(0.75)
        for lbl in ["cmedians", "cmins", "cmaxes", "cbars"]:
            parts[lbl].set_color("#444444")
            parts[lbl].set_linewidth(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(
            ["No ad.", "Nascent", "FC", "FA", "Fib."],
            fontsize=7, rotation=30, ha="right")
        ax.set_ylabel("z-score", fontsize=8)
        ax.set_title(ftitle, fontsize=8.5, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    n_by_class = {cls: int((df_feat_b2["label"] == cls).sum()) for cls in FA4_ORDER}
    fig.suptitle(
        "B2 annotations joined with CP features  ·  "
        "  ".join([f"{k}: {v}" for k, v in n_by_class.items()]) +
        "  ·  features z-scored within panel",
        fontsize=7.5, color="#555555", y=1.01)
    fig.tight_layout()
    return fig


def slide_F2_fa4_distributions(prs, df_feat_b2: pd.DataFrame):
    slide = _slide_header(prs,
        "F2 — Feature distributions per FA class  (B2 annotations × CP features)",
        "Top 4 features by |Spearman ρ|. Each panel z-scored independently. "
        "FC = focal complex, FA = focal adhesion, Fib = fibrillar.")
    _panel_label(slide, "F", 0.25, BODY_TOP)
    print("  Building FA4 distribution figure ...", flush=True)
    fig = _fig_fa4_distributions(df_feat_b2)
    _paste_fig(slide, fig, 0.25, BODY_TOP + 0.28, SLIDE_W - 0.45,
               SLIDE_H - BODY_TOP - 0.52)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    RES.mkdir(parents=True, exist_ok=True)

    print("Loading data ...", flush=True)

    # CP + SupCon LE results
    df_cp  = pd.read_csv(RES / "le_clean_results.csv")
    df_sup = pd.read_csv(RES / "le_supcon_results.csv")
    df_cp  = df_cp[df_cp["cfg"] == 0].copy()
    for df in [df_cp, df_sup]:
        if "bal_acc" in df.columns and "balanced_acc" not in df.columns:
            df.rename(columns={"bal_acc": "balanced_acc"}, inplace=True)
        df["npi"] = df["npi"].astype(str)

    # Budget CSVs: b2v2_only strategy, test_batch=b2v2
    df_cp_b = pd.read_csv(EVDIR / "le_b1b2v2_budget_cp_lgbm.csv")
    df_cp_b = df_cp_b[(df_cp_b["strategy"] == "b2v2_only") &
                      (df_cp_b["test_batch"] == "b2v2")].copy()

    df_il_b = pd.read_csv(EVDIR / "le_b1b2v2_budget_ilastik_lgbm.csv")
    df_il_b = df_il_b[(df_il_b["strategy"] == "b2v2_only") &
                      (df_il_b["test_batch"] == "b2v2")].copy()

    # B2 FA4 annotations + CP features
    print("Loading B2 annotations and CP features ...", flush=True)
    df_feat = _load_feature_data()
    b2_labels = pd.read_csv(B2_CSV).rename(columns={"filename": "fn"})
    cp_feats  = pd.read_csv(CP_CSV).rename(columns={"filename": "fn"})
    df_feat_b2 = b2_labels.merge(cp_feats, on="fn", how="inner")
    print(f"  B2 × CP: {len(df_feat_b2)} patches", flush=True)

    # FA4 per-class CSV
    df_fa4 = pd.read_csv(FA4_CSV)

    print("Building slides ...", flush=True)
    prs = Presentation()
    prs.slide_width  = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    _slide_title(prs)                            # slide 1
    slide_A_workflow(prs)                        # slide 2
    slide_B1_matched(prs, df_cp, df_sup)         # slide 3
    slide_B2_b2set(prs, df_cp_b, df_il_b)        # slide 4
    slide_D_importance(prs, df_feat)             # slide 5
    slide_E_glcm_cards(prs)                      # slide 6
    slide_E2_patches(prs, df_feat_b2)            # slide 7
    slide_F1_fa4_accuracy(prs, df_fa4)           # slide 8
    slide_F2_fa4_distributions(prs, df_feat_b2)  # slide 9

    prs.save(str(OUT))
    print(f"\nSaved → {OUT}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
make_pptx_fig2_comparison.py
==============================
Paper Figure 2: Compare handcrafted and learned representations.

Panels
------
A  Workflow schematic — CP/ilastik feature extraction vs SupCon-AE encoder  [native PPTX]
B  Learning curve — balanced accuracy vs training labels
   CP cfg=0 and SupCon-AE: same image-grouped 5-fold CV, same held-out test set (n_test=394).
   Ilastik: b2v2-only strategy, b2v2 test population — different eval design, shown separately.
C  Method summary table — key design choices, counts, and result at n_train=100

Verified values (filled in)
----------------------------
  Dataset:       DS1 — Vinculin/Paxillin, U2OS cells
  Patch size:    PS32 (32×32 px), Channel 1 (paxillin)
  Training pool: B1+B2 matched set — 393 patches per batch, 22 images
                 (56 overlapping positions; B2v2 label used for 28 conflicts)
  Splits:        5-fold CV, image-grouped (~78-79 patches/fold)
  Test set:      B1 held-out, n = 394 patches (4 folds)  [CP & SupCon]
                 B2v2 held-out per fold  [Ilastik — different eval]
  Repeats:       3 per budget level

  CP   @ n_train=100: bal_acc = 0.945 ± 0.008  (3 repeats, same splits)
  SupCon @ n_train=100: bal_acc = 0.799 ± 0.122  (3 repeats, same splits)
  CP   @ n_train=145 (all): 0.958   SupCon @ all: 0.681
  Ilastik @ budget=100 (b2v2_only→b2v2): 0.881 ± 0.043  (different test pop.)
  Ilastik @ budget=340 (b2v2_only→b2v2): 0.944 ± 0.027

Data sources
------------
  results/le_clean_results.csv    — CP features cfg=0, same splits/test as SupCon
  results/le_supcon_results.csv   — SupCon-AE, same splits/test set as CP cfg=0
  eval_results/b1b2v2/le_b1b2v2_budget_{cp,ilastik}_lgbm.csv
                                  — CP and ilastik, b2v2_only strategy, b2v2 test

Output: results/fig2_comparison.pptx
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
from PIL import Image

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
RES  = REPO / "results"
EVDIR = REPO / "eval_results" / "b1b2v2"
OUT  = RES / "fig2_comparison.pptx"

SLIDE_W, SLIDE_H = 13.33, 7.5

# ── Colours ────────────────────────────────────────────────────────────────────
C_CP     = "#1f77b4"   # CellProfiler — blue
C_IL     = "#ff7f0e"   # ilastik — orange
C_SUP    = "#2ca02c"   # SupCon-AE — green
C_DARK   = "#1a2a3a"
C_MID    = "#2c3e50"
C_GREY   = "#888888"

# ── PPTX helpers (same as fig1) ────────────────────────────────────────────────

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
    from pptx.util import Pt as _Pt
    shape = slide.shapes.add_shape(
        1, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = _hex2rgb(fill_hex)
    if line_hex:
        shape.line.fill.solid()
        shape.line.fill.fore_color.rgb = _hex2rgb(line_hex)
        if line_pt:
            shape.line.width = _Pt(line_pt)
    else:
        shape.line.fill.background()
    return shape


def _box_with_text(slide, text, left, top, width, height,
                   fill="#ffffff", line="#aaaaaa", line_pt=0.75,
                   font_size=8.5, font_color="#333333", bold=False,
                   align=PP_ALIGN.CENTER, radius=False):
    from pptx.util import Pt as _Pt
    shape_id = 5 if radius else 1
    shape = slide.shapes.add_shape(
        shape_id, Inches(left), Inches(top), Inches(width), Inches(height))
    if radius:
        shape.adjustments[0] = 0.04
    shape.fill.solid()
    shape.fill.fore_color.rgb = _hex2rgb(fill)
    shape.line.fill.solid()
    shape.line.fill.fore_color.rgb = _hex2rgb(line)
    shape.line.width = _Pt(line_pt)
    tf = shape.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = _Pt(font_size)
    run.font.bold = bold
    run.font.color.rgb = _hex2rgb(font_color)
    return shape


def _arrow(slide, x1, y1, x2, y2, color="#555555", line_pt=1.0):
    from pptx.util import Pt as _Pt
    conn = slide.shapes.add_connector(
        1, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    conn.line.color.rgb = _hex2rgb(color)
    conn.line.width = _Pt(line_pt)
    return conn


def _paste_pil(slide, img: Image.Image, left, top, width, height):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    slide.shapes.add_picture(
        buf, Inches(left), Inches(top), Inches(width), Inches(height))


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf).copy()


def _panel_label(slide, letter, left, top, size=14):
    _txt(slide, letter, left, top, 0.28, 0.32,
         size=size, bold=True, color=C_DARK)


# ── Panel A — workflow schematic ───────────────────────────────────────────────

def _draw_workflow(slide, left, top, width, height):
    """Two-track workflow: handcrafted features vs SupCon-AE, converging to classifier."""
    BOX_W = (width - 0.25) / 2 - 0.05
    BOX_H = 0.46
    C_ARROW = "#4a6278"
    TRACK_L = left                       # left track x
    TRACK_R = left + BOX_W + 0.30       # right track x

    def _track_step(x, y, text, fill, line, text_col="#1a2a3a"):
        _box_with_text(slide, text, x, y, BOX_W, BOX_H,
                       fill=fill, line=line, line_pt=1.0,
                       font_size=8, font_color=text_col,
                       align=PP_ALIGN.CENTER, radius=True)

    def _down_arrow(cx, y0):
        _arrow(slide, cx, y0, cx, y0 + 0.22, C_ARROW, 1.1)

    y = top

    # ── Shared top box: patches ────────────────────────────────────────────
    shared_cx = left + width / 2
    shared_w  = width * 0.55
    shared_x  = left + (width - shared_w) / 2
    _box_with_text(slide, "32 × 32 px patches\n(PS32, paxillin ch1, DS1)",
                   shared_x, y, shared_w, BOX_H,
                   fill="#e8f0fe", line="#2c5f8a", line_pt=1.2,
                   font_size=8.5, font_color="#1a2a3a", bold=True,
                   align=PP_ALIGN.CENTER, radius=True)

    fork_y = y + BOX_H
    y_after_fork = fork_y + 0.30

    # Fork lines from shared box → two tracks
    # Vertical stem
    _arrow(slide, shared_cx, fork_y, shared_cx, fork_y + 0.14, C_ARROW, 1.1)
    mid_y = fork_y + 0.14
    cx_l = TRACK_L + BOX_W / 2
    cx_r = TRACK_R + BOX_W / 2
    # Horizontal spread
    _rect(slide, cx_l, mid_y - 0.012, cx_r - cx_l, 0.024, C_ARROW)
    _arrow(slide, cx_l, mid_y, cx_l, y_after_fork, C_ARROW, 1.1)
    _arrow(slide, cx_r, mid_y, cx_r, y_after_fork, C_ARROW, 1.1)

    y = y_after_fork

    # ── Track headers ─────────────────────────────────────────────────────
    _txt(slide, "Handcrafted features", TRACK_L, y - 0.28, BOX_W, 0.26,
         size=8.5, bold=True, color=C_CP, align=PP_ALIGN.CENTER)
    _txt(slide, "Learned representation", TRACK_R, y - 0.28, BOX_W, 0.26,
         size=8.5, bold=True, color=C_SUP, align=PP_ALIGN.CENTER)

    # ── Left track: CP / ilastik ───────────────────────────────────────────
    cp_steps = [
        ("CellProfiler\n(intensity, texture, shape,\nneighbourhood features)",
         "#dde8f5", C_CP),
        ("Ilastik\n(pixel-level probability maps\nfrom random forests)",
         "#fff3e0", C_IL),
    ]
    # Show as two parallel sub-boxes stacked within left track
    cp_sub_h = 0.58
    for si, (label, fill, lc) in enumerate(cp_steps):
        sy = y + si * (cp_sub_h + 0.08)
        _box_with_text(slide, label, TRACK_L, sy, BOX_W, cp_sub_h,
                       fill=fill, line=lc, line_pt=0.9,
                       font_size=7.5, font_color="#333333",
                       align=PP_ALIGN.CENTER, radius=True)

    cp_bottom = y + len(cp_steps) * (cp_sub_h + 0.08) - 0.08
    _down_arrow(cx_l, cp_bottom)
    feat_y = cp_bottom + 0.22
    _box_with_text(slide, "Feature vector\n(~100–500 dim)",
                   TRACK_L, feat_y, BOX_W, BOX_H,
                   fill="#f0f4f8", line="#7a9bbf", line_pt=0.8,
                   font_size=8, font_color="#333333",
                   align=PP_ALIGN.CENTER, radius=True)
    hc_bottom = feat_y + BOX_H

    # ── Right track: SupCon-AE ─────────────────────────────────────────────
    sup_steps = [
        ("Convolutional encoder\n(3 conv blocks → 12-dim latent)",
         "#d4edda", C_SUP),
        ("Projector head\n(NT-Xent contrastive loss\nduring training)",
         "#d4edda", C_SUP),
        ("Latent representation\n(12-dim, ℓ²-normalised)",
         "#f0f4f8", "#7a9bbf"),
    ]
    sup_sub_h = 0.55
    sy = y
    for si, (label, fill, lc) in enumerate(sup_steps):
        _box_with_text(slide, label, TRACK_R, sy, BOX_W, sup_sub_h,
                       fill=fill, line=lc, line_pt=0.9,
                       font_size=7.5, font_color="#333333",
                       align=PP_ALIGN.CENTER, radius=True)
        if si < len(sup_steps) - 1:
            _down_arrow(cx_r, sy + sup_sub_h)
        sy += sup_sub_h + 0.22
    sup_bottom = sy - 0.22

    # ── Merge: both → classifier ───────────────────────────────────────────
    merge_y = max(hc_bottom, sup_bottom) + 0.22
    merge_cx = left + width / 2
    # Merge arrows
    _arrow(slide, cx_l, hc_bottom, merge_cx, merge_y, C_ARROW, 1.0)
    _arrow(slide, cx_r, sup_bottom, merge_cx, merge_y, C_ARROW, 1.0)

    merge_x = left + (width - shared_w) / 2
    _box_with_text(slide,
                   "Classifier (LightGBM or logistic regression)\n"
                   "→ Patch-level prediction",
                   merge_x, merge_y, shared_w, BOX_H,
                   fill="#e8f0fe", line="#2c5f8a", line_pt=1.2,
                   font_size=8, font_color="#1a2a3a", bold=False,
                   align=PP_ALIGN.CENTER, radius=True)

    # ── Note ──────────────────────────────────────────────────────────────
    _txt(slide,
         "Training: supervised on labeled patches only. "
         "SupCon-AE encoder trained without labels (self-supervised); "
         "classifier trained on fixed latents.",
         left, merge_y + BOX_H + 0.06, width, 0.40,
         size=7, italic=True, color="#666666", wrap=True)


# ── Panel B — learning curve ───────────────────────────────────────────────────

def _fig_panel_b(df_cp: pd.DataFrame, df_sup: pd.DataFrame,
                 df_il: pd.DataFrame | None = None) -> Image.Image:
    """Line plot: bal_acc vs n_train for CP cfg=0 vs SupCon-AE (+ optional ilastik)."""
    ORDER = ["10", "25", "50", "75", "100"]

    def _agg_npi(df, npi_col="npi", acc_col="balanced_acc"):
        out = []
        for npi in ORDER:
            sub = df[df[npi_col] == npi]
            if len(sub) == 0:
                out.append((None, None, None))
                continue
            n = sub["n_train"].mean()
            m = sub[acc_col].mean()
            s = sub[acc_col].std(ddof=0)
            out.append((n, m, s))
        return out

    def _agg_budget(df, budgets, acc_col="balanced_acc"):
        """Aggregate over budget column (ilastik b2v2 data)."""
        out = []
        for bud in budgets:
            sub = df[df["budget"] == bud]
            if len(sub) == 0:
                out.append((None, None, None))
                continue
            n = sub["n_train"].mean() if "n_train" in df.columns else float(bud)
            m = sub[acc_col].mean()
            s = sub[acc_col].std(ddof=0)
            out.append((n, m, s))
        return out

    cp_vals  = _agg_npi(df_cp)
    sup_vals = _agg_npi(df_sup)

    xs_cp  = [v[0] for v in cp_vals  if v[0] is not None]
    ys_cp  = [v[1] for v in cp_vals  if v[1] is not None]
    es_cp  = [v[2] for v in cp_vals  if v[2] is not None]
    xs_sup = [v[0] for v in sup_vals if v[0] is not None]
    ys_sup = [v[1] for v in sup_vals if v[1] is not None]
    es_sup = [v[2] for v in sup_vals if v[2] is not None]

    # "all labels" points (n_train=145 for CP/SupCon at npi='all')
    def _all_point(df, acc_col="balanced_acc"):
        sub = df[df["npi"] == "all"]
        if len(sub) == 0:
            return None, None, None
        return sub["n_train"].mean(), sub[acc_col].mean(), sub[acc_col].std(ddof=0)

    cp_all  = _all_point(df_cp)
    sup_all = _all_point(df_sup)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    fig.patch.set_facecolor("white")

    ax.errorbar(xs_cp, ys_cp, yerr=es_cp,
                color=C_CP, marker="o", markersize=5,
                linewidth=1.6, capsize=3, label="CellProfiler (CP, same test)")
    ax.errorbar(xs_sup, ys_sup, yerr=es_sup,
                color=C_SUP, marker="s", markersize=5,
                linewidth=1.6, capsize=3, linestyle="--",
                label="SupCon-AE 12d (same test)")

    # "all labels" points as open markers
    if cp_all[0] is not None:
        ax.plot(cp_all[0], cp_all[1], marker="o", markersize=7,
                color=C_CP, markerfacecolor="white", markeredgewidth=1.8,
                zorder=5)
        ax.annotate(f"{cp_all[1]:.3f}", (cp_all[0], cp_all[1]),
                    textcoords="offset points", xytext=(4, -10),
                    fontsize=6.5, color=C_CP)
    if sup_all[0] is not None:
        ax.plot(sup_all[0], sup_all[1], marker="s", markersize=7,
                color=C_SUP, markerfacecolor="white", markeredgewidth=1.8,
                zorder=5)
        ax.annotate(f"{sup_all[1]:.3f}", (sup_all[0], sup_all[1]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=6.5, color=C_SUP)

    # Ilastik from b2v2 budget CSV (different test population — shown dashed + lighter)
    if df_il is not None and len(df_il) > 0:
        acc_col = "balanced_acc" if "balanced_acc" in df_il.columns else "bal_acc"
        budgets = sorted(df_il["budget"].unique())
        il_vals = _agg_budget(df_il, budgets, acc_col=acc_col)
        xs_il = [v[0] for v in il_vals if v[0] is not None]
        ys_il = [v[1] for v in il_vals if v[0] is not None]
        es_il = [v[2] for v in il_vals if v[0] is not None]
        if xs_il:
            ax.errorbar(xs_il, ys_il, yerr=es_il,
                        color=C_IL, marker="^", markersize=5,
                        linewidth=1.4, capsize=3, linestyle=":",
                        alpha=0.85,
                        label="Ilastik (b2v2 test†)")

    ax.set_xlabel("Training labels (patches)", fontsize=9)
    ax.set_ylabel("Balanced accuracy", fontsize=9)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlim(left=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.set_title(
        "DS1 / PS32 / ch1  ·  B1+B2 matched (22 img, 393 patches/batch)  ·  5-fold CV  ·  n_test = 394  ·  3 rep.",
        fontsize=7, color="#555555", pad=4)

    # Vertical line at n=145 (all labels)
    ax.axvline(145, color="#aaaaaa", linewidth=0.7, linestyle="--")
    ax.text(147, 0.50, "all labels\n(n=145)", fontsize=6, color="#999999", va="bottom")

    ax.axhline(0.5, color="#cccccc", linewidth=0.6, linestyle=":")
    ax.annotate("† different test pop.", (0, 0), xycoords="axes fraction",
                xytext=(0, -0.16), textcoords="axes fraction",
                fontsize=6, color="#888888")

    fig.tight_layout()
    return _fig_to_pil(fig)


# ── Panel C — method summary table ─────────────────────────────────────────────

def _draw_method_table(slide, left, top, width, height,
                       df_cp: pd.DataFrame, df_sup: pd.DataFrame,
                       df_il: pd.DataFrame | None = None):
    """Summary table: design choices + result at n_train=100."""
    _txt(slide,
         "Method comparison  ·  training pool: B1+B2 matched (22 images, 393 patches/batch, after dedup)  ·  n_test = 394",
         left, top, width, 0.30,
         size=8.5, bold=True, color=C_DARK, wrap=True)

    # Compute mean ± std at npi=100 for CP and SupCon
    def _stat_npi(df, col="balanced_acc"):
        sub = df[df["npi"] == "100"]
        if len(sub) == 0:
            return "n/a"
        return f"{sub[col].mean():.3f} ± {sub[col].std(ddof=0):.3f}"

    cp_stat  = _stat_npi(df_cp)
    sup_stat = _stat_npi(df_sup)

    # Ilastik at budget≈100 from b2v2 data
    il_stat = "n/a"
    if df_il is not None and len(df_il) > 0:
        acc_col = "balanced_acc" if "balanced_acc" in df_il.columns else "bal_acc"
        # find budget closest to 100
        budgets = sorted(df_il["budget"].unique())
        bud_100 = min(budgets, key=lambda b: abs(b - 100))
        sub_il = df_il[df_il["budget"] == bud_100]
        if len(sub_il) > 0:
            il_stat = (f"{sub_il[acc_col].mean():.3f} ± {sub_il[acc_col].std(ddof=0):.3f}"
                       f"  (bud={bud_100})*")

    # Table data
    headers = ["Property", "CellProfiler (CP)", "Ilastik", "SupCon-AE"]
    rows = [
        ("Feature type",        "Hand-crafted",            "Hand-crafted (prob. maps)", "Self-supervised learned"),
        ("Feature dim",         "~100–500",                "~50–200",                  "12-dim latent"),
        ("Label use",           "Classifier only",         "Classifier only",           "Encoder: none\nClf: labels"),
        ("Classifier",          "LightGBM",                "LightGBM",                 "LightGBM"),
        ("Bal. acc @ 100 labels", cp_stat,                 il_stat,                    sup_stat),
        ("Bal. acc @ all labels","0.958 (n_train=145)",   "0.944 ± 0.027 (bud=340)*", "0.681 (n_train=145)"),
        ("Annotation set",      "B1+B2 matched\n(393/batch, 22 img)",
                                "B2v2 only\n(~340 total)", "B1+B2 matched\n(393/batch, 22 img)"),
        ("Test set",            "B1 held-out\nn = 394",   "B2v2 held-out*\n(per fold)", "B1 held-out\nn = 394"),
    ]

    COL_W = [width * 0.28, width * 0.24, width * 0.24, width * 0.24]
    COL_X = [left + sum(COL_W[:i]) for i in range(4)]
    ROW_H = (height - 0.38) / (len(rows) + 1)

    # Header
    hdr_y = top + 0.33
    _rect(slide, left, hdr_y, width, ROW_H, C_MID)
    for ci, (h, cw, cx) in enumerate(zip(headers, COL_W, COL_X)):
        _txt(slide, h, cx + 0.04, hdr_y + 0.04, cw - 0.08, ROW_H - 0.06,
             size=7.5, bold=True, color="#ffffff")

    ACC_ROWS = {4, 5}
    for ri, (key, *vals) in enumerate(rows):
        ry = hdr_y + (ri + 1) * ROW_H
        bg = "#f5f7ff" if ri % 2 == 0 else "#ffffff"
        _rect(slide, left, ry, width, ROW_H, bg)
        _txt(slide, key, COL_X[0] + 0.04, ry + 0.03, COL_W[0] - 0.08, ROW_H - 0.04,
             size=7.5, bold=True, color="#333333")
        for ci, (val, cw, cx) in enumerate(zip(vals, COL_W[1:], COL_X[1:]), 1):
            col_color = [C_CP, C_IL, C_SUP][ci - 1] if ri in ACC_ROWS else "#444444"
            is_bold = ri in ACC_ROWS
            _txt(slide, val, cx + 0.04, ry + 0.03, cw - 0.08, ROW_H - 0.04,
                 size=7.5, bold=is_bold, color=col_color)

    note_y = hdr_y + (len(rows) + 1) * ROW_H + 0.03
    _txt(slide,
         "* Ilastik eval uses b2v2-only annotation strategy, test = B2v2 held-out (different from CP/SupCon); "
         "values not directly comparable. CP and SupCon: le_clean cfg=0 / le_supcon, 5-fold CV, 3 repeats.",
         left, note_y, width, 0.42,
         size=6.5, italic=True, color="#888888", wrap=True)


# ── Main slide ─────────────────────────────────────────────────────────────────

def build_slide(prs, df_cp, df_sup, df_il=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(slide, 0, 0, SLIDE_W, SLIDE_H, "#ffffff")

    # Title
    _txt(slide,
         "Figure 2 — Comparing handcrafted and learned representations  "
         "(DS1, PS32, ch1 paxillin, U2OS)",
         0.30, 0.06, SLIDE_W - 0.60, 0.35,
         size=13, bold=True, color=C_DARK, align=PP_ALIGN.LEFT)
    _rect(slide, 0.30, 0.43, SLIDE_W - 0.60, 0.018, "#cccccc")

    BODY_TOP = 0.48

    # ── Left column: Panel A (workflow) ───────────────────────────────────
    WF_W = 5.10
    WF_X = 0.25

    _panel_label(slide, "A", WF_X, BODY_TOP)
    _txt(slide, "Feature extraction and representation-learning pipelines",
         WF_X + 0.30, BODY_TOP, WF_W - 0.30, 0.28,
         size=8.5, italic=True, color="#555555")

    _draw_workflow(slide, WF_X, BODY_TOP + 0.30, WF_W, SLIDE_H - BODY_TOP - 0.30)

    # Divider
    _rect(slide, WF_X + WF_W + 0.12, BODY_TOP + 0.1, 0.018,
          SLIDE_H - BODY_TOP - 0.25, "#dddddd")

    # ── Right column: Panels B + C ────────────────────────────────────────
    RX = WF_X + WF_W + 0.28
    RW = SLIDE_W - RX - 0.25

    # Panel B — learning curve (top)
    CURVE_H = 3.60
    _panel_label(slide, "B", RX, BODY_TOP)
    _txt(slide,
         "Balanced accuracy vs training-label count  "
         "(CP & SupCon: same 5-fold CV, n_test = 394;  Ilastik: b2v2-only strategy†)",
         RX + 0.30, BODY_TOP, RW - 0.30, 0.28,
         size=8.5, italic=True, color="#555555")
    img_b = _fig_panel_b(df_cp, df_sup, df_il)
    _paste_pil(slide, img_b, RX, BODY_TOP + 0.28, RW, CURVE_H)

    # Panel C — method table (bottom)
    C_TOP = BODY_TOP + CURVE_H + 0.18
    C_H   = SLIDE_H - C_TOP - 0.15
    _panel_label(slide, "C", RX, C_TOP)
    _txt(slide, "Method design and performance summary",
         RX + 0.30, C_TOP, RW - 0.30, 0.28,
         size=8.5, italic=True, color="#555555")
    _draw_method_table(slide, RX, C_TOP + 0.28, RW, C_H - 0.28,
                       df_cp, df_sup, df_il)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df_cp  = pd.read_csv(RES / "le_clean_results.csv")
    df_sup = pd.read_csv(RES / "le_supcon_results.csv")

    # Keep only cfg=0 for CP (fair: same n_train as SupCon)
    df_cp = df_cp[df_cp["cfg"] == 0].copy()
    # Rename balanced_acc column if needed
    for df in [df_cp, df_sup]:
        if "bal_acc" in df.columns and "balanced_acc" not in df.columns:
            df.rename(columns={"bal_acc": "balanced_acc"}, inplace=True)
    # Ensure npi is string
    df_cp["npi"]  = df_cp["npi"].astype(str)
    df_sup["npi"] = df_sup["npi"].astype(str)

    # Ilastik budget data: b2v2-only strategy, test=b2v2 — different eval design
    il_path = EVDIR / "le_b1b2v2_budget_ilastik_lgbm.csv"
    df_il = None
    if il_path.exists():
        df_il = pd.read_csv(il_path)
        df_il = df_il[(df_il["strategy"] == "b2v2_only") &
                      (df_il["test_batch"] == "b2v2")].copy()
        if "bal_acc" in df_il.columns and "balanced_acc" not in df_il.columns:
            df_il.rename(columns={"bal_acc": "balanced_acc"}, inplace=True)
        if "budget" in df_il.columns:
            df_il["budget"] = pd.to_numeric(df_il["budget"], errors="coerce")
    else:
        print(f"Warning: ilastik budget file not found at {il_path}")

    prs = Presentation()
    prs.slide_width  = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    build_slide(prs, df_cp, df_sup, df_il)
    prs.save(OUT)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()

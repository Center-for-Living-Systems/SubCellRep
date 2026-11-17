#!/usr/bin/env python3
"""
make_pptx_fig3_annotation_strategy.py
=======================================
Paper Figure 3: Annotation strategy and training-data requirements.

Panels
------
A  Annotation-strategy schematic: B1 (broadly sampled) vs B2 (prototype-selected)
   Sampling statistics; notes on B1 cell-front bias and error-focused additions.  [native PPTX]

B  Matched cross-evaluation 2×2 heatmap: {B1, B2} train × {B1, B2} test
   Shows training vs test-population effects. CP LightGBM; 5-fold CV.  [matplotlib]

C  Label-efficiency curves on fixed B1 test population (all-patch representative)
   Compares B1-only / B2-only / mixed training for CP and ilastik.  [matplotlib]

D  Image-efficiency curve: number of source images vs performance (budget=all)
   CP and ilastik; annotator b2v2; LOIO experiment.  [matplotlib]

Data sources
------------
  labelling/labels_vinc_20260521.csv                     — B1 annotations
  labelling/vinc_combined_label_Annabel_20260816.csv      — B2 annotations
  eval_results/b1b2v2/le_b1b2v2_cross_{cp,ilastik}_lgbm.csv     — Panel B
  eval_results/b1b2v2/le_b1b2v2_budget_{cp,ilastik}_lgbm.csv    — Panel C
  eval_results/image_efficiency_b1b2v2.csv                       — Panel D

Output: results/fig3_annotation_strategy.pptx
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from PIL import Image

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO   = Path(__file__).resolve().parents[1]
DATA   = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LAB    = DATA / "labelling"
EVDIR  = REPO / "eval_results" / "b1b2v2"
EVROOT = REPO / "eval_results"
OUT    = REPO / "results" / "fig3_annotation_strategy.pptx"

SLIDE_W, SLIDE_H = 13.33, 7.5

# ── Colours ────────────────────────────────────────────────────────────────────
C_B1    = "#1f77b4"   # blue  — B1 (broadly sampled)
C_B2    = "#d62728"   # red   — B2 (prototype-selected)
C_MIX   = "#9467bd"  # purple — mixed strategy
C_CP    = "#1f77b4"
C_IL    = "#ff7f0e"
C_DARK  = "#1a2a3a"
C_MID   = "#2c3e50"
C_GREY  = "#888888"

FA5_COLORS = {
    "No adhesion":        "#9467bd",
    "Nascent Adhesion":   "#1f77b4",
    "focal complex":      "#ff7f0e",
    "focal adhesion":     "#2ca02c",
    "fibrillar adhesion": "#d62728",
}

# ── PPTX helpers ───────────────────────────────────────────────────────────────

def _hex2rgb(h):
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _txt(slide, text, left, top, width, height,
         size=9, bold=False, color="#222222", align=PP_ALIGN.LEFT,
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


def _rect(slide, left, top, width, height, fill_hex, line_hex=None, line_pt=None):
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


def _box_txt(slide, text, left, top, width, height,
             fill="#f0f4f8", line="#aaaaaa", line_pt=0.75,
             size=8.5, color="#333333", bold=False,
             align=PP_ALIGN.CENTER, radius=False):
    from pptx.util import Pt as _Pt
    shape_id = 5 if radius else 1
    shape = slide.shapes.add_shape(
        shape_id, Inches(left), Inches(top), Inches(width), Inches(height))
    if radius:
        shape.adjustments[0] = 0.03
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
    run.font.size = _Pt(size)
    run.font.bold = bold
    run.font.color.rgb = _hex2rgb(color)
    return shape


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


def _panel_label(slide, letter, left, top, size=13):
    _txt(slide, letter, left, top, 0.28, 0.30,
         size=size, bold=True, color=C_DARK)


# ── Panel A — annotation strategy schematic ────────────────────────────────────

def _draw_panel_a(slide, left, top, width, height,
                  b1_stats: dict, b2_stats: dict):
    """Two-column comparison of B1 vs B2 annotation strategies."""
    col_w = (width - 0.15) / 2
    b1_x  = left
    b2_x  = left + col_w + 0.15

    # ── Colour bands ─────────────────────────────────────────────────────
    _rect(slide, b1_x, top, col_w, 0.28, C_B1)
    _txt(slide, "Batch 1 (B1)  —  Broadly sampled",
         b1_x + 0.06, top + 0.03, col_w - 0.1, 0.24,
         size=9, bold=True, color="#ffffff")

    _rect(slide, b2_x, top, col_w, 0.28, C_B2)
    _txt(slide, "Batch 2 (B2)  —  Prototype-selected",
         b2_x + 0.06, top + 0.03, col_w - 0.1, 0.24,
         size=9, bold=True, color="#ffffff")

    # ── Sampling description ──────────────────────────────────────────────
    b1_notes = (
        "Annotated via LabelStudio (Nov 2025 – Feb 2026).\n"
        "Cell-front bias: patches sampled from lamellipodia-adjacent regions "
        "and early-adhesion zones. Error-focused additions: hard cases "
        "returned from model predictions.\n"
        "Annotators: Margaret, Annabel."
    )
    b2_notes = (
        "Annotated via prototype interface (Aug 2026).\n"
        "Prototype-selected: Annabel chose clear, unambiguous examples of "
        "each class from a high-density grid scan. "
        "Concentrated on fewer images; high patch density per image.\n"
        "Annotator: Annabel."
    )
    desc_h = 1.28
    desc_y = top + 0.30
    _box_txt(slide, b1_notes, b1_x, desc_y, col_w, desc_h,
             fill="#dde8f5", line=C_B1, line_pt=0.8,
             size=7.5, color="#1a2a3a", align=PP_ALIGN.LEFT)
    _box_txt(slide, b2_notes, b2_x, desc_y, col_w, desc_h,
             fill="#fde8e8", line=C_B2, line_pt=0.8,
             size=7.5, color="#1a2a3a", align=PP_ALIGN.LEFT)

    # ── Stats table (rows = metric, cols = B1 / B2) ───────────────────────
    stat_y = desc_y + desc_h + 0.08
    rows = [
        ("Total patches",           str(b1_stats["n_patches"]),     str(b2_stats["n_patches"])),
        ("Unique images",           str(b1_stats["n_images"]),      str(b2_stats["n_images"])),
        ("Images (ctrl / ycomp)",   b1_stats["n_img_cond"],         b2_stats["n_img_cond"]),
        ("Patches / image (mean)",  b1_stats["ppi_mean"],           b2_stats["ppi_mean"]),
        ("Patches / image (range)", b1_stats["ppi_range"],          b2_stats["ppi_range"]),
        ("Conditions annotated",    b1_stats["conditions"],         b2_stats["conditions"]),
    ]
    ROW_H   = 0.285
    STAT_W  = col_w
    KEY_W   = STAT_W * 0.50
    VAL_W   = STAT_W * 0.50

    for bi, (bx, bcolor) in enumerate([(b1_x, C_B1), (b2_x, C_B2)]):
        for ri, row in enumerate(rows):
            ry = stat_y + ri * ROW_H
            bg = "#f0f4f8" if ri % 2 == 0 else "#ffffff"
            _rect(slide, bx, ry, STAT_W, ROW_H, bg)
            if bi == 0:
                _txt(slide, row[0],
                     bx + 0.04, ry + 0.04, KEY_W - 0.06, ROW_H - 0.06,
                     size=7.5, bold=True, color="#333333")
            _txt(slide, row[bi + 1],
                 bx + KEY_W if bi == 0 else bx + 0.04,
                 ry + 0.04,
                 VAL_W - 0.06 if bi == 0 else STAT_W - 0.08,
                 ROW_H - 0.06,
                 size=7.5, bold=False, color=bcolor)

    # ── Note ─────────────────────────────────────────────────────────────
    note_y = stat_y + len(rows) * ROW_H + 0.05
    _txt(slide,
         "B1 and B2 share 56 patch positions (28 label conflicts; B2 label used).\n"
         "B1 better reflects intended all-patch classification; "
         "B2 emphasises prototype clarity and may overstate apparent accuracy on own test split.",
         left, note_y, width, 0.50,
         size=7, italic=True, color=C_GREY, wrap=True)


# ── Panel B — cross-evaluation heatmap ────────────────────────────────────────

def _fig_panel_b(df_cross_cp: pd.DataFrame, df_cross_il: pd.DataFrame) -> Image.Image:
    """2×2 heatmap (+ ilastik side-by-side) showing train×test balanced accuracy."""
    def _build_matrix(df):
        mat   = np.full((2, 2), np.nan)
        se    = np.full((2, 2), np.nan)
        order = ["b1", "b2v2"]
        for ti, tb in enumerate(order):
            for tj, tr in enumerate(order):
                sub = df[(df["train_batch"] == tr) & (df["test_batch"] == tb)]
                if len(sub):
                    mat[ti, tj] = sub["bal_acc"].mean()
                    se[ti, tj]  = sub["bal_acc"].std(ddof=0)
        return mat, se

    mat_cp, se_cp = _build_matrix(df_cross_cp)
    mat_il, se_il = _build_matrix(df_cross_il)

    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.8))
    fig.patch.set_facecolor("white")
    labels   = ["B1 test", "B2 test"]
    xlabels  = ["B1 train", "B2 train"]

    for ax, mat, se, title in zip(axes,
                                   [mat_cp, mat_il],
                                   [se_cp, se_il],
                                   ["CellProfiler (LGBM)", "Ilastik (LGBM)"]):
        im = ax.imshow(mat, vmin=0.60, vmax=1.00,
                       cmap="RdYlGn", aspect="auto")
        for i in range(2):
            for j in range(2):
                v  = mat[i, j]
                sv = se[i, j]
                txt = f"{v:.3f}\n±{sv:.3f}" if not np.isnan(v) else "–"
                # Outline the diagonal (own-test)
                fc = "white" if v < 0.85 else "#1a1a1a"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8, fontweight="bold", color=fc)
                if i == j:
                    rect = mpatches.FancyBboxPatch(
                        (j - 0.49, i - 0.49), 0.98, 0.98,
                        boxstyle="round,pad=0.02",
                        linewidth=2, edgecolor="#333333", facecolor="none",
                        transform=ax.transData)
                    ax.add_patch(rect)

        ax.set_xticks([0, 1]);  ax.set_xticklabels(xlabels, fontsize=8)
        ax.set_yticks([0, 1]);  ax.set_yticklabels(labels,   fontsize=8)
        ax.set_title(title, fontsize=8.5, fontweight="bold", pad=5)

    cb = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Balanced accuracy", fontsize=7.5)

    fig.suptitle(
        "Cross-evaluation: training vs test annotation batch  "
        "(all available labels, 5-fold CV, mean ± SD)",
        fontsize=7.5, color="#444444", y=1.03)
    fig.tight_layout()
    return _fig_to_pil(fig)


# ── Panel C — label-efficiency curves ─────────────────────────────────────────

def _fig_panel_c(df_cp: pd.DataFrame, df_il: pd.DataFrame,
                 test_batch: str = "b1") -> Image.Image:
    """Learning curves: B1-only / B2-only / mixed training → fixed test population."""
    STRATS = {
        "b1_only":   (C_B1,  "solid",  "B1-only train"),
        "b2v2_only": (C_B2,  "solid",  "B2-only train"),
        "mixed":     (C_MIX, "solid",  "Mixed train"),
    }
    BUDGETS = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 260, 300, 340]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (df, feat_label) in zip(axes, [(df_cp, "CP (LGBM)"),
                                            (df_il, "Ilastik (LGBM)")]):
        sub = df[df["test_batch"] == test_batch]
        for strat, (col, ls, label) in STRATS.items():
            grp = sub[sub["strategy"] == strat]
            vals = []
            for b in BUDGETS:
                row = grp[grp["budget"] == b]
                if len(row):
                    vals.append((b, row["bal_acc"].mean(), row["bal_acc"].std(ddof=0)))
            if not vals:
                continue
            xs, ys, es = zip(*vals)
            ax.plot(xs, ys, color=col, linestyle=ls,
                    linewidth=1.5, label=label, marker="o", markersize=3)
            ax.fill_between(xs,
                            [y - e for y, e in zip(ys, es)],
                            [y + e for y, e in zip(ys, es)],
                            color=col, alpha=0.12)

        ax.set_title(feat_label, fontsize=9, fontweight="bold", pad=4)
        ax.set_xlabel("Training labels (budget)", fontsize=8)
        if ax == axes[0]:
            ax.set_ylabel("Balanced accuracy\n(B1 test population)", fontsize=8)
        ax.set_ylim(0.50, 1.02)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7.5)
        ax.axhline(0.5, color="#cccccc", linewidth=0.5, linestyle=":")
        if ax == axes[0]:
            ax.legend(fontsize=7.5, frameon=False)

    fig.suptitle(
        f"Test population: B1 all-patch  (intended all-patch use)  ·  "
        f"shading = ±1 SD across repeats",
        fontsize=7.5, color="#555555", y=1.04)
    fig.tight_layout()
    return _fig_to_pil(fig)


# ── Panel D — image-efficiency curves ─────────────────────────────────────────

def _fig_panel_d(df_imgeff: pd.DataFrame) -> Image.Image:
    """Balanced accuracy vs number of source training images."""
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    fig.patch.set_facecolor("white")

    sub = df_imgeff[
        (df_imgeff["annotator"] == "b2v2") &
        (df_imgeff["experiment"] == "image_budget") &
        (df_imgeff["label_budget"] == "all")
    ]

    for feat, col, lbl in [("cp", C_CP, "CP (LGBM)"), ("ilastik", C_IL, "Ilastik (LGBM)")]:
        grp = sub[sub["feature"] == feat].groupby("k_images")["bal_acc"]
        means = grp.mean()
        stds  = grp.std(ddof=0).fillna(0)
        ax.plot(means.index, means.values, color=col,
                marker="o", markersize=4, linewidth=1.5, label=lbl)
        ax.fill_between(means.index,
                        means.values - stds.values,
                        means.values + stds.values,
                        color=col, alpha=0.12)

    ax.set_xlabel("Source images in training pool", fontsize=8)
    ax.set_ylabel("Balanced accuracy", fontsize=8)
    ax.set_title("Image efficiency\n(b2v2 annotations, budget = all labels)",
                 fontsize=8.5, fontweight="bold", pad=4)
    ax.set_ylim(0.40, 1.02)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.axhline(0.5, color="#cccccc", linewidth=0.5, linestyle=":")
    ax.legend(fontsize=7.5, frameon=False)

    # Annotate plateau
    ax.axvline(5, color="#888888", linewidth=0.7, linestyle="--")
    ax.text(5.2, 0.44, "≈ plateau\n(5 images)", fontsize=6.5, color="#888888")

    fig.tight_layout()
    return _fig_to_pil(fig)


# ── Main slide ─────────────────────────────────────────────────────────────────

def build_slide(prs, b1, b2, df_cross_cp, df_cross_il,
                df_budget_cp, df_budget_il, df_imgeff):

    # ── Annotation stats ─────────────────────────────────────────────────
    def _img_count(df, cond_col, frame_col):
        df = df.copy()
        df["_id"] = df[cond_col] + "_" + df[frame_col].astype(str)
        return df["_id"].nunique(), df.groupby(cond_col)["_id"].nunique()

    b1["_frame"] = b1["czi_filename"].str.extract(r"-(\d+)\.czi").iloc[:, 0]
    b1["_frame"] = b1["_frame"].apply(
        lambda x: f"f{int(x)-1:04d}" if pd.notna(x) else None)
    b1["_id"] = b1["condition"] + "_" + b1["_frame"]
    b1_n_img   = b1["_id"].nunique()
    b1_per_img = b1.groupby("_id").size()
    b1_cond    = b1.groupby("condition")["_id"].nunique()

    b2["_cond"]  = b2["filename"].str.extract(r"^(control|ycomp)").iloc[:, 0]
    b2["_frame"] = b2["filename"].str.extract(r"^(?:control|ycomp)_(f\d+)").iloc[:, 0]
    b2["_id"]    = b2["_cond"] + "_" + b2["_frame"]
    b2_n_img     = b2["_id"].nunique()
    b2_per_img   = b2.groupby("_id").size()
    b2_cond      = b2.groupby("_cond")["_id"].nunique()

    b1_stats = dict(
        n_patches  = f"{len(b1):,}",
        n_images   = str(b1_n_img),
        n_img_cond = f"ctrl {b1_cond.get('control', 0)} / ycomp {b1_cond.get('ycomp', 0)}",
        ppi_mean   = f"{b1_per_img.mean():.1f}",
        ppi_range  = f"{b1_per_img.min()}–{b1_per_img.max()}",
        conditions = "control + ycomp",
    )
    b2_stats = dict(
        n_patches  = f"{len(b2):,}",
        n_images   = str(b2_n_img),
        n_img_cond = f"ctrl {b2_cond.get('control', 0)} / ycomp {b2_cond.get('ycomp', 0)}",
        ppi_mean   = f"{b2_per_img.mean():.1f}",
        ppi_range  = f"{b2_per_img.min()}–{b2_per_img.max()}",
        conditions = "control + ycomp",
    )

    # ── Slide ────────────────────────────────────────────────────────────
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(slide, 0, 0, SLIDE_W, SLIDE_H, "#ffffff")

    # Title
    _txt(slide,
         "Figure 3 — Annotation strategy and training-data requirements  "
         "(DS1, PS32, ch1 paxillin)",
         0.30, 0.06, SLIDE_W - 0.60, 0.35,
         size=13, bold=True, color=C_DARK, align=PP_ALIGN.LEFT)
    _rect(slide, 0.30, 0.43, SLIDE_W - 0.60, 0.018, "#cccccc")

    BODY_TOP  = 0.48
    MID_Y     = BODY_TOP + (SLIDE_H - BODY_TOP) / 2 + 0.05   # divider between rows
    LEFT_W    = 5.25   # Panel A width
    RIGHT_X   = LEFT_W + 0.45
    RIGHT_W   = SLIDE_W - RIGHT_X - 0.25

    # ── Panel A — annotation schematic ───────────────────────────────────
    _panel_label(slide, "A", 0.25, BODY_TOP)
    _txt(slide, "Annotation strategies: sampling approach and dataset statistics",
         0.55, BODY_TOP, LEFT_W - 0.30, 0.26,
         size=8.5, italic=True, color="#555555")

    panel_a_h = MID_Y - BODY_TOP - 0.38
    _draw_panel_a(slide, 0.30, BODY_TOP + 0.26, LEFT_W - 0.05, panel_a_h,
                  b1_stats, b2_stats)

    # ── Panel B — cross-eval heatmap ─────────────────────────────────────
    _panel_label(slide, "B", RIGHT_X, BODY_TOP)
    _txt(slide, "Cross-evaluation: train batch × test batch (all labels, 5-fold CV)",
         RIGHT_X + 0.30, BODY_TOP, RIGHT_W - 0.30, 0.26,
         size=8.5, italic=True, color="#555555")

    img_b = _fig_panel_b(df_cross_cp, df_cross_il)
    _paste_pil(slide, img_b, RIGHT_X, BODY_TOP + 0.26,
               RIGHT_W, MID_Y - BODY_TOP - 0.38)

    # ── Horizontal divider ────────────────────────────────────────────────
    _rect(slide, 0.30, MID_Y - 0.01, SLIDE_W - 0.60, 0.018, "#dddddd")

    # ── Panel C — label-efficiency curves ────────────────────────────────
    BOT_TOP = MID_Y + 0.05
    BOT_H   = SLIDE_H - BOT_TOP - 0.15
    C_W     = SLIDE_W * 0.53   # wider for the label curves

    _panel_label(slide, "C", 0.25, BOT_TOP)
    _txt(slide, "Label efficiency on B1 test population (all-patch, intended use)",
         0.55, BOT_TOP, C_W - 0.30, 0.26,
         size=8.5, italic=True, color="#555555")

    img_c = _fig_panel_c(df_budget_cp, df_budget_il, test_batch="b1")
    _paste_pil(slide, img_c, 0.30, BOT_TOP + 0.26, C_W - 0.05, BOT_H - 0.26)

    # ── Vertical divider ─────────────────────────────────────────────────
    _rect(slide, C_W + 0.18, BOT_TOP + 0.1, 0.018, BOT_H - 0.15, "#dddddd")

    # ── Panel D — image-efficiency ────────────────────────────────────────
    D_X = C_W + 0.32
    D_W = SLIDE_W - D_X - 0.25

    _panel_label(slide, "D", D_X, BOT_TOP)
    _txt(slide, "Image efficiency (b2v2, budget = all labels)",
         D_X + 0.30, BOT_TOP, D_W - 0.30, 0.26,
         size=8.5, italic=True, color="#555555")

    img_d = _fig_panel_d(df_imgeff)
    _paste_pil(slide, img_d, D_X, BOT_TOP + 0.26, D_W, BOT_H - 0.26)

    # ── Vertical divider (A / B) ─────────────────────────────────────────
    _rect(slide, LEFT_W + 0.20, BODY_TOP + 0.1, 0.018,
          MID_Y - BODY_TOP - 0.15, "#dddddd")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    b1 = pd.read_csv(LAB / "labels_vinc_20260521.csv")
    b2 = pd.read_csv(LAB / "vinc_combined_label_Annabel_20260816.csv")

    df_cross_cp = pd.read_csv(EVDIR / "le_b1b2v2_cross_cp_lgbm.csv")
    df_cross_il = pd.read_csv(EVDIR / "le_b1b2v2_cross_ilastik_lgbm.csv")
    df_budget_cp = pd.read_csv(EVDIR / "le_b1b2v2_budget_cp_lgbm.csv")
    df_budget_il = pd.read_csv(EVDIR / "le_b1b2v2_budget_ilastik_lgbm.csv")
    df_imgeff    = pd.read_csv(EVROOT / "image_efficiency_b1b2v2.csv")

    prs = Presentation()
    prs.slide_width  = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    build_slide(prs, b1, b2, df_cross_cp, df_cross_il,
                df_budget_cp, df_budget_il, df_imgeff)

    prs.save(OUT)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()

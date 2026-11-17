#!/usr/bin/env python3
"""
make_pptx_fig1_framework.py
============================
Paper Figure 1: Local subcellular representation and the FA test case.

Single slide with four panels:
  A. Full-field images (paxillin / actin / placeholder) with patch-grid overlay
  B. Enlarged patch examples by class (adhesion-assoc, non-adhesion, ambiguous)
  C. Workflow: image → patches → features → classifier → spatial map  [native PPTX shapes]
  D. FA baseline definition: PS32 / ch1 (paxillin) / stride=PS / no rotation  [native text]

Output: results/fig1_framework.pptx
"""
from __future__ import annotations

import csv
import io
import re
import random
from pathlib import Path

import h5py
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
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu
from pptx.enum.shapes import MSO_SHAPE_TYPE

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[1]
DATA = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LAB  = DATA / "labelling"
H5_C = DATA / "ae_results/patches/cio_rb/vinc/vinc_control_label.h5"
OUT  = REPO / "results" / "fig1_framework.pptx"

SLIDE_W, SLIDE_H = 13.33, 7.5   # inches (widescreen 16:9)

# ── Class palette (matching codebase) ─────────────────────────────────────────
FA5_ORDER = [
    "No adhesion", "Nascent Adhesion", "focal complex",
    "focal adhesion", "fibrillar adhesion",
]
FA5_COLORS = {
    "No adhesion":        "#9467bd",
    "Nascent Adhesion":   "#1f77b4",
    "focal complex":      "#ff7f0e",
    "focal adhesion":     "#2ca02c",
    "fibrillar adhesion": "#d62728",
}
FA5_SHORT = {
    "No adhesion":        "No adhesion",
    "Nascent Adhesion":   "Nascent adhesion (NA)",
    "focal complex":      "Focal complex (FC)",
    "focal adhesion":     "Focal adhesion (FA)",
    "fibrillar adhesion": "Fibrillar adhesion",
}

# Three display groups for Panel B
GROUP_ADH    = ["focal adhesion", "focal complex", "fibrillar adhesion"]
GROUP_NONADH = ["No adhesion"]
GROUP_AMBIG  = ["Nascent Adhesion"]

# Panel colours
COL_DARK  = "#1a2a3a"
COL_MID   = "#2c3e50"
COL_LIGHT = "#ecf0f1"
COL_ACCENT = "#2980b9"


# ── PPTX helpers ──────────────────────────────────────────────────────────────

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


def _txt_multiline(slide, lines, left, top, width, height,
                   size=9, color="#333333", line_bold=None):
    """Add a text box with multiple paragraphs. line_bold: list of bool per line."""
    tb = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = _hex2rgb(color)
        if line_bold and i < len(line_bold):
            run.font.bold = line_bold[i]


def _rect(slide, left, top, width, height, fill_hex,
          line_hex=None, line_pt=None, radius=None):
    """Add a filled rectangle (optionally rounded)."""
    from pptx.util import Pt as _Pt
    if radius:
        # rounded rectangle = MSO_SHAPE_TYPE 5 … easier via add_shape with shape_id
        shape = slide.shapes.add_shape(
            5,  # MSO_SHAPE_TYPE.ROUNDED_RECTANGLE
            Inches(left), Inches(top), Inches(width), Inches(height))
        shape.adjustments[0] = radius
    else:
        shape = slide.shapes.add_shape(
            1,  # MSO_SHAPE_TYPE.RECTANGLE
            Inches(left), Inches(top), Inches(width), Inches(height))
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
                   font_size=9, font_color="#333333", bold=False,
                   align=PP_ALIGN.CENTER, radius=None):
    """Add labelled box (for workflow nodes)."""
    from pptx.util import Pt as _Pt
    if radius:
        shape = slide.shapes.add_shape(
            5, Inches(left), Inches(top), Inches(width), Inches(height))
        shape.adjustments[0] = radius
    else:
        shape = slide.shapes.add_shape(
            1, Inches(left), Inches(top), Inches(width), Inches(height))
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


def _arrow(slide, x1, y1, x2, y2, color="#555555", line_pt=1.2):
    """Add a connector line with arrowhead from (x1,y1) to (x2,y2) in inches."""
    from pptx.util import Pt as _Pt
    from pptx.oxml import parse_xml
    # Use add_connector
    connector = slide.shapes.add_connector(
        1,  # MSO_CONNECTOR_TYPE.STRAIGHT
        Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    connector.line.color.rgb = _hex2rgb(color)
    connector.line.width = _Pt(line_pt)
    # Add arrowhead via xml
    sp_tree = connector._element
    ln = sp_tree.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}ln")
    if ln is None:
        ln = sp_tree.find(".//{http://schemas.openxmlformats.org/drawingml/2006/main}spPr")
    # The connector's line element gets an endArrow via XML
    cxnSp = connector._element
    spPr  = cxnSp.find(qn("p:spPr"))
    if spPr is None:
        spPr = cxnSp.find(qn("p:nvCxnSpPr"))
    # python-pptx exposes line through .line attribute
    connector.line.color.rgb = _hex2rgb(color)
    connector.line.width = _Pt(line_pt)
    return connector


def _paste_pil(slide, img: Image.Image, left, top, width, height):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    slide.shapes.add_picture(buf, Inches(left), Inches(top), Inches(width), Inches(height))


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    plt.close(fig)
    return Image.open(buf).copy()


def _panel_label(slide, letter, left, top, size=14):
    _txt(slide, letter, left, top, 0.28, 0.32,
         size=size, bold=True, color=COL_DARK)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data():
    with h5py.File(H5_C, "r") as f:
        imgs_pax  = f["images/raw"][:]    # (50, 1024, 1024) — paxillin
        imgs_act  = f["images/act"][:]    # (50, 1024, 1024) — actin
        patches   = f["patches/raw"][:]   # (N, 32, 32)
        meta_raw  = f["meta/csv"][()].decode()
    meta_rows = list(csv.DictReader(io.StringIO(meta_raw)))
    fn_to_idx = {r["filename"]: i for i, r in enumerate(meta_rows)}

    lab_df  = pd.read_csv(LAB / "vinc_combined_label_Annabel_20260816.csv")
    lab_dict = dict(zip(lab_df["filename"], lab_df["label"]))
    return imgs_pax, imgs_act, patches, meta_rows, fn_to_idx, lab_dict


def parse_fn(fn: str):
    """Return (frame_idx, x_tl, y_tl, ps) from filename like control_f0000x0080y0784ps32.tif."""
    m = re.match(r'(?:control|ycomp)_f(\d+)x(\d+)y(\d+)ps(\d+)', fn)
    if not m:
        return None
    return int(m[1]), int(m[2]), int(m[3]), int(m[4])


def norm_img(arr: np.ndarray) -> np.ndarray:
    """Percentile-clip and normalise image to [0, 1]."""
    p1, p99 = np.percentile(arr, 1), np.percentile(arr, 99.5)
    return np.clip((arr.astype(float) - p1) / max(p99 - p1, 1e-6), 0, 1)


def norm_patch(patch: np.ndarray) -> np.ndarray:
    p2, p98 = np.percentile(patch, 2), np.percentile(patch, 98)
    return np.clip((patch.astype(float) - p2) / max(p98 - p2, 1e-6), 0, 1)


# ── Panel A — full-field images with patch overlay ────────────────────────────

def _fig_panel_a(imgs_pax, imgs_act, meta_rows, lab_dict,
                 frame_idx: int = 0,
                 crop_tl: tuple = (80, 80), crop_size: int = 448) -> Image.Image:
    """Two-channel insets (paxillin, actin) + placeholder, with patch-colour overlay."""
    cx, cy = crop_tl
    cs = crop_size

    pax_crop = norm_img(imgs_pax[frame_idx][cy:cy+cs, cx:cx+cs])
    act_crop = norm_img(imgs_act[frame_idx][cy:cy+cs, cx:cx+cs])

    # Get labeled patches in this frame/crop
    f_rows = [r for r in meta_rows if int(r["frame_idx"]) == frame_idx]
    patch_info = []
    for r in f_rows:
        parsed = parse_fn(r["filename"])
        if parsed is None:
            continue
        _, x_tl, y_tl, ps = parsed
        label = lab_dict.get(r["filename"])
        # Keep only patches that fall inside the crop
        if x_tl >= cx and y_tl >= cy and x_tl + ps <= cx + cs and y_tl + ps <= cy + cs:
            patch_info.append((x_tl - cx, y_tl - cy, ps, label))

    # Scale patches to actual crop display coordinates (no-op when crop_tl=(0,0), crop_size=1024)

    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.9))
    fig.patch.set_facecolor("white")

    def _overlay(ax, img, title, channel_color):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        # Overlay patches
        for (px, py, ps, lbl) in patch_info:
            ec = FA5_COLORS.get(lbl, "#888888") if lbl else "#cccccc"
            alpha = 0.55 if lbl else 0.18
            lw = 0.9 if lbl else 0.4
            rect = mpatches.Rectangle(
                (px, py), ps, ps,
                linewidth=lw, edgecolor=ec,
                facecolor=ec if lbl else "none",
                alpha=alpha)
            ax.add_patch(rect)
        ax.set_title(title, fontsize=10, fontweight="bold", color="#222222", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(1.0)
            sp.set_edgecolor(channel_color)

    _overlay(axes[0], pax_crop, "Paxillin (ch1, FA marker)", "#2c7bb6")
    _overlay(axes[1], act_crop, "Actin (ch3)", "#d7191c")

    # Microtubule placeholder
    axes[2].set_facecolor("#f5f5f5")
    axes[2].text(0.5, 0.5, "Microtubule\n(example –\nother structure)",
                 ha="center", va="center", fontsize=9, color="#888888",
                 transform=axes[2].transAxes, style="italic")
    axes[2].set_title("Microtubule (placeholder)", fontsize=10,
                      fontweight="bold", color="#888888", pad=4)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    for sp in axes[2].spines.values():
        sp.set_linewidth(1.0)
        sp.set_edgecolor("#bbbbbb")
        sp.set_linestyle("dashed")

    # Shared legend for FA classes
    legend_patches = [
        mpatches.Patch(color=FA5_COLORS[c], label=FA5_SHORT[c])
        for c in FA5_ORDER
    ] + [mpatches.Patch(color="#888888", label="Unlabelled patch")]
    fig.legend(handles=legend_patches, loc="lower center", ncol=3,
               fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.09, 1, 1])
    return _fig_to_pil(fig)


# ── Panel B — patch example grid ──────────────────────────────────────────────

def _sample_patches_by_group(patches, meta_rows, lab_dict, n_per_group=5, seed=42):
    """Return dict group_name → list of (patch_array, label_str)."""
    rng = random.Random(seed)
    by_label: dict[str, list] = {}
    for r in meta_rows:
        lbl = lab_dict.get(r["filename"])
        if lbl:
            by_label.setdefault(lbl, []).append(r)

    groups = {
        "Adhesion-associated": GROUP_ADH,
        "Non-adhesion": GROUP_NONADH,
        "Nascent / ambiguous": GROUP_AMBIG,
    }
    result = {}
    for grp_name, labels in groups.items():
        pool = []
        for lbl in labels:
            pool.extend(by_label.get(lbl, []))
        rng.shuffle(pool)
        sampled = pool[:n_per_group]
        fn_to_idx = {r["filename"]: i for i, r in enumerate(meta_rows)}
        result[grp_name] = [
            (norm_patch(patches[fn_to_idx[r["filename"]]]), lab_dict.get(r["filename"]))
            for r in sampled
            if r["filename"] in fn_to_idx
        ]
    return groups, result


def _fig_panel_b(patches, meta_rows, lab_dict,
                 n_per_group: int = 5) -> Image.Image:
    """Grid of example patches grouped by category."""
    groups_def, group_data = _sample_patches_by_group(
        patches, meta_rows, lab_dict, n_per_group=n_per_group)
    group_names = list(groups_def.keys())
    n_rows = len(group_names)
    n_cols = n_per_group

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 0.72, n_rows * 0.9))
    fig.patch.set_facecolor("white")

    for ri, grp_name in enumerate(group_names):
        items = group_data[grp_name]
        # Representative colour for this group
        if grp_name == "Adhesion-associated":
            grp_color = FA5_COLORS["focal adhesion"]
        elif grp_name == "Non-adhesion":
            grp_color = FA5_COLORS["No adhesion"]
        else:
            grp_color = FA5_COLORS["Nascent Adhesion"]

        for ci in range(n_cols):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(1.2)
                sp.set_edgecolor(grp_color)
            if ci < len(items):
                patch_arr, lbl = items[ci]
                ax.imshow(patch_arr, cmap="gray", vmin=0, vmax=1,
                          interpolation="nearest")
                if ci == 0:
                    ax.set_ylabel(grp_name, fontsize=7.5, rotation=0,
                                  labelpad=56, ha="right", va="center",
                                  fontweight="bold", color=grp_color)
            else:
                ax.set_facecolor("#eeeeee")
                ax.imshow(np.zeros((32, 32)), cmap="gray", vmin=0, vmax=1)

    fig.subplots_adjust(wspace=0.04, hspace=0.1, left=0.22, right=0.99,
                        top=0.95, bottom=0.02)
    return _fig_to_pil(fig)


# ── Panel C — workflow (native PPTX shapes) ────────────────────────────────────

def _draw_workflow(slide, left, top, width, height):
    """Draw the analysis workflow as editable PPTX shapes."""
    # Colour scheme
    C_BOX  = "#dde8f5"   # box fill
    C_LINE = "#2c5f8a"   # box border & arrow colour
    C_TEXT = "#1a2a3a"   # text

    BOX_H = 0.48
    BOX_W = width - 0.1
    bx    = left + 0.05  # box x

    step_labels = [
        "Fluorescence image\n(raw channel)",
        "Extract local patches\n(PS = 32 × 32 px, stride = 32 px)",
    ]
    split_labels = [
        "Hand-crafted features\n(CellProfiler / Ilastik)",
        "Learned features\n(SupCon-AE)",
    ]
    merge_labels = [
        "Classifier\n(linear / LightGBM)",
        "Spatial prediction map\n(per-patch class labels)",
    ]

    y = top

    # Steps 1 & 2
    for label in step_labels:
        _box_with_text(slide, label, bx, y, BOX_W, BOX_H,
                       fill=C_BOX, line=C_LINE, line_pt=0.8,
                       font_size=8, font_color=C_TEXT, bold=False,
                       align=PP_ALIGN.CENTER, radius=0.04)
        arrow_y = y + BOX_H
        y = arrow_y + 0.26
        if label != step_labels[-1]:
            _arrow(slide, bx + BOX_W / 2, arrow_y,
                   bx + BOX_W / 2, y, color=C_LINE, line_pt=1.2)
        else:
            # Fork arrow: from bottom of step 2 to two split boxes
            fork_y = y
            split_w = (BOX_W - 0.1) / 2
            left_bx  = bx
            right_bx = bx + split_w + 0.1

            # Vertical stem
            _arrow(slide,
                   bx + BOX_W / 2, arrow_y,
                   bx + BOX_W / 2, arrow_y + 0.13,
                   color=C_LINE, line_pt=1.0)

            # Horizontal bar (just a thin rectangle)
            mid_y = arrow_y + 0.13
            _rect(slide, left_bx + split_w / 2, mid_y - 0.012,
                  split_w + 0.1, 0.024, C_LINE)

            # Down to left split box
            _arrow(slide,
                   left_bx + split_w / 2, mid_y,
                   left_bx + split_w / 2, fork_y,
                   color=C_LINE, line_pt=1.0)
            # Down to right split box
            _arrow(slide,
                   right_bx + split_w / 2, mid_y,
                   right_bx + split_w / 2, fork_y,
                   color=C_LINE, line_pt=1.0)

            # Two split boxes (feature extraction methods)
            _box_with_text(slide, split_labels[0],
                           left_bx, fork_y, split_w, BOX_H,
                           fill="#fef3cd", line="#e0a800", line_pt=0.8,
                           font_size=7.5, font_color="#5a3800", bold=False,
                           align=PP_ALIGN.CENTER, radius=0.04)
            _box_with_text(slide, split_labels[1],
                           right_bx, fork_y, split_w, BOX_H,
                           fill="#d4edda", line="#28a745", line_pt=0.8,
                           font_size=7.5, font_color="#155724", bold=False,
                           align=PP_ALIGN.CENTER, radius=0.04)

            # Merge arrows
            merge_y = fork_y + BOX_H + 0.26
            cx_mid = bx + BOX_W / 2
            _arrow(slide, left_bx + split_w / 2, fork_y + BOX_H,
                   cx_mid, merge_y, color=C_LINE, line_pt=1.0)
            _arrow(slide, right_bx + split_w / 2, fork_y + BOX_H,
                   cx_mid, merge_y, color=C_LINE, line_pt=1.0)

            y = merge_y

    # Classifier
    _box_with_text(slide, merge_labels[0], bx, y, BOX_W, BOX_H,
                   fill=C_BOX, line=C_LINE, line_pt=0.8,
                   font_size=8, font_color=C_TEXT, bold=False,
                   align=PP_ALIGN.CENTER, radius=0.04)
    arrow_y = y + BOX_H
    y = arrow_y + 0.26
    _arrow(slide, bx + BOX_W / 2, arrow_y,
           bx + BOX_W / 2, y, color=C_LINE, line_pt=1.2)

    # Spatial map
    _box_with_text(slide, merge_labels[1], bx, y, BOX_W, BOX_H,
                   fill="#d5e8f5", line=C_LINE, line_pt=1.0,
                   font_size=8, font_color=C_TEXT, bold=True,
                   align=PP_ALIGN.CENTER, radius=0.04)


# ── Panel D — baseline definition (native PPTX text) ─────────────────────────

def _draw_baseline_def(slide, left, top, width, height):
    """Draw the FA baseline definition as editable PPTX text and table."""
    _txt(slide, "FA Benchmark Baseline — Definition",
         left, top, width, 0.30,
         size=9.5, bold=True, color=COL_DARK)

    rows = [
        ("Patch size (PS)",        "32 × 32 px  (~5.5 × 5.5 µm at 170 nm/px)"),
        ("Channel",                "ch1 — paxillin (FA marker; Dataset 1)"),
        ("Sampling grid",          "Stride = PS (non-overlapping); full image"),
        ("Orientation",            "Original image axes — no rotational alignment"),
        ("Positive class",         "Focal adhesion (FA) + focal complex (FC) + fibrillar"),
        ("Negative class",         "No adhesion"),
        ("Ambiguous",              "Nascent adhesion (NA) — retained in data, flagged"),
    ]

    ROW_H = 0.315
    COL1_W = 1.75
    COL2_W = width - COL1_W

    for ri, (key, val) in enumerate(rows):
        ry = top + 0.35 + ri * ROW_H
        bg = "#f0f4f8" if ri % 2 == 0 else "#ffffff"
        _rect(slide, left, ry, width, ROW_H - 0.02, bg)
        _txt(slide, key, left + 0.06, ry + 0.03, COL1_W - 0.1, ROW_H - 0.05,
             size=8, bold=True, color="#2c5f8a")
        _txt(slide, val, left + COL1_W, ry + 0.03, COL2_W - 0.06, ROW_H - 0.05,
             size=8, bold=False, color="#222222")

    note_y = top + 0.35 + len(rows) * ROW_H + 0.05
    _txt(slide,
         "* PS = 32 is the baseline for detailed benchmarking. Other patch sizes and channels "
         "are evaluated in supplementary comparisons. This definition applies to Dataset 1 (vinculin/paxillin, U2OS).",
         left, note_y, width, 0.5,
         size=7, italic=True, color="#888888")


# ── Main slide builder ─────────────────────────────────────────────────────────

def build_slide(prs, imgs_pax, imgs_act, patches, meta_rows, lab_dict):
    slide = prs.slides.add_slide(prs.slide_layouts[6])   # blank

    # ── Background ──────────────────────────────────────────────────────────
    _rect(slide, 0, 0, SLIDE_W, SLIDE_H, "#ffffff")

    # ── Figure title ────────────────────────────────────────────────────────
    _txt(slide,
         "Figure 1 — Local subcellular representation and the FA test case",
         0.3, 0.06, SLIDE_W - 0.6, 0.35,
         size=13, bold=True, color=COL_DARK, align=PP_ALIGN.LEFT)

    # Thin separator line
    _rect(slide, 0.3, 0.42, SLIDE_W - 0.6, 0.02, "#cccccc")

    # ── Layout constants ─────────────────────────────────────────────────────
    #  Left half  (panels A + B): x = 0.25 … 7.70
    #  Right half (panels C + D): x = 7.85 … 13.10
    LEFT_W = 7.35
    RIGHT_X = 7.80
    RIGHT_W = SLIDE_W - RIGHT_X - 0.25
    BODY_TOP = 0.48

    # ── Panel A ─────────────────────────────────────────────────────────────
    panel_a_h = 3.30
    _panel_label(slide, "A", 0.25, BODY_TOP)
    _txt(slide, "Representative images with local patch grid overlay",
         0.54, BODY_TOP, LEFT_W - 0.3, 0.28,
         size=8.5, bold=False, italic=True, color="#555555")

    img_a = _fig_panel_a(imgs_pax, imgs_act, meta_rows, lab_dict,
                         frame_idx=0, crop_tl=(0, 0), crop_size=1024)
    _paste_pil(slide, img_a, 0.30, BODY_TOP + 0.28, LEFT_W - 0.05, panel_a_h)

    # ── Panel B ─────────────────────────────────────────────────────────────
    b_top = BODY_TOP + panel_a_h + 0.35
    panel_b_h = SLIDE_H - b_top - 0.20
    _panel_label(slide, "B", 0.25, b_top)
    _txt(slide, "Enlarged patch examples by classification category (32 × 32 px each)",
         0.54, b_top, LEFT_W - 0.3, 0.28,
         size=8.5, bold=False, italic=True, color="#555555")

    img_b = _fig_panel_b(patches, meta_rows, lab_dict, n_per_group=6)
    _paste_pil(slide, img_b, 0.30, b_top + 0.28, LEFT_W - 0.05, panel_b_h - 0.28)

    # ── Divider ──────────────────────────────────────────────────────────────
    _rect(slide, 7.65, BODY_TOP + 0.1, 0.02, SLIDE_H - BODY_TOP - 0.3, "#dddddd")

    # ── Panel C ─────────────────────────────────────────────────────────────
    c_top = BODY_TOP
    panel_c_h = 5.05
    _panel_label(slide, "C", RIGHT_X, c_top)
    _txt(slide, "Analysis workflow",
         RIGHT_X + 0.30, c_top, RIGHT_W - 0.30, 0.28,
         size=8.5, bold=False, italic=True, color="#555555")

    _draw_workflow(slide, RIGHT_X + 0.05, c_top + 0.30, RIGHT_W - 0.05, panel_c_h)

    # ── Panel D ─────────────────────────────────────────────────────────────
    d_top = c_top + panel_c_h + 0.20
    _panel_label(slide, "D", RIGHT_X, d_top)
    panel_d_h = SLIDE_H - d_top - 0.15
    _draw_baseline_def(slide, RIGHT_X + 0.05, d_top, RIGHT_W - 0.05, panel_d_h)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print("Loading data …")
    imgs_pax, imgs_act, patches, meta_rows, fn_to_idx, lab_dict = load_data()

    prs = Presentation()
    prs.slide_width  = Inches(SLIDE_W)
    prs.slide_height = Inches(SLIDE_H)

    build_slide(prs, imgs_pax, imgs_act, patches, meta_rows, lab_dict)
    prs.save(OUT)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()

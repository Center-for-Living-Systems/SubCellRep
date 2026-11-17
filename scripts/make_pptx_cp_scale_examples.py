#!/usr/bin/env python3
"""
make_pptx_cp_scale_examples.py  —  cp_benchmark_scale project

Build a 2-slide PPT showing 3 DS1 labelled FA examples at 4 spatial scales.

Slide 1: Title
Slide 2: 3 rows (FA / FC / FiBA) × 4 columns (16×16, 32×32, 64×64, 128×128)
         Each cell shows the same FA patch cropped at that scale.

Output: results/cp_scale_examples.pptx

Usage:
  python scripts/make_pptx_cp_scale_examples.py
  # Optional --scratch-dir to specify where intermediate PNGs are saved
"""
from __future__ import annotations

import argparse
import io
import tempfile
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt, Emu
from pptx.enum.text import PP_ALIGN

# ── Paths ────────────────────────────────────────────────────────────────────
REPO     = Path(__file__).resolve().parents[1]
H5_PATH  = Path("/net/projects/CLS/lding/data/fa_data_analysis"
                "/ae_results/patches/cio/vinc/data.h5")
LAB_PATH = Path("/net/projects/CLS/lding/data/fa_data_analysis"
                "/labelling/labels_vinc_20260521.csv")
OUT_PATH = REPO / "results" / "cp_scale_examples.pptx"

# ── Palette ──────────────────────────────────────────────────────────────────
def rgb(h: str) -> RGBColor:
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return RGBColor(r, g, b)

BG      = rgb("0D0D1A")
PANEL   = rgb("13132E")
COL_HD  = rgb("17174A")
ACC1    = rgb("E94560")   # coral — FA
ACC2    = rgb("53D8FB")   # cyan  — FC
ACC3    = rgb("F5A623")   # amber — FiBA
WHITE   = rgb("FFFFFF")
LGRAY   = rgb("AAAACC")
DGRAY   = rgb("3A3A5A")

# ── Data ─────────────────────────────────────────────────────────────────────
IMAGE_PAD = 64
CMAP      = "inferno"

EXAMPLES = [
    dict(label="Focal Adhesion",     badge="FA",   short="FA",   badge_color=ACC1,
         h5_idx=438,  frame=1,  cx=432, cy=848,
         file="control_f0001x0496y0912ps32.tif"),
    dict(label="Focal Complex",      badge="FC",   short="FC",   badge_color=ACC2,
         h5_idx=983,  frame=2,  cx=848, cy=304,
         file="control_f0002x0912y0368ps32.tif"),
    dict(label="Fibrillar Adhesion", badge="FiBA", short="FiBA", badge_color=ACC3,
         h5_idx=4215, frame=13, cx=464, cy=592,
         file="control_f0013x0528y0656ps32.tif"),
]
SCALES = [16, 32, 64, 128]
CLASS_FULL = ["Focal\nAdhesion", "Focal\nComplex", "Fibrillar\nAdhesion"]

# ── Slide dimensions (13.33" × 7.5" widescreen) ──────────────────────────────
SW = Inches(13.33)
SH = Inches(7.5)

# ── Helpers ───────────────────────────────────────────────────────────────────
def _in(v): return Inches(v)

def add_bg(slide, color: RGBColor):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_rect(slide, x, y, w, h, fill_color=None, line_color=None, line_w=None):
    shape = slide.shapes.add_shape(1, _in(x), _in(y), _in(w), _in(h))
    if fill_color:
        shape.fill.solid(); shape.fill.fore_color.rgb = fill_color
    else:
        shape.fill.background()
    if line_color:
        shape.line.color.rgb = line_color
        if line_w: shape.line.width = int(line_w * 12700)
    else:
        shape.line.fill.background()
    return shape

def add_text(slide, text, x, y, w, h,
             size=12, bold=False, color=WHITE, align=PP_ALIGN.LEFT,
             fontface="Calibri"):
    txb = slide.shapes.add_textbox(_in(x), _in(y), _in(w), _in(h))
    tf  = txb.text_frame
    tf.word_wrap = True
    p   = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = fontface
    return txb

def add_image_file(slide, path, x, y, w, h):
    slide.shapes.add_picture(str(path), _in(x), _in(y), _in(w), _in(h))

def pct_norm(img, lo=1, hi=99):
    a, b = np.percentile(img, lo), np.percentile(img, hi)
    if b <= a: return np.zeros_like(img)
    return np.clip((img.astype(np.float32) - a) / (b - a), 0, 1)

def save_patch_png(arr, path, px=256):
    fig, ax = plt.subplots(figsize=(1,1), dpi=px)
    ax.imshow(pct_norm(arr), cmap=CMAP, interpolation="nearest", vmin=0, vmax=1)
    ax.axis("off")
    fig.subplots_adjust(0,0,1,1)
    fig.savefig(path, dpi=px, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

# ── Patch extraction ──────────────────────────────────────────────────────────
def extract_all_patches(scratch: Path) -> dict[tuple, Path]:
    """
    Returns dict: (short, scale) -> Path to PNG.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    paths = {}
    with h5py.File(H5_PATH, "r") as f:
        patches = f["patches/raw"][:]
        images  = f["images/raw"][:]
    padded = np.pad(images,
                    ((0,0),(IMAGE_PAD,IMAGE_PAD),(IMAGE_PAD,IMAGE_PAD)),
                    mode="edge")
    for ex in EXAMPLES:
        p32 = patches[ex["h5_idx"]]
        fi, cx, cy = ex["frame"], ex["cx"], ex["cy"]
        crops = {
            16:  p32[8:24, 8:24],
            32:  p32,
            64:  padded[fi, cy+IMAGE_PAD-32 : cy+IMAGE_PAD+32,
                            cx+IMAGE_PAD-32 : cx+IMAGE_PAD+32],
            128: padded[fi, cy+IMAGE_PAD-64 : cy+IMAGE_PAD+64,
                            cx+IMAGE_PAD-64 : cx+IMAGE_PAD+64],
        }
        for sc, crop in crops.items():
            p = scratch / f"{ex['short']}_s{sc}.png"
            save_patch_png(crop, p)
            paths[(ex["short"], sc)] = p
    return paths

# ── Slide 1 — Title ───────────────────────────────────────────────────────────
def build_title_slide(prs: Presentation):
    layout = prs.slide_layouts[6]   # blank
    s = prs.slides.add_slide(layout)
    add_bg(s, BG)

    # Decorative circles — ACC1/ACC2 tinted 85% toward BG for subtle glow
    for (x, y, w, tint) in [
        (9.5, -1.0, 4.8, rgb("251016")),   # ACC1 #E94560 tinted dark
        (-0.6, 5.4, 3.0, rgb("152128")),   # ACC2 #53D8FB tinted dark
    ]:
        sh = s.shapes.add_shape(9, _in(x), _in(y), _in(w), _in(w))  # 9=ellipse
        sh.fill.solid(); sh.fill.fore_color.rgb = tint
        sh.line.fill.background()

    add_text(s, "Multi-scale CP Feature Windows", 0.7, 1.8, 10.5, 1.1,
             size=40, bold=True, color=WHITE, align=PP_ALIGN.LEFT)
    add_text(s, "DS1 (vinculin)  ·  3 FA classes  ·  4 spatial scales (16 / 32 / 64 / 128 px)",
             0.7, 3.05, 11.5, 0.5,
             size=16, bold=False, color=ACC2, align=PP_ALIGN.LEFT)

    # Scale badges
    badge_colors = [ACC2, rgb("A0C878"), ACC3, ACC1]
    for i, sc in enumerate(SCALES):
        bx = 0.7 + i * 2.18
        sh = s.shapes.add_shape(5, _in(bx), _in(3.72), _in(1.9), _in(0.36))  # 5=rounded_rect
        sh.fill.solid(); sh.fill.fore_color.rgb = badge_colors[i]
        sh.line.fill.background()
        add_text(s, f"{sc} × {sc} px", bx + 0.02, 3.74, 1.86, 0.32,
                 size=12, bold=True, color=BG, align=PP_ALIGN.CENTER)

    add_text(s, "cp_benchmark_scale  ·  SubCellAE  ·  UChicago CLS",
             0.5, 7.1, 12.0, 0.3,
             size=9, color=LGRAY, align=PP_ALIGN.LEFT)

# ── Slide 2 — Grid ────────────────────────────────────────────────────────────
def build_grid_slide(prs: Presentation, img_paths: dict):
    layout = prs.slide_layouts[6]
    s = prs.slides.add_slide(layout)
    add_bg(s, BG)

    # Layout parameters
    LABEL_X   = 0.22
    LABEL_W   = 1.82
    IMG_AX    = 2.12
    IMG_GAP   = 0.16
    IMG_COLS  = 4
    IMG_AW    = 13.33 - IMG_AX - 0.18
    IMG_W     = (IMG_AW - IMG_GAP * (IMG_COLS - 1)) / IMG_COLS   # ≈ 2.60"
    IMG_H     = IMG_W

    COLHD_Y   = 0.10
    COLHD_H   = 0.34
    ROW0_Y    = 0.52
    ROW_H     = (7.5 - ROW0_Y - 0.10) / 3

    def col_x(c): return IMG_AX + c * (IMG_W + IMG_GAP)
    def row_y(r): return ROW0_Y + r * ROW_H

    # Column-header background strip
    add_rect(s, IMG_AX - 0.04, COLHD_Y, IMG_AW + 0.04, COLHD_H, fill_color=COL_HD)

    # Column headers
    for c, sc in enumerate(SCALES):
        cx = col_x(c)
        add_text(s, f"{sc} × {sc} px", cx, COLHD_Y + 0.04, IMG_W, COLHD_H - 0.08,
                 size=13, bold=True, color=ACC2, align=PP_ALIGN.CENTER)

    badge_clrs = [ACC1, ACC2, ACC3]

    for r, ex in enumerate(EXAMPLES):
        ry = row_y(r)
        # Row panel
        add_rect(s, LABEL_X - 0.04, ry + 0.06,
                 13.33 - LABEL_X + 0.04 - 0.15, ROW_H - 0.12,
                 fill_color=PANEL)

        # Badge
        img_cy = ry + (ROW_H - IMG_H) / 2   # vertical centre of images
        badge_y = img_cy + (IMG_H - 0.28) / 2 - 0.38
        sh = s.shapes.add_shape(5,
            _in(LABEL_X), _in(badge_y), _in(0.62), _in(0.26))
        sh.fill.solid(); sh.fill.fore_color.rgb = badge_clrs[r]
        sh.line.fill.background()
        add_text(s, ex["badge"], LABEL_X + 0.01, badge_y + 0.01, 0.60, 0.24,
                 size=10, bold=True, color=BG, align=PP_ALIGN.CENTER)

        # Class name (2 lines)
        add_text(s, CLASS_FULL[r],
                 LABEL_X, badge_y + 0.32, LABEL_W - 0.05, 0.60,
                 size=11, bold=True, color=WHITE, align=PP_ALIGN.LEFT)

        # Filename
        add_text(s, ex["file"],
                 LABEL_X, badge_y + 0.96, LABEL_W - 0.05, 0.26,
                 size=7, color=LGRAY, align=PP_ALIGN.LEFT)

        # 4 images
        for c, sc in enumerate(SCALES):
            ix = col_x(c)
            iy = img_cy
            path = img_paths[(ex["short"], sc)]
            add_image_file(s, path, ix, iy, IMG_W, IMG_H)
            # Border
            add_rect(s, ix, iy, IMG_W, IMG_H,
                     fill_color=None, line_color=DGRAY, line_w=0.5)

        # Row divider (except last)
        if r < 2:
            ln = s.shapes.add_shape(1, _in(LABEL_X), _in(row_y(r+1)),
                                    _in(13.33 - LABEL_X*2), _in(0.006))
            ln.fill.solid(); ln.fill.fore_color.rgb = DGRAY
            ln.line.fill.background()

    # Footer
    add_text(s,
             "DS1 · vinc · control  |  H5: ae_results/patches/cio/vinc/data.h5  "
             "|  inferno cmap  |  1–99th pct normalisation",
             0.22, 7.22, 12.9, 0.22,
             size=8, color=LGRAY, align=PP_ALIGN.LEFT)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-dir",
                    default="/tmp/cp_scale_pptx_patches",
                    help="Directory for intermediate patch PNGs")
    args = ap.parse_args()

    scratch = Path(args.scratch_dir)
    print("Extracting example patches …")
    img_paths = extract_all_patches(scratch)

    prs = Presentation()
    prs.slide_width  = Inches(13.33)
    prs.slide_height = Inches(7.5)

    print("Building title slide …")
    build_title_slide(prs)
    print("Building grid slide …")
    build_grid_slide(prs, img_paths)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT_PATH))
    print(f"Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()

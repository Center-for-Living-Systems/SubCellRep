#!/usr/bin/env python3
"""
make_b1_panel.py

Grid panel of B1-labeled control frames f0000–f0013 (14 frames, 2 rows × 7 cols).
Background: paxillin (images/raw, ch1).
Patches drawn as semi-transparent colored fills (5 FA classes).

Output: results/control_f0000_f0013_panel.png
"""
from __future__ import annotations
import re
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

DATA  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
H5    = DATA / "ae_results/patches/cio_rb/vinc/vinc_control_label.h5"
B1CSV = DATA / "labelling/vinc_combined_labels.csv"
OUT   = Path(__file__).resolve().parents[1] / "results/control_f0000_f0013_panel.png"

FRAMES     = list(range(14))   # f0000 – f0013
GRID_COLS  = 7
GRID_ROWS  = 2
THUMB_SIZE = 400               # px per frame (before gap)
GAP        = 6                 # px between thumbnails
LABEL_H    = 20                # px for frame label below thumbnail
PS         = 32                # patch size in full-res px
PAD        = 64                # coord padding in filenames
ALPHA      = 0.55              # fill opacity

# Channel: "raw" = paxillin (ch1)
CH_KEY = "raw"

LABEL_COLORS = {
    "No adhesion":        (148,  30, 220),
    "Nascent Adhesion":   ( 30, 100, 240),
    "focal complex":      (240, 130,  20),
    "focal adhesion":     ( 30, 180,  60),
    "fibrillar adhesion": (220,  30,  30),
}
LEGEND_ORDER = ["No adhesion", "Nascent Adhesion",
                "focal complex", "focal adhesion", "fibrillar adhesion"]


def norm_to_uint8(arr: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(arr, [1, 99])
    arr = np.clip(arr, p1, p99)
    if p99 > p1:
        arr = (arr - p1) / (p99 - p1)
    return (arr * 255).astype(np.uint8)


def coords_from_filename(fn: str) -> tuple[int, int]:
    m = re.search(r"x(\d+)y(\d+)ps", fn)
    if m:
        return int(m.group(1)) - PAD, int(m.group(2)) - PAD
    raise ValueError(f"Cannot parse coords from {fn}")


def render_frame(frame_arr: np.ndarray, patches: pd.DataFrame,
                 frame_name: str) -> Image.Image:
    """Render one frame with patch overlays, return as THUMB_SIZE×THUMB_SIZE RGB."""
    H, W = frame_arr.shape
    base  = norm_to_uint8(frame_arr)
    rgb   = np.stack([base, base, base], axis=2)
    img   = Image.fromarray(rgb, "RGB")

    if len(patches):
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        odraw   = ImageDraw.Draw(overlay)
        half    = PS // 2
        for _, r in patches.iterrows():
            cx, cy = int(r["cx"]), int(r["cy"])
            color  = LABEL_COLORS.get(r["label"], (200, 200, 200))
            x0 = max(0, cx - half); y0 = max(0, cy - half)
            x1 = min(W-1, cx+half); y1 = min(H-1, cy+half)
            a  = int(ALPHA * 255)
            odraw.rectangle([x0, y0, x1, y1], fill=(*color, a))
            odraw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=1)
        img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")

    img = img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)

    # Frame label
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    n = len(patches)
    draw.text((6, 4), f"{frame_name}  ({n} patches)", fill=(255,255,180), font=font)

    return img


def add_legend(canvas: Image.Image, x0: int, y0: int) -> None:
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    swatch = 16
    pad    = 6
    lx, ly = x0, y0
    for lbl in LEGEND_ORDER:
        color = LABEL_COLORS[lbl]
        draw.rectangle([lx, ly, lx+swatch, ly+swatch], fill=color)
        draw.text((lx+swatch+5, ly+1), lbl, fill=(240,240,240), font=font)
        ly += swatch + pad


def main():
    # ── Load all frames ───────────────────────────────────────────────────────
    print(f"Loading H5 (channel: images/{CH_KEY}) …", flush=True)
    with h5py.File(H5) as f:
        raw_frames = f[f"images/{CH_KEY}"][:]   # (50, 1024, 1024)
    print(f"  Loaded {raw_frames.shape[0]} frames", flush=True)

    # ── Load B1 labels (control only) ─────────────────────────────────────────
    b1 = pd.read_csv(B1CSV)
    b1 = b1[b1["condition"] == "control"].copy()
    b1 = b1.drop_duplicates(subset=["crop_img_filename", "classification"])

    # Build per-frame patch tables
    frame_patches: dict[int, pd.DataFrame] = {}
    for idx in FRAMES:
        tag   = f"f{idx:04d}"
        sub   = b1[b1["crop_img_filename"].str.contains(tag, na=False)].copy()
        rows  = []
        for _, r in sub.iterrows():
            fn = r["crop_img_filename"]
            try:
                cx, cy = coords_from_filename(fn)
                rows.append({"fn": fn, "label": r["classification"], "cx": cx, "cy": cy})
            except ValueError:
                pass
        frame_patches[idx] = pd.DataFrame(rows)
        print(f"  f{idx:04d}: {len(frame_patches[idx])} patches", flush=True)

    # ── Render thumbnails ─────────────────────────────────────────────────────
    thumbs = []
    for idx in FRAMES:
        frame_arr = raw_frames[idx]
        thumb     = render_frame(frame_arr, frame_patches[idx], f"f{idx:04d}")
        thumbs.append(thumb)

    # ── Assemble grid ─────────────────────────────────────────────────────────
    legend_h = len(LEGEND_ORDER) * (16 + 6) + 20
    panel_w  = GRID_COLS * THUMB_SIZE + (GRID_COLS - 1) * GAP
    panel_h  = GRID_ROWS * THUMB_SIZE + (GRID_ROWS - 1) * GAP + legend_h

    panel = Image.new("RGB", (panel_w, panel_h), (20, 20, 20))

    for i, thumb in enumerate(thumbs):
        row = i // GRID_COLS
        col = i %  GRID_COLS
        x   = col * (THUMB_SIZE + GAP)
        y   = row * (THUMB_SIZE + GAP)
        panel.paste(thumb, (x, y))

    # Legend below the grid
    legend_y = GRID_ROWS * THUMB_SIZE + (GRID_ROWS - 1) * GAP + 8
    add_legend(panel, x0=10, y0=legend_y)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.save(str(OUT), dpi=(150, 150))
    print(f"\nSaved → {OUT}  ({panel.width}×{panel.height} px)", flush=True)


if __name__ == "__main__":
    main()

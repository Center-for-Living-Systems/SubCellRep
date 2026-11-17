#!/usr/bin/env python3
"""
make_f0012_overlay.py

Overlay B1-labeled patches on the paxillin frame for control f0012.
Uses H5 image as background (guarantees coordinate consistency).
Patches are drawn as semi-transparent colored fills.

Color scheme:
  Purple  — No adhesion
  Blue    — Nascent Adhesion
  Orange  — focal complex
  Green   — focal adhesion
  Red     — fibrillar adhesion
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
OUT   = Path(__file__).resolve().parents[1] / "results/control_f0012_labeled.png"

FRAME_IDX = 12
PS = 32
PAD = 64

# Channel key in the H5 file:
#   "raw"  = paxillin (ch1)   <-- use this
#   "act"  = actin    (ch3)
#   "vinc" = vinculin (ch0)
#   "zyx"  = zyxin   (ch2)
CH_KEY = "raw"  # paxillin

LABEL_COLORS = {
    "No adhesion":        (148,  30, 220),   # purple
    "Nascent Adhesion":   ( 30, 100, 240),   # blue
    "focal complex":      (240, 130,  20),   # orange
    "focal adhesion":     ( 30, 180,  60),   # green
    "fibrillar adhesion": (220,  30,  30),   # red
}
ALPHA = 0.55   # fill opacity


def norm_to_uint8(arr: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(arr, [1, 99])
    arr = np.clip(arr, p1, p99)
    if p99 > p1:
        arr = (arr - p1) / (p99 - p1)
    return (arr * 255).astype(np.uint8)


def coords_from_filename(fn: str) -> tuple[int, int]:
    """Extract canvas_cx, canvas_cy from filename like f0012x0688y0304ps32.tif."""
    m = re.search(r"x(\d+)y(\d+)ps", fn)
    if m:
        return int(m.group(1)) - PAD, int(m.group(2)) - PAD
    raise ValueError(f"Cannot parse coords from {fn}")


def main():
    # ── Load H5 background ────────────────────────────────────────────────────
    print(f"Loading H5 (channel: images/{CH_KEY}) ...", flush=True)
    with h5py.File(H5) as f:
        frame = f[f"images/{CH_KEY}"][FRAME_IDX]   # (1024, 1024)
    print(f"  H5 frame shape: {frame.shape}", flush=True)

    # ── Load B1 labels for f0012 ──────────────────────────────────────────────
    b1 = pd.read_csv(B1CSV)
    b1_12 = b1[(b1["condition"] == "control") &
               (b1["crop_img_filename"].str.contains("f0012", na=False))].copy()
    b1_12 = b1_12.drop_duplicates(subset=["crop_img_filename", "classification"])
    print(f"  Labeled patches: {len(b1_12)}", flush=True)

    # Resolve canvas coordinates from filename
    rows = []
    for _, r in b1_12.iterrows():
        fn    = r["crop_img_filename"]
        label = r["classification"]
        cx, cy = coords_from_filename(fn)
        rows.append({"fn": fn, "label": label, "cx": cx, "cy": cy})

    df = pd.DataFrame(rows)
    print(df[["fn","label","cx","cy"]].to_string(index=False), flush=True)

    # ── Build RGB base image ──────────────────────────────────────────────────
    base = norm_to_uint8(frame)
    rgb  = np.stack([base, base, base], axis=2)  # H×W×3 uint8
    img  = Image.fromarray(rgb, "RGB")
    H, W = frame.shape

    # ── Draw patch overlays ───────────────────────────────────────────────────
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    odraw   = ImageDraw.Draw(overlay)

    half = PS // 2
    for _, r in df.iterrows():
        cx, cy   = int(r["cx"]), int(r["cy"])
        color    = LABEL_COLORS.get(r["label"], (200, 200, 200))
        x0, y0   = max(0, cx - half), max(0, cy - half)
        x1, y1   = min(W - 1, cx + half), min(H - 1, cy + half)
        a        = int(ALPHA * 255)
        odraw.rectangle([x0, y0, x1, y1], fill=(*color, a))
        # thin bright border
        odraw.rectangle([x0, y0, x1, y1], outline=(*color, 255), width=1)

    img_rgba = img.convert("RGBA")
    img_rgba = Image.alpha_composite(img_rgba, overlay)
    result   = img_rgba.convert("RGB")

    # ── Legend ───────────────────────────────────────────────────────────────
    draw = ImageDraw.Draw(result)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        font_sm = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except Exception:
        font = font_sm = ImageFont.load_default()

    legend_items = list(LABEL_COLORS.items())
    swatch = 18
    pad    = 8
    lx     = 12
    ly     = 12
    for label, color in legend_items:
        draw.rectangle([lx, ly, lx + swatch, ly + swatch], fill=color)
        draw.text((lx + swatch + 6, ly + 1), label, fill=(255, 255, 255), font=font_sm)
        ly += swatch + pad

    # Frame label
    draw.text((W - 200, H - 28), f"f0012 | B1 control", fill=(200, 200, 200), font=font_sm)

    # ── Save ─────────────────────────────────────────────────────────────────
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(OUT), dpi=(200, 200))
    print(f"\nSaved → {OUT}  ({result.width}×{result.height} px)", flush=True)


if __name__ == "__main__":
    main()

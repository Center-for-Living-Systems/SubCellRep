#!/usr/bin/env python3
"""
Visualise corrected cell segmentation and background pixels for vinc/control frames 0–29.

Colour key:
  Red    — main cell (centre large region, used for inside stats)
  Orange — other cell-like regions excluded from background (border cells, partial cells)
  Green  — true background pixels used for outside-cell stats

Saved to ./tmp/seg_check_vinc_control_NN.png
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as _ndi
from PIL import Image
from skimage.morphology import (binary_closing, binary_opening, disk,
                                remove_small_objects)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import subcellae.dataprep.patch_prep as _pp

# ── Config (mirrors pack_data_h5.py) ────────────────────────────────────────
CZI_DIR = Path(
    "/mnt/p/Annabel/FA-ML/"
    "For-Liya_Data-Sets-that-look-good-and-contain-paxillin/"
    "20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568/Control"
)
PAX_CH = 1
SEG_PARAMS = dict(
    threshold=0.1, close_size=11,
    min_size_initial=3, min_size_post_close=10, min_size_final=30000,
)
N_FRAMES = 30
ALPHA    = 0.50
OUT_DIR  = Path(__file__).parent


# ── Helpers ──────────────────────────────────────────────────────────────────

def _segment_exclude_mask(pax_raw: np.ndarray, min_size_final: int = 1000) -> np.ndarray:
    """All cell-like regions, no centre filter — for background exclusion."""
    img  = _pp._correct_seg_illumination(pax_raw)
    norm = _pp._percentile_stretch(img,
                                   p1=float(np.percentile(img, 1)),
                                   p99=float(np.percentile(img, 99)))
    mask = norm > SEG_PARAMS["threshold"]
    mask = remove_small_objects(mask, min_size=SEG_PARAMS["min_size_initial"], connectivity=1)
    mask = binary_closing(mask, disk(SEG_PARAMS["close_size"]))
    mask = remove_small_objects(mask, min_size=SEG_PARAMS["min_size_post_close"], connectivity=1)
    mask = binary_opening(mask, disk(3))
    mask = _ndi.binary_fill_holes(mask)
    mask = remove_small_objects(mask, min_size=min_size_final, connectivity=1)
    return mask.astype(bool)


def to_uint8(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi == lo:
        return np.zeros_like(arr, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def overlay_3(gray: np.ndarray,
              main_cell: np.ndarray,
              other_cells: np.ndarray,
              true_bg: np.ndarray,
              alpha: float = 0.50) -> np.ndarray:
    """
    gray       : (H, W) uint8
    main_cell  : (H, W) bool — red
    other_cells: (H, W) bool — orange
    true_bg    : (H, W) bool — green
    Returns (H, W, 3) uint8 RGB
    """
    g = gray.astype(np.float32)
    r, gn, b = g.copy(), g.copy(), g.copy()

    def _tint(r_ch, g_ch, b_ch, mask, tr, tg, tb):
        r_ch[mask] = np.clip(alpha * tr + (1 - alpha) * r_ch[mask], 0, 255)
        g_ch[mask] = np.clip(alpha * tg + (1 - alpha) * g_ch[mask], 0, 255)
        b_ch[mask] = np.clip(alpha * tb + (1 - alpha) * b_ch[mask], 0, 255)

    _tint(r, gn, b, main_cell,   255,  30,  30)   # red
    _tint(r, gn, b, other_cells, 255, 160,   0)   # orange
    _tint(r, gn, b, true_bg,      20, 200,  20)   # green

    return np.stack([r, gn, b], axis=-1).astype(np.uint8)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    filenames = _pp.list_image_files(str(CZI_DIR), file_type="czi")
    filenames = filenames[:N_FRAMES]
    print(f"Processing {len(filenames)} CZI files → {OUT_DIR}", flush=True)
    print(f"  Red=main cell  Orange=other cells excluded  Green=true background\n", flush=True)

    for fi, fname in enumerate(filenames):
        raw = _pp._load_raw_squeezed(str(CZI_DIR), fname, "czi")
        pax = _pp._extract_channel(raw, PAX_CH, fname, "czi").astype(np.float32)

        # Main cell mask (centre region, min_size=30000)
        seg = _pp.segment_cell_mask(
            pax,
            threshold=SEG_PARAMS["threshold"],
            close_size=SEG_PARAMS["close_size"],
            min_size_initial=SEG_PARAMS["min_size_initial"],
            min_size_post_close=SEG_PARAMS["min_size_post_close"],
            min_size_final=SEG_PARAMS["min_size_final"],
        ).astype(bool)

        # Broad exclusion mask (all cells, no centre filter, min_size=1000)
        seg_all = _segment_exclude_mask(pax, min_size_final=1000)

        main_cell   = seg
        other_cells = seg_all & ~seg   # cell-like but not the main cell
        true_bg     = ~seg_all         # true background

        # Stats on true background
        bg_vals  = pax[true_bg]
        out_mean = float(np.mean(bg_vals))   if bg_vals.size else float('nan')
        out_med  = float(np.median(bg_vals)) if bg_vals.size else float('nan')

        # Mode: round to nearest 16-bit count and find most frequent value
        if bg_vals.size:
            raw_counts = np.clip(np.round(bg_vals * 65536).astype(int), 0, 65535)
            out_mode = float(np.bincount(raw_counts).argmax()) / 65536.0
        else:
            out_mode = float('nan')

        gray = to_uint8(pax)
        rgb  = overlay_3(gray, main_cell, other_cells, true_bg, alpha=ALPHA)

        out_path = OUT_DIR / f"seg_check_vinc_control_{fi:02d}.png"
        Image.fromarray(rgb).save(str(out_path))
        n_other = int(other_cells.sum())
        scale = 65536
        print(f"  [{fi:02d}] {fname.split('_control-')[1].replace('.czi','')}  "
              f"other-cell px={n_other:7d}  "
              f"mean={out_mean*scale:5.1f}  median={out_med*scale:5.1f}  mode={out_mode*scale:5.1f}  "
              f"(×2^16)", flush=True)


if __name__ == "__main__":
    main()

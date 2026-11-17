#!/usr/bin/env python3
"""
extract_dist_features.py

Extract rotation- and scale-invariant distance-to-cell-boundary features
(d00–d07 + equiv_diam) for all patches in DS1–DS4.

Source: ae_results/patches/cio_rb/{ds}/data.h5
  images/raw   (N_frames, 1024, 1024) — CIO-normalised paxillin images
  meta/csv     — per-patch: filename, condition_name, frame_idx, canvas_cx, canvas_cy

Each frame is segmented once using the same parameters as the patchprep pipeline,
then 8-ray distance features are computed at each patch centre and normalised by
the cell equivalent diameter (scale-invariant). The cyclic shift makes them
rotation-invariant (d00 = direction of minimum distance).

Output: ae_results/features/dist_features/ds{1..4}.csv
  Columns: filename, condition_name, frame_idx, canvas_cx, canvas_cy,
           equiv_diam, d00..d07

Dataset mapping
---------------
  ds1 → vinc (vinculin, DS1)
  ds2 → pfak (pFAK, DS2)
  ds3 → ppax (ppax, DS3)
  ds4 → nih3t3 (NIH3T3, DS4)

Usage
-----
  python scripts/extract_dist_features.py --dataset ds1
  python scripts/extract_dist_features.py --all
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from subcellae.dataprep.patch_prep import (
    distance_to_boundary_features,
    segment_cell_mask,
)

# ---------------------------------------------------------------------------
# Paths

DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
H5_ROOT   = DATA_ROOT / "ae_results/patches/cio_rb"
OUT_ROOT  = Path("/net/spaces/clinic/CLS/lding/data/fa_data_analysis/ae_results/features/dist_features")

DATASETS = {
    "ds1": "vinc",
    "ds2": "pfak",
    "ds3": "ppax",
    "ds4": "nih3t3",
}

# ---------------------------------------------------------------------------
# Segmentation parameters — identical to the cio_rb patchprep configs

SEG_THRESHOLD        = 0.1
SEG_CLOSE_SIZE       = 11
SEG_MIN_SIZE_INITIAL = 3
SEG_MIN_SIZE_POST_CLOSE = 10
SEG_MIN_SIZE_FINAL   = 30_000

N_ORIENTATIONS = 8

META_COLS = ["filename", "condition_name", "frame_idx", "canvas_cx", "canvas_cy"]
DIST_COLS = ["equiv_diam"] + [f"d{k:02d}" for k in range(N_ORIENTATIONS)]

# ---------------------------------------------------------------------------


def _segment_frame(img: np.ndarray) -> np.ndarray:
    """Return binary cell mask for one 1024×1024 CIO-normalised paxillin image."""
    return segment_cell_mask(
        img,
        threshold=SEG_THRESHOLD,
        close_size=SEG_CLOSE_SIZE,
        min_size_initial=SEG_MIN_SIZE_INITIAL,
        min_size_post_close=SEG_MIN_SIZE_POST_CLOSE,
        min_size_final=SEG_MIN_SIZE_FINAL,
    )


def _equiv_diam(mask: np.ndarray) -> float:
    cell_area = float(np.sum(mask > 0))
    return 2.0 * np.sqrt(cell_area / np.pi) if cell_area > 0 else 1.0


def process_dataset(ds_key: str) -> pd.DataFrame:
    ds_name = DATASETS[ds_key]
    h5_path = H5_ROOT / ds_name / "data.h5"
    print(f"\n[{ds_key}] {ds_name} — loading {h5_path}", flush=True)

    with h5py.File(h5_path, "r") as f:
        images   = f["images/raw"][:]          # (N_frames, 1024, 1024) float32
        meta_csv = f["meta/csv"][()].decode()

    meta_df = pd.read_csv(io.StringIO(meta_csv))
    n_patches = len(meta_df)
    n_frames  = images.shape[0]
    print(f"  {n_patches} patches, {n_frames} frames", flush=True)

    # Segment each frame once and cache
    print("  Segmenting frames ...", flush=True)
    segs   = {}   # frame_idx → mask array
    equivs = {}   # frame_idx → equiv_diam
    for fi in range(n_frames):
        if fi % 10 == 0:
            print(f"    frame {fi}/{n_frames}", flush=True)
        mask = _segment_frame(images[fi])
        segs[fi]   = mask
        equivs[fi] = _equiv_diam(mask)

    # Compute distance features for each patch
    print("  Computing distance features ...", flush=True)
    rows = []
    for i, row in enumerate(meta_df.itertuples(index=False)):
        if i % 2000 == 0:
            print(f"    {i}/{n_patches}", flush=True)

        fi   = int(row.frame_idx)
        cx   = int(row.canvas_cx)
        cy   = int(row.canvas_cy)
        mask = segs[fi]
        eq   = equivs[fi]

        dist = distance_to_boundary_features(mask, cy, cx, n_orientations=N_ORIENTATIONS)
        dist_norm = dist / eq

        rec = {
            "filename":       row.filename,
            "condition_name": row.condition_name,
            "frame_idx":      fi,
            "canvas_cx":      cx,
            "canvas_cy":      cy,
            "equiv_diam":     eq,
        }
        for k, val in enumerate(dist_norm):
            rec[f"d{k:02d}"] = float(val)
        rows.append(rec)

    df = pd.DataFrame(rows, columns=META_COLS + DIST_COLS)
    out_path = OUT_ROOT / f"{ds_key}.csv"
    df.to_csv(out_path, index=False)
    print(f"  -> saved {len(df)} rows × {len(DIST_COLS)} dist cols: {out_path}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dataset", choices=list(DATASETS), metavar="DS",
                     help="Single dataset key: ds1 ds2 ds3 ds4")
    grp.add_argument("--all", action="store_true",
                     help="Process all four datasets sequentially")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    targets = list(DATASETS) if args.all else [args.dataset]
    for ds in targets:
        process_dataset(ds)

    print("\nDone.")


if __name__ == "__main__":
    main()

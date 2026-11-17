#!/usr/bin/env python3
"""
tiff_to_png.py

Convert all .tif files under a directory to .png (uint8, no compression).
Each patch is normalised independently using 1–99th percentile so dim patches
are still visible. Output PNGs are written alongside the source TIFFs.

Usage:
  python scripts/tiff_to_png.py [--root eval_results/actin_kmeans_patches]
                                 [--delete_tif]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

DEFAULT_ROOT = (Path(__file__).resolve().parents[1]
                / "eval_results" / "actin_kmeans_patches")


def convert_one(tif_path: Path, delete_tif: bool = False) -> Path:
    arr = tifffile.imread(str(tif_path)).astype(np.float32)

    # Per-patch 1–99 percentile normalisation → uint8
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    arr = np.clip(arr, 0.0, 1.0)
    img_u8 = (arr * 255).round().astype(np.uint8)

    png_path = tif_path.with_suffix(".png")
    Image.fromarray(img_u8, mode="L").save(
        str(png_path), format="PNG", compress_level=0)   # no compression

    if delete_tif:
        tif_path.unlink()

    return png_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",       default=str(DEFAULT_ROOT),
                    help="Root directory to search for .tif files recursively")
    ap.add_argument("--delete_tif", action="store_true",
                    help="Remove source .tif after successful conversion")
    args = ap.parse_args()

    root = Path(args.root)
    tifs = sorted(root.rglob("*.tif"))
    print(f"Found {len(tifs)} .tif files under {root}", flush=True)

    ok = err = 0
    for tif in tifs:
        try:
            png = convert_one(tif, delete_tif=args.delete_tif)
            print(f"  OK  {png.relative_to(root)}", flush=True)
            ok += 1
        except Exception as e:
            print(f"  ERR {tif.relative_to(root)}: {e}", flush=True)
            err += 1

    print(f"\nDone — {ok} converted, {err} errors", flush=True)
    if args.delete_tif:
        print("Source .tif files deleted.", flush=True)


if __name__ == "__main__":
    main()

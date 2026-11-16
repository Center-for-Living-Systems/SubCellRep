#!/usr/bin/env python3
"""
extract_mt_patches.py

Microtubule patch extraction from the Ernest 3-channel TIFFs (mt_tiff/).

  - ch0 -> cell-region segmentation (same pipeline as patch_prep.segment_cell_mask,
           steps 0-8, but ALL components >= MIN_CELL are kept; the
           "keep centre cell" step is dropped because these are large fields)
  - ch1 -> microtubule patches, cell_insideoutside normalisation (no rolling ball)
  - if no cell region >= MIN_CELL px exists in a frame, no patches are produced
  - non-overlapping grid, patch kept if cell-mask coverage >= MASK_RATIO
  - at most MAX_PER_IMG patches per frame per size (random, seeded) to bound disk use

Outputs (per patch size S in 32, 64), under OUT/:
  patches{S}.npy        (N, S, S) float16
  patches{S}_index.csv  idx, name, file, series, y0, x0, cell_cov
  display/              overlay PNGs (ch0+mask, ch1+patch boxes)
  frame_summary.csv     n cell regions / patches per frame

Usage: python scripts/extract_mt_patches.py [--display-every 5]
"""
from __future__ import annotations

import argparse
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import tifffile
from skimage.measure import label
from skimage.morphology import binary_closing, binary_opening, disk, remove_small_objects
from skimage.segmentation import find_boundaries

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from subcellae.dataprep import patch_prep as pp  # noqa: E402

ROOT = Path("/home/lding/lding/data/Ernest")
TIFF = ROOT / "mt_tiff"
OUT = ROOT / "mt_patches"

SEG_CH, MT_CH = 0, 1
SEG_THRESHOLD, CLOSE_SIZE = 0.1, 11
MIN_CELL = 40000
MAX_FRAC = 0.6      # a cell mask covering more of the field than this is a segmentation failure
MIN_CONTRAST = 3.0   # ch0 p99.9 / median below this => no cells in frame
MASK_RATIO = 0.1
MAX_PER_IMG = 100
SIZES = (32, 64)


def segment(ch0: np.ndarray) -> np.ndarray:
    """Cell mask from ch0; keeps every connected region >= MIN_CELL px.

    Empty frames (ch0 is pure camera noise: p99.9/median < MIN_CONTRAST) return an
    empty mask -- otherwise the percentile stretch would amplify noise into a
    "cell" covering the whole field.
    """
    med = float(np.median(ch0))
    if np.percentile(ch0, 99.9) < MIN_CONTRAST * med:
        return np.zeros(ch0.shape, dtype=np.int32)
    img0 = pp._correct_seg_illumination(ch0)
    for hi in (99.0, 99.9):
        # A small cell (<1 % of pixels) puts p99 inside the background noise, so the
        # stretch turns background into foreground (mask covers the field).  If the
        # mask is implausibly large, redo with the stretch top at p99.9.
        img = pp._percentile_stretch(img0, float(np.percentile(img0, 1)), float(np.percentile(img0, hi)))
        m = img > SEG_THRESHOLD
        m = remove_small_objects(m, min_size=3, connectivity=1)
        m = binary_closing(m, disk(CLOSE_SIZE))
        m = remove_small_objects(m, min_size=10, connectivity=1)
        m = binary_opening(m, disk(3))
        m = ndi.binary_fill_holes(m)
        m = remove_small_objects(m, min_size=MIN_CELL, connectivity=1)
        if m.mean() <= MAX_FRAC:
            return label(m)
    return np.zeros(ch0.shape, dtype=np.int32)   # still field-wide: not a real cell mask


def load(path: Path) -> np.ndarray:
    """(C, Y, X) float32; CZI files are scaled by 1/65025 like patch_prep."""
    if path.suffix == ".czi":
        import czifile
        return (czifile.imread(str(path)).squeeze().astype(np.float32) / (255 * 255))
    return tifffile.imread(str(path)).astype(np.float32)


def series_of(tif: Path) -> int:
    try:
        return int(tif.stem.rsplit("_s", 1)[1])
    except (IndexError, ValueError):
        return -1


def process(job):
    tif, display_path = job
    img = load(tif)
    seg = segment(img[SEG_CH])
    n_regions = int(seg.max())
    res = {"tif": tif, "n_regions": n_regions,
           "patches": {s: [] for s in SIZES}, "rows": {s: [] for s in SIZES}}
    boxes = {s: [] for s in SIZES}
    if n_regions > 0:
        norm = pp.normalize_cell_insideoutside(img[MT_CH], seg, scale=5.0)
        inside = (seg > 0).astype(np.float32)
        H, W = norm.shape
        rng = np.random.default_rng(zlib.crc32(tif.stem.encode()))
        for S in SIZES:
            nx, ny = W // S, H // S
            x0, y0 = (W - nx * S) // 2, (H - ny * S) // 2
            cand = []
            for yi in range(ny):
                for xi in range(nx):
                    y, x = y0 + yi * S, x0 + xi * S
                    cov = float(inside[y:y + S, x:x + S].mean())
                    if cov >= MASK_RATIO:
                        cand.append((y, x, cov))
            if len(cand) > MAX_PER_IMG:
                cand = [cand[i] for i in sorted(rng.choice(len(cand), MAX_PER_IMG, replace=False))]
            boxes[S] = cand
            for y, x, cov in cand:
                res["patches"][S].append(norm[y:y + S, x:x + S].astype(np.float16))
                res["rows"][S].append({
                    "name": f"{tif.stem}_y{y:04d}x{x:04d}_ps{S}", "file": tif.parent.name,
                    "series": series_of(tif), "y0": y, "x0": x, "cell_cov": cov})
    res["n_patches"] = {s: len(boxes[s]) for s in SIZES}
    if display_path:
        _display(img, seg, boxes, display_path)
    return res


def _display(img, seg, boxes, path):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6.2))
    for a in ax:
        a.axis("off")
    c0 = img[SEG_CH]
    ov = np.zeros(seg.shape + (4,))
    ov[ndi.binary_dilation(find_boundaries(seg > 0), iterations=2)] = (1, 0.1, 0.1, 1)
    ax[0].imshow(c0, cmap="gray", vmax=np.percentile(c0, 99.5))
    ax[0].imshow(ov)
    ax[0].set_title(f"ch{SEG_CH} + cell mask ({int(seg.max())} regions >= {MIN_CELL}px)")
    c1 = img[MT_CH]
    for a, S, col in ((ax[1], 32, "yellow"), (ax[2], 64, "cyan")):
        a.imshow(c1, cmap="gray", vmax=np.percentile(c1, 99.8))
        a.imshow(ov)
        for y, x, _ in boxes.get(S, []):
            a.add_patch(plt.Rectangle((x, y), S, S, fill=False, ec=col, lw=0.6))
        a.set_title(f"ch{MT_CH} + {len(boxes.get(S, []))} patches {S}px")
    fig.suptitle(path.stem, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--display-every", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N frames")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--src", nargs="*", default=None,
                    help="image files/globs (default: mt_tiff/*/*.tif); .czi supported")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--seg-ch", type=int, default=None)
    ap.add_argument("--mt-ch", type=int, default=None)
    ap.add_argument("--max-per-img", type=int, default=None)
    ap.add_argument("--mask-ratio", type=float, default=None, help="min cell-mask coverage of a patch (default 0.1)")
    a = ap.parse_args()
    global OUT, SEG_CH, MT_CH, MAX_PER_IMG, MASK_RATIO
    MASK_RATIO = MASK_RATIO if a.mask_ratio is None else a.mask_ratio
    OUT = a.out or OUT
    SEG_CH = SEG_CH if a.seg_ch is None else a.seg_ch
    MT_CH = MT_CH if a.mt_ch is None else a.mt_ch
    MAX_PER_IMG = MAX_PER_IMG if a.max_per_img is None else a.max_per_img

    if a.src:
        import glob
        tifs = sorted(Path(f) for g in a.src for f in glob.glob(g))
    else:
        tifs = sorted(TIFF.glob("*/*.tif"))
    if a.limit:
        tifs = tifs[::max(1, len(tifs) // a.limit)][:a.limit]
    (OUT / "display").mkdir(parents=True, exist_ok=True)
    jobs = []
    for t in tifs:
        s = series_of(t) if series_of(t) >= 0 else len(jobs)
        jobs.append((t, OUT / "display" / f"{t.stem}.png" if s % a.display_every == 0 else None))

    stacks = {s: [] for s in SIZES}
    rows = {s: [] for s in SIZES}
    summ = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(process, jobs, chunksize=2)):
            for s in SIZES:
                stacks[s].extend(r["patches"][s])
                rows[s].extend(r["rows"][s])
            summ.append({"frame": r["tif"].stem, "n_regions": r["n_regions"],
                         **{f"n_ps{s}": r["n_patches"][s] for s in SIZES}})
            if i % 20 == 0:
                print(f"{i}/{len(jobs)} {r['tif'].stem} regions={r['n_regions']} n={r['n_patches']}", flush=True)
    pd.DataFrame(summ).to_csv(OUT / "frame_summary.csv", index=False)
    for s in SIZES:
        np.save(OUT / f"patches{s}.npy", np.stack(stacks[s]))
        df = pd.DataFrame(rows[s])
        df.insert(0, "idx", range(len(df)))
        df.to_csv(OUT / f"patches{s}_index.csv", index=False)
        print(f"ps{s}: {len(df)} patches")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Compute background pixel mode (×2^16) for all 4 datasets × 2 conditions × 4 channels.

Cell segmentation is done on paxillin (ch1) using the broad exclusion mask
(no centre-region filter, min_size_final=1000) so partial border cells are
excluded from the background.

Output columns:
  ds  cond  file  ch_name  mode(×2^16)  median(×2^16)  n_bg_px
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as _ndi
from skimage.morphology import (binary_closing, binary_opening, disk,
                                remove_small_objects)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import subcellae.dataprep.patch_prep as _pp

# ── Dataset definitions (mirrors run_local_cio_prep.py) ──────────────────────
DATA_SRC = Path(
    "/mnt/p/Annabel/FA-ML/"
    "For-Liya_Data-Sets-that-look-good-and-contain-paxillin"
)

_DS_DATE = {
    "vinc":   "20250311_eGFPZyxin488_Phalloidin405_Vinculin(rb)647_paxillin(m)568",
    "pfak":   "20250720_eGFP-Zyxin 488, Phalloidin 405, pFAK (rb) 647, paxillin(m)568/072025",
    "ppax":   "20250721_eGFP-Zyxin 488_Phalloidin405_pPaxy118(rb) 647_Pax(m)568",
    "nih3t3": "20260227_NIH3T3_ZyxinGFP,Phalloidin405,Vinc_rb647,Pax_m555_reduced_size_AH",
}
_DS_COND_FOLDERS = {
    "vinc":   {"control": "Control",   "ycomp": "Ycomp"},
    "pfak":   {"control": "Control",   "ycomp": "Ycomp"},
    "ppax":   {"control": "Control",   "ycomp": "Y-comp"},
    "nih3t3": {"control": "Control",   "ycomp": "YCompound"},
}
_DS_CHANNELS = {
    "vinc":   [(0, "vinc"), (1, "pax"), (2, "zyx"), (3, "act")],
    "pfak":   [(0, "pfak"), (1, "pax"), (2, "zyx"), (3, "act")],
    "ppax":   [(0, "ppax"), (1, "pax"), (2, "zyx"), (3, "act")],
    "nih3t3": [(0, "vinc"), (1, "pax"), (2, "zyx"), (3, "act")],
}

SEG_PARAMS = dict(threshold=0.1, close_size=11,
                  min_size_initial=3, min_size_post_close=10, min_size_final=30000)
SCALE = 65536


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


def bg_mode(vals: np.ndarray) -> float:
    """Most frequent 16-bit count in vals (float [0,1] input)."""
    counts = np.clip(np.round(vals * SCALE).astype(int), 0, 65535)
    return float(np.bincount(counts).argmax())


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    header = f"{'ds':8s} {'cond':8s} {'file':6s}  " + \
             "  ".join(f"{ch:>12s}" for _, ch in [("","mode"), ("","med")] * 4)

    ch_names_header = "  ".join(
        f"{'mode':>6s} {'med':>5s}"
        for _ in range(4)
    )

    all_rows = []

    for ds in ["vinc", "pfak", "ppax", "nih3t3"]:
        channels = _DS_CHANNELS[ds]
        ch_labels = [name for _, name in channels]

        for cond in ["control", "ycomp"]:
            czi_dir = DATA_SRC / _DS_DATE[ds] / _DS_COND_FOLDERS[ds][cond]
            if not czi_dir.exists():
                print(f"  [SKIP] {ds}/{cond}: {czi_dir}", flush=True)
                continue

            filenames = _pp.list_image_files(str(czi_dir), file_type="czi")
            print(f"\n=== {ds}/{cond}: {len(filenames)} files ===", flush=True)
            print(f"  {'file':>4s}  " +
                  "  ".join(f"{name:>13s}" for name in ch_labels), flush=True)
            print(f"  {'':>4s}  " +
                  "  ".join(f"{'mode':>6s} {'med':>5s}" for _ in ch_labels), flush=True)

            for fi, fname in enumerate(filenames):
                raw = _pp._load_raw_squeezed(str(czi_dir), fname, "czi")

                # Broad exclusion mask from paxillin (ch1)
                pax = _pp._extract_channel(raw, 1, fname, "czi").astype(np.float32)
                seg_all  = _segment_exclude_mask(pax, min_size_final=1000)
                true_bg  = ~seg_all

                parts = []
                for ch_idx, ch_name in channels:
                    ch_img = _pp._extract_channel(raw, ch_idx, fname, "czi").astype(np.float32)
                    bg_vals = ch_img[true_bg]
                    mode_v   = bg_mode(bg_vals)  if bg_vals.size else float('nan')
                    median_v = float(np.median(bg_vals) * SCALE) if bg_vals.size else float('nan')
                    parts.append(f"{mode_v:6.0f} {median_v:5.1f}")
                    all_rows.append({
                        "ds": ds, "cond": cond, "fi": fi,
                        "ch": ch_name, "mode": mode_v, "median": median_v,
                    })

                short = fname.split("_control-")[-1].split("_ycomp-")[-1].replace(".czi","")
                print(f"  {short:>4s}  " + "  ".join(parts), flush=True)

    # ── Summary: per-dataset/channel mean mode ───────────────────────────────
    print("\n\n=== Summary: mean mode across all files (×2^16) ===", flush=True)
    import collections
    by_ds_ch: dict = collections.defaultdict(list)
    for r in all_rows:
        by_ds_ch[(r["ds"], r["ch"])].append(r["mode"])

    ds_list = ["vinc", "pfak", "ppax", "nih3t3"]
    ch_list = ["pax", "vinc", "pfak", "ppax", "zyx", "act"]

    # Print table: rows=ds, cols=channel
    present_chs = sorted({r["ch"] for r in all_rows},
                         key=lambda c: ["pax","vinc","pfak","ppax","zyx","act"].index(c)
                         if c in ["pax","vinc","pfak","ppax","zyx","act"] else 99)
    print(f"  {'ds':8s}  " + "  ".join(f"{c:>10s}" for c in present_chs), flush=True)
    for ds in ds_list:
        vals = []
        for ch in present_chs:
            modes = by_ds_ch.get((ds, ch), [])
            vals.append(f"{np.mean(modes):10.1f}" if modes else f"{'—':>10s}")
        print(f"  {ds:8s}  " + "  ".join(vals), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Subset an extract_mt_patches.py output dir to patches with cell_cov >= --min-cov and reuse its
CP/IL features (features_{cp,il}{S}.csv); output goes to a new dir, ready for run_mt_features_clusters.py
(--kinds cp il --reuse-features) to redo quantile transform + KMeans + UMAP on the subset."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("src", type=Path)
ap.add_argument("dst", type=Path)
ap.add_argument("--min-cov", type=float, default=0.8)
ap.add_argument("--sizes", type=int, nargs="+", default=[32, 64])
a = ap.parse_args()
a.dst.mkdir(parents=True, exist_ok=True)
cnt = {}
for S in a.sizes:
    idx = pd.read_csv(a.src / f"patches{S}_index.csv")
    keep = (idx.cell_cov >= a.min_cov).values
    P = np.load(a.src / f"patches{S}.npy", mmap_mode="r")
    np.save(a.dst / f"patches{S}.npy", np.ascontiguousarray(P[keep]))
    new = idx[keep].reset_index(drop=True)
    new["idx"] = range(len(new))
    new.to_csv(a.dst / f"patches{S}_index.csv", index=False)
    for kind in ("cp", "il"):
        f = pd.read_csv(a.src / f"features_{kind}{S}.csv")
        assert (f.name.values == idx.name.values).all()
        f[keep].to_csv(a.dst / f"features_{kind}{S}.csv", index=False)
    cnt[S] = new.name.str.rsplit("_y", n=1).str[0].value_counts()
    print(f"ps{S}: {keep.sum()}/{len(idx)} kept (cov>={a.min_cov})")

# frame summary (patch counts recomputed on the subset) + link to the original overlay PNGs
fs = pd.read_csv(a.src / "frame_summary.csv")
for S in a.sizes:
    fs[f"n_ps{S}"] = fs.frame.map(cnt[S]).fillna(0).astype(int)
fs.to_csv(a.dst / "frame_summary.csv", index=False)
if not (a.dst / "display").exists():
    (a.dst / "display").symlink_to(a.src / "display")

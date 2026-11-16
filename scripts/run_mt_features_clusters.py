#!/usr/bin/env python3
"""
run_mt_features_clusters.py

For the MT patches from extract_mt_patches.py (ps 32 and 64):
  1. CellProfiler-style (50) and ilastik-style (80) features  -> features_{cp,il}{S}.csv
  2. quantile-transform to N(0,1) per feature (heavy-tailed features gave
     singleton clusters with plain z-scoring), KMeans k=10 (seed 42)                           -> clusters_{cp,il}{S}.csv
     (cluster, dist_to_centroid in transformed feature space)
  3. 2-D UMAP of the transformed features                         -> umap_{cp,il}{S}.npy

All outputs go to /home/lding/lding/data/Ernest/mt_patches/.
"""
from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_cp_features as cp  # noqa: E402
import extract_ilastik_features as il  # noqa: E402

OUT = Path("/home/lding/lding/data/Ernest/mt_patches")
K = 10


def _cp(img):
    img = img.astype(np.float32)
    return np.concatenate([cp.intensity_features(img), cp.haralick_features(img)])


def _il(img):
    return il.ilastik_features(img.astype(np.float32))


def _chunk(args):
    fn, arr = args
    f = _cp if fn == "cp" else _il
    return np.stack([f(a) for a in arr])


def feats(kind, patches, workers=24):
    chunks = np.array_split(patches, workers * 4)
    with ProcessPoolExecutor(workers) as ex:
        out = list(ex.map(_chunk, [(kind, c) for c in chunks if len(c)]))
    return np.concatenate(out)


def choose_k(Z, kmin=4, kmax=12, tol=0.02, n_sub=6000, seed=0):
    """Silhouette (subsample) over kmin..kmax.  The curves are nearly flat for these continuous
    patch populations, so the raw argmax is unstable: take the LARGEST k whose silhouette is
    within `tol` of the best (finest granularity not clearly worse).  Returns (k, curve)."""
    from sklearn.metrics import silhouette_score
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(Z), min(n_sub, len(Z)), replace=False)
    rows = []
    for k in range(kmin, kmax + 1):
        km = KMeans(k, random_state=42, n_init=5).fit(Z)
        rows.append({"k": k, "silhouette": silhouette_score(Z[sub], km.labels_[sub]), "inertia": km.inertia_})
    df = pd.DataFrame(rows)
    return int(df.k[df.silhouette >= df.silhouette.max() - tol].max()), df


def main():
    import argparse
    import umap
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', type=Path, default=OUT)
    ap.add_argument('--sizes', type=int, nargs='+', default=[32, 64])
    ap.add_argument('--kinds', nargs='+', default=['cp', 'il'], choices=['cp', 'il', 'ae', 'hc'])
    ap.add_argument('--k', default='10', help="int or 'auto' (silhouette, see choose_k)")
    ap.add_argument('--reuse-features', action='store_true', help='reuse existing features_*.csv (cp/il)')
    a = ap.parse_args()
    OUT = a.dir
    names = {"cp": cp.FEATURE_NAMES, "il": il.FEATURE_NAMES}
    for S in a.sizes:
        idx = pd.read_csv(OUT / f"patches{S}_index.csv")
        patches = None
        for kind in a.kinds:
            fpath = OUT / f"features_{kind}{S}.csv"
            if kind in ("ae", "hc") or (a.reuse_features and fpath.exists()):
                F = pd.read_csv(fpath).drop(columns="name").values.astype(np.float64)
            else:
                patches = np.load(OUT / f"patches{S}.npy") if patches is None else patches
                print(f"ps{S} {kind}: {len(patches)} patches", flush=True)
                F = feats(kind, patches)
                df = pd.DataFrame(F, columns=names[kind])
                df.insert(0, "name", idx["name"])
                df.to_csv(fpath, index=False)

            ok = np.isfinite(F).all(1)
            Z = QuantileTransformer(output_distribution='normal', n_quantiles=1000,
                                    random_state=0).fit_transform(np.nan_to_num(F))
            if a.k == 'auto':
                k, curve = choose_k(Z)
                curve.to_csv(OUT / f"kcurve_{kind}{S}.csv", index=False)
            else:
                k = int(a.k)
            km = KMeans(k, random_state=42, n_init=10).fit(Z)
            d = np.linalg.norm(Z - km.cluster_centers_[km.labels_], axis=1)
            pd.DataFrame({"idx": idx["idx"], "cluster": km.labels_, "dist": d,
                          "finite": ok}).to_csv(OUT / f"clusters_{kind}{S}.csv", index=False)
            if not (a.reuse_features and (OUT / f"umap_{kind}{S}.npy").exists()):
                emb = umap.UMAP(n_components=2, random_state=42).fit_transform(Z)
                np.save(OUT / f"umap_{kind}{S}.npy", emb)
            print(f"  {kind}{S}: k={k}; cluster sizes {np.bincount(km.labels_)}", flush=True)


if __name__ == "__main__":
    main()

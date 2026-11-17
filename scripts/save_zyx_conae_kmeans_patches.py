#!/usr/bin/env python3
"""
save_zyx_conae_kmeans_patches.py

KMeans k=10 on ConAE latent features (z_0..z_11) of the zyxin (ch2) model,
then save the patches closest to each cluster centre as PNG files.

Output:
  eval_results/zyx_conae_kmeans_patches/<run>/
    cluster_00/ cluster_00_ex01_<name>.png ...
    ...
    overview.png          (one row per cluster, examples left→right)
    cluster_assignments.csv

Display scaling is shared across all saved patches (global 1–99.5 percentile)
so brightness is comparable between clusters.

Usage:
  python scripts/save_zyx_conae_kmeans_patches.py [--run conae_zyx_n_l1]
                                                   [--n_examples 16]
                                                   [--upscale 8]
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
SWEEP_DIR = DATA_ROOT / "ae_results/protein_sweep"
REPO      = Path(__file__).resolve().parents[1]

N_CLUSTERS   = 10
RANDOM_STATE = 42


def to_uint8(patch: np.ndarray, lo: float, hi: float, upscale: int) -> Image.Image:
    arr = np.clip((patch.astype(np.float32) - lo) / max(hi - lo, 1e-9), 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    return img.resize((img.width * upscale, img.height * upscale), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="conae_zyx_n_l1",
                    help="run directory name under ae_results/protein_sweep")
    ap.add_argument("--n_examples", type=int, default=16)
    ap.add_argument("--upscale", type=int, default=8)
    ap.add_argument("--out_dir", default=str(REPO / "eval_results/zyx_conae_kmeans_patches"))
    args = ap.parse_args()

    out_root = Path(args.out_dir) / args.run
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SWEEP_DIR / args.run / "latents.csv")
    zcols = [c for c in df.columns if c.startswith("z_")]
    X = StandardScaler().fit_transform(df[zcols].values.astype(np.float32))
    print(f"{len(df)} patches, {len(zcols)}-D latent", flush=True)

    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE)
    df["cluster"] = km.fit_predict(X)
    print("cluster sizes:", np.bincount(df["cluster"]).tolist(), flush=True)

    # closest-to-centre examples per cluster
    examples = []
    for k in range(N_CLUSTERS):
        idxs = np.where(df["cluster"].values == k)[0]
        d = np.linalg.norm(X[idxs] - km.cluster_centers_[k], axis=1)
        examples.append(idxs[np.argsort(d)[:args.n_examples]])

    patches = [[tifffile.imread(df["filepath"].iloc[i]).astype(np.float32).squeeze()
                for i in idxs] for idxs in examples]
    allv = np.concatenate([p.ravel() for row in patches for p in row])
    lo, hi = np.percentile(allv, [1, 99.5])
    print(f"display range {lo:.4f} – {hi:.4f}", flush=True)

    rows = []
    for k, (idxs, row) in enumerate(zip(examples, patches)):
        cdir = out_root / f"cluster_{k:02d}"
        cdir.mkdir(exist_ok=True)
        imgs = []
        for rank, (i, p) in enumerate(zip(idxs, row)):
            img = to_uint8(p, lo, hi, args.upscale)
            img.save(cdir / f"cluster_{k:02d}_ex{rank+1:02d}_{Path(df['filename'].iloc[i]).stem}.png")
            imgs.append(img)
        rows.append(imgs)
        print(f"cluster {k:02d} (n={int((df['cluster']==k).sum())}): saved {len(imgs)}", flush=True)

    # overview grid
    s, gap = rows[0][0].width, 4
    W = args.n_examples * (s + gap)
    H = N_CLUSTERS * (s + gap)
    canvas = Image.new("L", (W, H), 255)
    for k, imgs in enumerate(rows):
        for j, img in enumerate(imgs):
            canvas.paste(img, (j * (s + gap), k * (s + gap)))
    canvas.save(out_root / "overview.png")

    df[["filename", "condition_name", "cluster"]].to_csv(out_root / "cluster_assignments.csv", index=False)
    print(f"Done → {out_root}", flush=True)


if __name__ == "__main__":
    main()

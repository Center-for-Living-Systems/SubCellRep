#!/usr/bin/env python3
"""HDBSCAN (density-based, noise allowed) on three representations of the same patches:
CP features (quantile->PCA), the 2-D UMAP of CP, and AE latents.  Per representation: UMAP coloured
by cluster (noise grey), patches closest to each core, and random members.
Usage: make_pptx_hdbscan_compare.py SRC_DIR(full) SUBSET_DIR SIZE OUT.pptx [min_cluster_size]"""
import math, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import QuantileTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_pptx_mt_patches as M  # noqa

SRC, D, S, out = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
MCS = int(sys.argv[5]) if len(sys.argv) > 5 else 100
M.D = D
P = np.load(D / f"patches{S}.npy", mmap_mode="r")
idx = pd.read_csv(D / f"patches{S}_index.csv")
emb = np.load(D / f"umap_cp{S}.npy")
full = pd.read_csv(SRC / f"patches{S}_index.csv")
keep = (full.cell_cov >= idx.cell_cov.min()).values
assert keep.sum() == len(idx)

cp = pd.read_csv(D / f"features_cp{S}.csv").drop(columns="name").values.astype(np.float64)
Zcp = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0).fit_transform(np.nan_to_num(cp))
ae = pd.read_csv(SRC / f"features_ae{S}.csv").drop(columns="name").values[keep]
reps = {"CP features (quantile + PCA10)": PCA(10, random_state=0).fit_transform(Zcp),
        "UMAP of CP (2-D)": emb,
        "AE latents (z, standardised)": StandardScaler().fit_transform(ae)}

prs = Presentation(); prs.slide_width, prs.slide_height = M.SW, M.SH
summary = []
for name, X in reps.items():
    lab = HDBSCAN(min_cluster_size=MCS, min_samples=20).fit_predict(X)
    ids = [k for k in np.argsort(-np.bincount(lab[lab >= 0])) ][:15] if (lab >= 0).any() else []
    noise = (lab < 0).mean()
    summary.append(f"{name}: {len(set(lab)) - (1 if (lab < 0).any() else 0)} clusters, {noise:.0%} noise")
    s = M.blank(prs); M.header(s, f"HDBSCAN on {name}  (min_cluster_size={MCS})")
    fig, a = plt.subplots(figsize=(8, 6.3))
    a.scatter(*emb[lab < 0].T, c="lightgrey", s=2, alpha=0.5)
    a.scatter(*emb[lab >= 0].T, c=lab[lab >= 0], cmap="tab20", s=3, alpha=0.7)
    for k in ids:
        a.text(*np.median(emb[lab == k], 0), str(k), fontsize=12, weight="bold", ha="center")
    a.set_xlabel("UMAP 1"); a.set_ylabel("UMAP 2"); fig.tight_layout(); fig.canvas.draw()
    M.pic(s, Image.fromarray(np.asarray(fig.canvas.buffer_rgba())), Inches(0.3), Inches(0.85), h=Inches(6.4)); plt.close(fig)
    M.text(s, summary[-1] + "\n(grey = noise / unassigned)\n\nsizes: " + ", ".join(f"{k}:{(lab == k).sum()}" for k in ids),
           Inches(8.6), Inches(1.5), Inches(4.5), Inches(5), 14)
    if not ids:
        continue
    for mode in ("closest to core", "RANDOM members"):
        s = M.blank(prs); M.header(s, f"{name}: {M.N_PANEL} {mode} per cluster (largest {min(10, len(ids))})")
        rng = np.random.default_rng(0)
        sel = ids[:10]
        for j, k in enumerate(sel):
            m = np.where(lab == k)[0]
            if mode.startswith("closest"):
                c = np.median(X[m], 0); m = m[np.argsort(np.linalg.norm(X[m] - c, axis=1))[:M.N_PANEL]]
            else:
                m = rng.choice(m, min(M.N_PANEL, len(m)), replace=False)
            M.pic(s, M.panel([P[i] for i in m], f"Cluster {k} (N={(lab == k).sum()})", 64, 3),
                  Inches(0.15) + (j % 5) * Inches(2.6), Inches(0.8) + (j // 5) * Inches(3.35), w=Inches(2.5))
s = M.blank(prs); M.header(s, "Summary"); M.text(s, "\n".join(summary), Inches(0.5), Inches(1.2), Inches(12), Inches(3), 18)
prs.save(out); (D / "_tmp.png").unlink(missing_ok=True); print("\n".join(summary))

#!/usr/bin/env python3
"""KMeans (k=10) in the 2-D UMAP space instead of the feature space; slides: UMAP coloured by the
feature-space vs UMAP-space clusters (+ARI), and the patches closest to each UMAP-space centre.
Usage: make_pptx_umap_space_clusters.py DIR KIND SIZE OUT.pptx [K|auto]
auto: k = argmax silhouette over k=4..20.
SPACE (6th arg): umap (default) or feat = quantile-transformed feature space."""
import sys
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.util import Inches
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_pptx_mt_patches as M  # noqa: E402

D, kind, S, out = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
M.D = D
ARG = sys.argv[5] if len(sys.argv) > 5 else "10"
P = np.load(D / f"patches{S}.npy", mmap_mode="r")
old = pd.read_csv(D / f"clusters_{kind}{S}.csv")
emb = np.load(D / f"umap_{kind}{S}.npy")
SPACE = sys.argv[6] if len(sys.argv) > 6 else "umap"
if SPACE == "feat":
    from sklearn.preprocessing import QuantileTransformer
    X = QuantileTransformer(output_distribution="normal", n_quantiles=1000, random_state=0).fit_transform(
        np.nan_to_num(pd.read_csv(D / f"features_{kind}{S}.csv").drop(columns="name").values.astype(np.float64)))
else:
    X = emb
curve = None
if ARG == "auto":
    from sklearn.metrics import silhouette_score
    sub = np.random.default_rng(0).choice(len(X), min(8000, len(X)), replace=False)
    curve = []
    for k_ in range(4, 21):
        l_ = KMeans(k_, random_state=42, n_init=5).fit_predict(X)
        curve.append((k_, silhouette_score(X[sub], l_[sub])))
    K = max(curve, key=lambda t: t[1])[0]
else:
    K = int(ARG)
km = KMeans(K, random_state=42, n_init=10).fit(X)
lab = km.labels_
dist = np.linalg.norm(X - km.cluster_centers_[lab], axis=1)
pd.DataFrame({"idx": old.idx, "cluster": lab, "dist": dist}).to_csv(D / f"clusters_{SPACE}km{ARG}_{kind}{S}.csv", index=False)
ari = adjusted_rand_score(old.cluster, lab)

prs = Presentation()
prs.slide_width, prs.slide_height = M.SW, M.SH
s = M.blank(prs)
M.header(s, f"{kind.upper()} {S} px: UMAP coloured by feature-space KMeans vs UMAP-space KMeans (k={K})")
fig, ax = plt.subplots(1, 2, figsize=(14, 6.2))
for a, c, t in ((ax[0], old.cluster.values, "KMeans on features (50-D)"), (ax[1], lab, f"KMeans on {SPACE} space (k={K})")):
    a.scatter(emb[:, 0], emb[:, 1], c=c, cmap="tab10", s=3, alpha=0.6)
    for k in np.unique(c):
        a.text(*np.median(emb[c == k], 0), str(k), fontsize=13, weight="bold", ha="center")
    a.set_title(t); a.set_xlabel("UMAP 1"); a.set_ylabel("UMAP 2")
fig.tight_layout(); fig.canvas.draw()
M.pic(s, Image.fromarray(np.asarray(fig.canvas.buffer_rgba())), Inches(0.2), Inches(0.9), w=Inches(12.9))
plt.close(fig)
M.text(s, f"Adjusted Rand index between the two labelings: {ari:.2f}", Inches(0.4), Inches(6.9), Inches(8), Inches(0.4), 14)

s = M.blank(prs)
M.header(s, f"{kind.upper()} {S} px, clusters from {SPACE} space: {M.N_PANEL} patches closest to each centre")
sz = np.bincount(lab, minlength=K)
cols = 5 if K <= 10 else 6
rows = math.ceil(K / cols)
W = min(2.5, 12.9 / cols - 0.1, 6.5 / rows / 1.05)
def place(k):
    return Inches(0.15) + (k % cols) * Inches(W + 0.1), Inches(0.8) + (k // cols) * Inches(W * 1.05 + 0.1)
for k in range(K):
    sub = np.argsort(np.where(lab == k, dist, np.inf))[:M.N_PANEL]
    M.pic(s, M.panel([P[i] for i in sub], f"Cluster {k} (N={sz[k]})", 64, 3),
          *place(k), w=Inches(W))
# random members (not just most central) to show within-cluster spread
s = M.blank(prs)
M.header(s, f"{kind.upper()} {S} px, clusters from {SPACE} space: {M.N_PANEL} RANDOM members per cluster")
rng = np.random.default_rng(0)
for k in range(K):
    ids = rng.choice(np.where(lab == k)[0], M.N_PANEL, replace=False)
    M.pic(s, M.panel([P[i] for i in ids], f"Cluster {k} (N={sz[k]})", 64, 3),
          *place(k), w=Inches(W))
if curve:
    s = M.blank(prs)
    M.header(s, f"Choosing k: silhouette of KMeans in {SPACE} space")
    fig, a = plt.subplots(figsize=(8, 5))
    a.plot(*zip(*curve), "o-"); a.axvline(K, color="r", ls="--"); a.set_xlabel("k"); a.set_ylabel("silhouette")
    a.set_title(f"best k = {K}"); fig.tight_layout(); fig.canvas.draw()
    M.pic(s, Image.fromarray(np.asarray(fig.canvas.buffer_rgba())), Inches(2.5), Inches(1.0), h=Inches(5.8))
    plt.close(fig)
prs.save(out); (D / "_tmp.png").unlink(missing_ok=True)
print(out, "ARI", round(ari, 2), sz)

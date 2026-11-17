#!/usr/bin/env python3
"""
embed_stage2_rebuild.py

Alternative 2D views of the stage-2 rebuild latents, looking for continuous
"stripe"-like progressions instead of UMAP blobs.

Panels per run:
  PCA (PC1/PC2) · LDA (LD1/LD2, fit on train-split labelled) · ordinal axis vs PC1 ·
  UMAP (init=pca, n_neighbors=150, min_dist=0.05) · PHATE · diffusion map · Isomap

LDA / ordinal fits use train-split labelled patches only; val-split labelled
patches are drawn with a black edge so out-of-sample behaviour is visible.
PHATE / diffusion / Isomap run on a subsample (all labelled + random unlabelled,
capped at --max-n) for speed.

Ordinal ranks: [no-ad=0 if present], nascent < focal complex < focal adhesion < fibrillar.
(This ordering is an assumption.)

Output: results/stage2_rebuild/embed_{run}.png, embed_{tag}_coords.csv

Usage:
  python scripts/embed_stage2_rebuild.py --runs stage2 whole
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
for extra in ("/tmp/claude-25565/phate_pkgs",):
    if Path(extra).exists():
        sys.path.insert(0, extra)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
R    = Path("/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run")
OUT  = REPO / "results" / "stage2_rebuild"

ORDER = ["No adhesion", "Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
COLORS = {"No adhesion": "#8e44ad", "Nascent Adhesion": "#1f77b4", "focal complex": "#ff7f0e",
          "focal adhesion": "#2ca02c", "fibrillar adhesion": "#d62728"}


FEAT_CSV = {"cp": "cellprofiler", "ilastik": "ilastik"}
FEAT_DIR = Path("/net/projects/CLS/lding/data/fa_data_analysis/ae_results/features")


def load_run(run: str, feat: str):
    """Return (df with filename/split/label + feature columns, feature column names).

    feat = "latent" -> AE latents z_*; "cp"/"ilastik" -> hand-crafted features for the
    same patches and the same train/val split as the AE run."""
    d = R / f"stage2_rebuild_{run}"
    lat = pd.read_csv(d / "latents.csv")
    if feat == "latent":
        z = [c for c in lat.columns if c.startswith("z_")]
        lat = lat[["filename", "split"] + z]
    elif feat == "conae":      # unsupervised ConAE trained on all patches; split taken from this condition's run
        c = pd.read_csv(R / "stage2_rebuild_conae" / "latents.csv")
        z = [x for x in c.columns if x.startswith("z_")]
        lat = lat[["filename", "split"]].merge(c[["filename"] + z], on="filename", how="left")
    else:
        f = pd.read_csv(FEAT_DIR / FEAT_CSV[feat] / "ds1.csv")
        z = [c for c in f.columns if c != "filename"]
        lat = lat[["filename", "split"]].merge(f, on="filename", how="left")
    lat = lat.merge(pd.read_csv(d / "annotations/labels_used.csv"), on="filename", how="left")
    return lat, z


def diffusion_map(X, n_comp=2, k=30, seed=0):
    """Diffusion map: top non-trivial eigenvectors of the kNN Gaussian Markov operator."""
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, idx = nn.kneighbors(X)
    sig = d[:, -1][:, None]
    w = np.exp(-(d ** 2) / sig ** 2)           # per-point bandwidth = distance to k-th neighbour
    n = len(X)
    A = sp.csr_matrix((w.ravel(), (np.repeat(np.arange(n), k + 1), idx.ravel())), shape=(n, n))
    A = A.maximum(A.T)
    deg = np.asarray(A.sum(1)).ravel()
    Dm = sp.diags(1 / np.sqrt(deg))
    S = Dm @ A @ Dm
    vals, vecs = eigsh(S, k=n_comp + 1, which="LA", v0=np.random.RandomState(seed).rand(n))
    o = np.argsort(-vals)
    vals, vecs = vals[o], vecs[:, o]
    psi = (Dm @ vecs)[:, 1:n_comp + 1] * vals[1:n_comp + 1]      # t=1 diffusion coordinates
    return psi


def scatter(ax, xy, df, title, xl, yl):
    bg = df["label"].isna().values
    ax.scatter(xy[bg, 0], xy[bg, 1], s=2, c="#d9d9d9", alpha=0.5, rasterized=True, linewidths=0)
    for name in ORDER:
        m = (df["label"] == name).values
        if not m.any():
            continue
        tr = m & (df["split"] == "train").values
        va = m & (df["split"] == "val").values
        ax.scatter(xy[tr, 0], xy[tr, 1], s=9, c=COLORS[name], linewidths=0)
        ax.scatter(xy[va, 0], xy[va, 1], s=14, c=COLORS[name], edgecolors="k", linewidths=0.5)
    ax.set_title(title, fontsize=10); ax.set_xlabel(xl, fontsize=8); ax.set_ylabel(yl, fontsize=8)
    ax.tick_params(labelsize=7)


def run_one(run: str, max_n: int, feat: str = "latent"):
    import umap
    lat, z = load_run(run, feat)
    tag = run if feat == "latent" else f"{feat}_{run}"
    X = StandardScaler().fit_transform(lat[z].values)
    labelled = lat["label"].notna().values
    is_tr = (lat["split"] == "train").values & labelled
    classes = [c for c in ORDER if c in set(lat["label"].dropna())]
    print(f"[{run}] n={len(lat)} labelled={labelled.sum()} classes={classes}", flush=True)

    out = lat[["filename", "split", "label"]].copy()

    pca = PCA(n_components=2, random_state=0).fit(X)
    P = pca.transform(X); out["pca1"], out["pca2"] = P[:, 0], P[:, 1]
    ev = pca.explained_variance_ratio_

    lda = LinearDiscriminantAnalysis(n_components=min(2, len(classes) - 1)).fit(X[is_tr], lat.loc[is_tr, "label"])
    L = lda.transform(X)
    if L.shape[1] < 2:
        L = np.c_[L, np.zeros(len(L))]
    out["lda1"], out["lda2"] = L[:, 0], L[:, 1]

    rank = {c: i for i, c in enumerate(classes)}
    rg = Ridge(alpha=10.0).fit(X[is_tr], lat.loc[is_tr, "label"].map(rank).values)
    ordax = rg.predict(X); out["ordinal"] = ordax

    U = umap.UMAP(n_neighbors=150, min_dist=0.05, init="pca", random_state=42).fit_transform(X)
    out["umap_pca1"], out["umap_pca2"] = U[:, 0], U[:, 1]

    rng = np.random.RandomState(0)
    un = np.where(~labelled)[0]
    keep = np.sort(np.r_[np.where(labelled)[0], rng.choice(un, size=max(0, min(len(un), max_n - labelled.sum())), replace=False)])
    Xs, ds = X[keep], lat.iloc[keep].reset_index(drop=True)
    print(f"[{run}] subsample for PHATE/diffusion/Isomap: {len(keep)}", flush=True)

    panels = [
        (P, lat, f"PCA (var {ev[0]:.0%}/{ev[1]:.0%})", "PC1", "PC2"),
        (L, lat, "LDA (train-fit)", "LD1", "LD2"),
        (np.c_[P[:, 0], ordax], lat, "ordinal axis vs PC1", "PC1", "ordinal score"),
        (U, lat, "UMAP init=pca, nn=150", "UMAP1", "UMAP2"),
    ]
    try:
        import phate
        Ph = phate.PHATE(n_components=2, knn=30, random_state=0, verbose=0, n_jobs=4).fit_transform(Xs)
        panels.append((Ph, ds, f"PHATE (n={len(keep)})", "PHATE1", "PHATE2"))
        out.loc[keep, "phate1"], out.loc[keep, "phate2"] = Ph[:, 0], Ph[:, 1]
    except Exception as e:
        print(f"[{run}] PHATE skipped: {e}", flush=True)
    Df = diffusion_map(Xs)
    panels.append((Df, ds, f"diffusion map (n={len(keep)})", "DC1", "DC2"))
    out.loc[keep, "dc1"], out.loc[keep, "dc2"] = Df[:, 0], Df[:, 1]
    Iso = Isomap(n_neighbors=30, n_components=2).fit_transform(Xs)
    panels.append((Iso, ds, f"Isomap (n={len(keep)})", "Iso1", "Iso2"))
    out.loc[keep, "iso1"], out.loc[keep, "iso2"] = Iso[:, 0], Iso[:, 1]

    ncol = 4; nrow = int(np.ceil((len(panels) + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 4.0 * nrow))
    axes = axes.ravel()
    for ax, (xy, df, t, xl, yl) in zip(axes, panels):
        scatter(ax, xy, df, t, xl, yl)
    lax = axes[len(panels)]
    lax.axis("off")
    for i, c in enumerate(classes):
        n = int((lat["label"] == c).sum())
        lax.scatter([], [], c=COLORS[c], s=30, label=f"{c} (n={n})")
    lax.scatter([], [], c="#d9d9d9", s=30, label="unlabelled")
    lax.scatter([], [], facecolors="w", edgecolors="k", s=30, label="black edge = val split")
    lax.legend(loc="center", frameon=False, fontsize=9)
    for ax in axes[len(panels) + 1:]:
        ax.axis("off")
    names = {"latent": "SupCon-AE latents", "conae": "ConAE (unsupervised) latents"}
    fig.suptitle(f"{run}: alternative embeddings of {names.get(feat, feat.upper() + ' features')} (n={len(lat)})", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / f"embed_{tag}.png", dpi=150)
    plt.close(fig)
    out.to_csv(OUT / f"embed_{tag}_coords.csv", index=False)
    print(f"[{run}] saved embed_{tag}.png", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["stage2", "stage2_conf90", "stage2_b1", "stage2_b1_conf90",
                                                  "stage2_b2", "stage2_b2_conf90", "whole"])
    ap.add_argument("--max-n", type=int, default=8000)
    ap.add_argument("--feat", choices=["latent", "conae", "cp", "ilastik"], default="latent")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for r in args.runs:
        run_one(r, args.max_n, args.feat)


if __name__ == "__main__":
    main()

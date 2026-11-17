#!/usr/bin/env python3
"""
replot_stage2_rebuild.py

Re-draw the stage-2 rebuild UMAP / alternative-embedding figures from the saved
coordinate CSVs (no recomputation), using the colour scheme in the embed/umap
scripts. Also writes axis_order_spearman.csv: |Spearman| between each 1-D axis and
the class order (nascent < focal complex < focal adhesion < fibrillar), val-split
labelled patches only.

Usage: python scripts/replot_stage2_rebuild.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_stage2_rebuild import ORDER, COLORS, scatter

OUT  = Path(__file__).resolve().parents[1] / "results" / "stage2_rebuild"
RUNS = ["stage2", "stage2_conf90", "stage2_b1", "stage2_b1_conf90", "stage2_b2", "stage2_b2_conf90", "whole"]
REPS = {"": "SupCon-AE latents", "cp_": "CP features", "ilastik_": "ILASTIK features"}
RANK = {"Nascent Adhesion": 0, "focal complex": 1, "focal adhesion": 2, "fibrillar adhesion": 3}
AXES = {"pca1": "PC1", "lda1": "LD1", "ordinal": "ordinal", "umap_pca1": "UMAP1", "phate1": "PHATE1", "dc1": "DC1", "iso1": "Iso1"}


def umap_fig(run):
    df = pd.read_csv(OUT / f"umap_{run}_coords.csv")
    classes = [c for c in ORDER if c in set(df["label"].dropna())]
    fig, ax = plt.subplots(figsize=(8, 7))
    bg = df["label"].isna()
    ax.scatter(df.loc[bg, "umap1"], df.loc[bg, "umap2"], s=2, c="#d9d9d9", alpha=0.5, rasterized=True, label=f"unlabelled (n={bg.sum()})")
    for name in classes:
        m = df["label"] == name
        ax.scatter(df.loc[m, "umap1"], df.loc[m, "umap2"], s=12, c=COLORS[name], edgecolors="none", label=f"{name} (n={m.sum()})")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title(f"SupCon-AE latents, {'whole set' if run == 'whole' else 'stage-2 pool'}, {run} labels (n={len(df)})")
    ax.legend(markerscale=2, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / f"umap_{run}.png", dpi=200); plt.close(fig)


def embed_fig(run, pfx):
    tag = f"{pfx}{run}"
    df = pd.read_csv(OUT / f"embed_{tag}_coords.csv")
    classes = [c for c in ORDER if c in set(df["label"].dropna())]
    P = [("pca1", "pca2", "PCA", "PC1", "PC2"), ("lda1", "lda2", "LDA (train-fit)", "LD1", "LD2"),
         ("pca1", "ordinal", "ordinal axis vs PC1", "PC1", "ordinal score"),
         ("umap_pca1", "umap_pca2", "UMAP init=pca, nn=150", "UMAP1", "UMAP2"),
         ("phate1", "phate2", "PHATE", "PHATE1", "PHATE2"), ("dc1", "dc2", "diffusion map", "DC1", "DC2"),
         ("iso1", "iso2", "Isomap", "Iso1", "Iso2")]
    fig, axes = plt.subplots(2, 4, figsize=(17.2, 8.0)); axes = axes.ravel()
    for ax, (cx, cy, t, xl, yl) in zip(axes, P):
        d = df[df[cx].notna()].reset_index(drop=True)
        scatter(ax, d[[cx, cy]].values, d, f"{t}" + (f" (n={len(d)})" if len(d) < len(df) else ""), xl, yl)
    lax = axes[7]; lax.axis("off")
    for c in classes:
        lax.scatter([], [], c=COLORS[c], s=30, label=f"{c} (n={(df['label'] == c).sum()})")
    lax.scatter([], [], c="#d9d9d9", s=30, label="unlabelled")
    lax.scatter([], [], facecolors="w", edgecolors="k", s=30, label="black edge = val split")
    lax.legend(loc="center", frameon=False, fontsize=9)
    fig.suptitle(f"{run}: alternative embeddings of {REPS[pfx]} (n={len(df)})", fontsize=13)
    fig.tight_layout(); fig.savefig(OUT / f"embed_{tag}.png", dpi=150); plt.close(fig)


def spearman_rows(run, pfx):
    df = pd.read_csv(OUT / f"embed_{pfx}{run}_coords.csv")
    v = df[(df["split"] == "val") & df["label"].isin(RANK)]
    r = {"rep": REPS[pfx], "run": run, "n_val": len(v)}
    for col, name in AXES.items():
        d = v[v[col].notna()]
        r[name] = abs(spearmanr(d[col], d["label"].map(RANK))[0]) if len(d) > 5 else np.nan
    return r


if __name__ == "__main__":
    rows = []
    for run in RUNS:
        umap_fig(run)
        for pfx in REPS:
            embed_fig(run, pfx); rows.append(spearman_rows(run, pfx))
    t = pd.DataFrame(rows); t.to_csv(OUT / "axis_order_spearman.csv", index=False)
    print(t.round(2).to_string(index=False))

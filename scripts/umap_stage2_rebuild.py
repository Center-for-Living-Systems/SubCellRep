#!/usr/bin/env python3
"""
umap_stage2_rebuild.py

UMAP of the latents from run_stage2_rebuild_ae.py. Labelled patches are coloured
by class; unlabelled patches are grey background.

  stage2 : colours = 4 FA classes
  whole  : colours = No adhesion + 4 FA classes

Output: results/stage2_rebuild/umap_{mode}.png  (+ umap_{mode}_coords.csv)

Usage:
  python scripts/umap_stage2_rebuild.py --mode stage2
  python scripts/umap_stage2_rebuild.py --mode whole
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

REPO    = Path(__file__).resolve().parents[1]
DATA    = Path("/net/projects/CLS/lding/data/fa_data_analysis")
PREP    = DATA / "labelling" / "stage2_rebuild"
RUN_DIR = DATA / "ae_results" / "contrastive_run"
OUT     = REPO / "results" / "stage2_rebuild"

FA4 = ["Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
COLORS = {"No adhesion": "#8e44ad", "Nascent Adhesion": "#1f77b4", "focal complex": "#ff7f0e",
          "focal adhesion": "#2ca02c", "fibrillar adhesion": "#d62728"}
ORDER = {"stage2": FA4, "whole": ["No adhesion"] + FA4, "stage2_b1": FA4, "stage2_b2": FA4}
for _m in ("stage2", "stage2_b1", "stage2_b2"):
    ORDER[_m + "_conf90"] = FA4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(ORDER), required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    lat = pd.read_csv(RUN_DIR / f"stage2_rebuild_{args.mode}" / "latents.csv")
    z = [c for c in lat.columns if c.startswith("z_")]
    used = RUN_DIR / f"stage2_rebuild_{args.mode}" / "annotations" / "labels_used.csv"
    lab = pd.read_csv(used if used.exists() else PREP / "combined_labels.csv")[["filename", "label"]]
    lat = lat.merge(lab, on="filename", how="left")
    lat.loc[~lat["label"].isin(ORDER[args.mode]), "label"] = np.nan

    xy = umap.UMAP(n_neighbors=30, min_dist=0.2, random_state=42).fit_transform(lat[z].values)
    lat["umap1"], lat["umap2"] = xy[:, 0], xy[:, 1]
    lat[["filename", "condition_name", "split", "label", "umap1", "umap2"]].to_csv(
        OUT / f"umap_{args.mode}_coords.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 7))
    bg = lat["label"].isna()
    ax.scatter(lat.loc[bg, "umap1"], lat.loc[bg, "umap2"], s=2, c="#d9d9d9", alpha=0.5,
               rasterized=True, label=f"unlabelled (n={bg.sum()})")
    for name in ORDER[args.mode]:
        m = lat["label"] == name
        ax.scatter(lat.loc[m, "umap1"], lat.loc[m, "umap2"], s=12, c=COLORS[name],
                   edgecolors="none", label=f"{name} (n={m.sum()})")
    ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")
    ax.set_title(f"SupCon-AE latents, {'whole set' if args.mode == 'whole' else 'stage-2 pool'}, {args.mode} labels "
                 f"(n={len(lat)})")
    ax.legend(markerscale=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / f"umap_{args.mode}.png", dpi=200)
    print("saved", OUT / f"umap_{args.mode}.png")


if __name__ == "__main__":
    main()

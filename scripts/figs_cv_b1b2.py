#!/usr/bin/env python3
"""
figs_cv_b1b2.py

Figures for the B1-vs-B2 cross-annotator study (needs results/cv_b1b2/oof.csv from eval_cv_b1b2.py):
  cv_cm_{rep}.png        2x3 confusion matrices: rows = test set (B1, B2), cols = probe training source (B1, B2, B1+B2).
                         AE reps use the encoder trained with the matching label arm (fold labels hidden).
  cross_umap_{run}.png   UMAP of the (stage2_b1 / stage2_b2 / stage2) AE latents coloured by B1 labels (left) and B2 labels (right).
  label_sets.png         B1 vs B2 class composition.
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_common import *

OUT = Path(__file__).resolve().parents[1] / "results" / "cv_b1b2"
RES = Path(__file__).resolve().parents[1] / "results" / "stage2_rebuild"
COLORS = {"Nascent Adhesion": "#1f77b4", "focal complex": "#ff7f0e", "focal adhesion": "#2ca02c", "fibrillar adhesion": "#d62728"}
SHORT = {"Nascent Adhesion": "nascent", "focal complex": "f.complex", "focal adhesion": "f.adhesion", "fibrillar adhesion": "fibrillar"}
SRC_ARM = {"B1": "b1", "B2": "b2", "B1+B2": "b12"}

oof = pd.read_csv(OUT / "oof.csv"); oof = oof[oof.seed == 0]
REPS = {"CP": ("CP", False), "IL": ("IL", False), "CP+IL": ("CP+IL", False), "ConAE": ("ConAE", False),
        "AE": ("AE_", True), "CP+IL+AE": ("CP+IL+AE_", True)}

for key, (rep, arm_dep) in REPS.items():
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 6.6))
    for j, src in enumerate(["B1", "B2", "B1+B2"]):
        name = rep + SRC_ARM[src] if arm_dep else rep
        for i, test in enumerate(["B1", "B2"]):
            ax = axes[i, j]
            d = oof[(oof.rep == name) & (oof.src == src) & (oof.test == test)]
            if d.empty:
                ax.axis("off"); continue
            cm = confusion_matrix(d.y, d.pred, labels=FA4); cmn = cm / cm.sum(1, keepdims=True)
            ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
            for a in range(4):
                for b in range(4):
                    ax.text(b, a, f"{cmn[a, b]:.2f}\n({cm[a, b]})", ha="center", va="center", fontsize=16, color="white" if cmn[a, b] > 0.55 else "black")
            ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[c] for c in FA4], rotation=30, ha="right", fontsize=16)
            ax.set_yticks(range(4)); ax.set_yticklabels([SHORT[c] for c in FA4], fontsize=16)
            kind = "within" if src == test else ("cross" if src != "B1+B2" else "mixed")
            ax.set_title(f"train {src} → test {test} ({kind})  bal.acc {balanced_accuracy_score(d.y, d.pred):.3f}", fontsize=18)
            if j == 0: ax.set_ylabel("true", fontsize=18)
            if i == 1: ax.set_xlabel("predicted", fontsize=18)
    fig.suptitle(f"{key}: out-of-fold confusion matrices (frame-grouped 5-fold CV, seed 0)" + (" · encoder trained with the matching label arm" if arm_dep else ""), fontsize=22)
    fig.tight_layout(); fig.savefig(OUT / f"cv_cm_{key.replace('+', '_')}.png", dpi=150); plt.close(fig)

b1, b2 = load_fa4()
TRAIN_DESC = {"stage2_b1": "encoder supervised with B1 labels", "stage2_b2": "encoder supervised with B2 labels", "stage2": "encoder supervised with B1+B2 labels"}
for run, desc in TRAIN_DESC.items():
    u = pd.read_csv(RES / f"umap_{run}_coords.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4))
    for ax, (nm, lab) in zip(axes, (("B1", b1), ("B2", b2))):
        m = u.merge(lab[["fn", "label"]].rename(columns={"fn": "filename", "label": "lab"}), on="filename", how="left")
        bg = m.lab.isna()
        ax.scatter(m.loc[bg, "umap1"], m.loc[bg, "umap2"], s=2, c="#d9d9d9", alpha=0.5, rasterized=True, linewidths=0)
        for c in FA4:
            s = m.lab == c
            ax.scatter(m.loc[s, "umap1"], m.loc[s, "umap2"], s=10, c=COLORS[c], linewidths=0, label=f"{c} (n={s.sum()})")
        ax.set_title(f"coloured by {nm} labels" + ("  [training labels]" if (run == 'stage2_b1' and nm == 'B1') or (run == 'stage2_b2' and nm == 'B2') else "  [labels not used for training]" if run != 'stage2' else ""), fontsize=10)
        ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2"); ax.legend(markerscale=2, fontsize=8, frameon=False)
    fig.suptitle(f"AE latents, stage-2 pool (n={len(u)}): {desc}", fontsize=12)
    fig.tight_layout(); fig.savefig(OUT / f"cross_umap_{run}.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
x = np.arange(4)
for k, (nm, d, c) in enumerate((("B1 (random sample)", b1, "#0f3d79"), ("B2 (prototype-picked)", b2, "#d65f0e"))):
    ax.bar(x + (k - 0.5) * 0.38, [(d.label == c_).mean() for c_ in FA4], 0.36, label=f"{nm}: {len(d)} FA4 patches, {d.g.nunique()} images", color=c)
ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in FA4]); ax.set_ylabel("share of FA4 labels"); ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=0.3)
fig.savefig(OUT / "label_sets.png", dpi=150); plt.close(fig)
print("figures saved to", OUT)

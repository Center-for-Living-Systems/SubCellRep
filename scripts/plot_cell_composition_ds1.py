#!/usr/bin/env python3
"""
plot_cell_composition_ds1.py — temp illustration: per-cell 5-class composition on DS1 (91 cells).

Takes the two-stage predictions (stage A adhesion gate -> stage B FA4) from lm_fa_pipeline.py and, for each
image (cell), computes the fraction of patches in each of 5 classes (No adhesion / NA / FC / FA / FiBA).
Panels: stacked composition per cell, per-class histograms ctrl vs ycomp, PCA of composition vectors,
and a leave-one-out ctrl-vs-ycomp logistic regression on the composition (AUC).

Usage: python scripts/plot_cell_composition_ds1.py --run lm_fa/il_logreg_ps32
"""
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict, LeaveOneOut

ROOT = Path(__file__).resolve().parents[1] / "eval_results"
CLASSES = ["No adhesion", "Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
COLOR = {"Nascent Adhesion": "#1f77b4", "focal complex": "#ff7f0e", "focal adhesion": "#2ca02c",
         "fibrillar adhesion": "#d62728", "No adhesion": "#9467bd"}
COND = {"control": "#d62728", "ycomp": "#2ca02c"}

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="lm_fa/il_logreg_ps32")
a = ap.parse_args()
d = pd.read_csv(ROOT / a.run / "predictions.csv", usecols=["condition", "frame_idx", "pred_adhesion", "fa_pred"])
d["cls"] = np.where(d.pred_adhesion == 1, d.fa_pred, "No adhesion")
comp = (d.groupby(["condition", "frame_idx"]).cls.value_counts(normalize=True).unstack(fill_value=0)
        .reindex(columns=CLASSES, fill_value=0).reset_index())
n = d.groupby(["condition", "frame_idx"]).size().values
comp["n_patches"] = n
X = comp[CLASSES].values
y = (comp.condition == "ycomp").astype(int).values
print(f"{len(comp)} cells; per condition {comp.condition.value_counts().to_dict()}")
print(comp.groupby("condition")[CLASSES].mean().round(3))

fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.1])

# (a) stacked composition, cells sorted by condition then adhesion fraction
ax = fig.add_subplot(gs[0, :])
o = comp.assign(adh=1 - comp["No adhesion"]).sort_values(["condition", "adh"]).reset_index(drop=True)
bottom = np.zeros(len(o))
for c in CLASSES:
    ax.bar(range(len(o)), o[c], bottom=bottom, color=COLOR[c], width=0.9, label=c)
    bottom += o[c].values
for i, c in enumerate(o.condition):
    ax.add_patch(plt.Rectangle((i - .45, 1.02), .9, .04, color=COND[c], clip_on=False))
nc = (o.condition == "control").sum()
ax.axvline(nc - .5, color="k", lw=1)
ax.set_xlim(-.6, len(o) - .4); ax.set_ylim(0, 1)
ax.set_xticks([]); ax.set_ylabel("fraction of patches")
ax.set_title(f"DS1 — per-cell 5-class composition ({len(o)} cells; top strip: red=control, green=ycomp; sorted by adhesion fraction)", pad=18)
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, -.02), frameon=False)

# (b) per-class histograms
sub = gs[1, :].subgridspec(2, 5, hspace=.55, wspace=.3)
axes = [fig.add_subplot(sub[0, j]) for j in range(5)]
for ax, c in zip(axes, CLASSES):
    hi = comp[c].max()
    b = np.linspace(0, hi * 1.0001, 14)
    for cond in COND:
        ax.hist(comp.loc[comp.condition == cond, c], bins=b, alpha=.55, color=COND[cond], label=cond)
    ax.set_title(c, color=COLOR[c], fontsize=10); ax.set_xlabel("cell fraction")
axes[0].set_ylabel("# cells"); axes[0].legend(frameon=False, fontsize=8)

# (c) PCA of composition + LOO classifier
ax = fig.add_subplot(sub[1, 0:2])
Z = PCA(2).fit(X); P = Z.transform(X)
for cond in COND:
    m = (comp.condition == cond).values
    ax.scatter(P[m, 0], P[m, 1], s=22, color=COND[cond], alpha=.8, label=cond)
ax.set_xlabel(f"PC1 ({Z.explained_variance_ratio_[0]:.0%})"); ax.set_ylabel(f"PC2 ({Z.explained_variance_ratio_[1]:.0%})")
ax.set_title("PCA of 5-d composition", fontsize=10)

p = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), np.log(X + 0.01), y, cv=LeaveOneOut(), method="predict_proba")[:, 1]
auc = roc_auc_score(y, p)
ax = fig.add_subplot(sub[1, 2:5])
for cond in COND:
    ax.hist(p[(comp.condition == cond).values], bins=np.linspace(0, 1, 16), alpha=.55, color=COND[cond], label=cond)
ax.set_xlabel("LOO P(ycomp) from composition"); ax.set_ylabel("# cells")
ax.set_title(f"composition → ctrl vs ycomp (logreg, leave-one-cell-out AUC={auc:.2f})", fontsize=10)
fig.suptitle(f"{a.run}: two-stage predictions aggregated per cell", y=.995, fontsize=12)
out = ROOT / "cell_composition" / f"ds1_cell_composition_{a.run.replace('/', '_')}"
fig.savefig(str(out) + ".png", dpi=140, bbox_inches="tight")
comp.to_csv(str(out) + ".csv", index=False)
print("LOO AUC", round(auc, 3), "->", out)

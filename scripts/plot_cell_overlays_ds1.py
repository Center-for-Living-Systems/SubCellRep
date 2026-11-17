#!/usr/bin/env python3
"""
plot_cell_overlays_ds1.py — example DS1 cells: raw image | 5-class prediction overlay on the 32px grid patches |
5-class composition histogram. Picks cells along PC1 of the composition (extremes + median) per condition.
Output: eval_results/cell_composition/overlays/{control,ycomp}_examples.png (+ examples.csv)
Usage: python scripts/plot_cell_overlays_ds1.py --run lm_fa_grid/il_logreg_ps32
"""
import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1] / "eval_results"
H5 = "/net/projects/CLS/lding/data/fa_data_analysis/ae_results/patches/cio/vinc/data.h5"
CL = ["No adhesion", "Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
HEX = ["#9467bd", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
PS = 32

ap = argparse.ArgumentParser(); ap.add_argument("--run", default="lm_fa_grid/il_logreg_ps32"); a = ap.parse_args()
d = pd.read_csv(ROOT / a.run / "predictions.csv", usecols=["image_idx", "condition", "frame_idx", "grid_ix", "grid_iy", "pred_adhesion", "fa_pred"])
d["k"] = np.where(d.pred_adhesion == 1, d.fa_pred, "No adhesion")
d["k"] = d.k.map({c: i for i, c in enumerate(CL)})
comp = d.groupby("image_idx").k.apply(lambda s: np.bincount(s, minlength=5) / len(s))
comp = pd.DataFrame(np.vstack(comp.values), index=comp.index, columns=CL)
cond = d.groupby("image_idx").condition.first()
pc1 = pd.Series(PCA(1).fit_transform(comp.values)[:, 0], index=comp.index)
if pc1[cond == "control"].mean() < pc1[cond == "ycomp"].mean(): pc1 = -pc1   # high = control-like (mature adhesion)

picks = {}
for c in ["control", "ycomp"]:
    s = pc1[cond == c].sort_values()
    picks[c] = [s.index[-1], s.index[len(s) // 2], s.index[0]]   # most mature, median, least mature
rows = []
with h5py.File(H5, "r") as f:
    for c, idxs in picks.items():
        fig, ax = plt.subplots(3, 3, figsize=(15, 14), gridspec_kw={"width_ratios": [1, 1, .8]})
        for r, i in enumerate(idxs):
            img = f["images/raw"][int(i)].astype(float)
            lo, hi = np.percentile(img, [1, 99.5]); g = np.clip((img - lo) / (hi - lo), 0, 1)
            ov = np.zeros(img.shape + (3,)); al = np.zeros(img.shape)
            for _, p in d[d.image_idx == i].iterrows():
                y0, x0 = int(p.grid_iy) * PS, int(p.grid_ix) * PS
                ov[y0:y0 + PS, x0:x0 + PS] = to_rgb(HEX[int(p.k)]); al[y0:y0 + PS, x0:x0 + PS] = 1
            a0, a1, a2 = ax[r]
            a0.imshow(g, cmap="gray"); a1.imshow(g, cmap="gray")
            a1.imshow(np.dstack([ov, al * .45]))
            for q in (a0, a1): q.axis("off")
            fr = d[d.image_idx == i].frame_idx.iloc[0]
            a0.set_title(f"{c} frame {fr} (image_idx {i}) raw", fontsize=11); a1.set_title("5-class prediction on grid patches", fontsize=11)
            v = comp.loc[i].values
            a2.bar(range(5), v, color=HEX); a2.set_xticks(range(5)); a2.set_xticklabels(["No adh", "NA", "FC", "FA", "FiBA"])
            a2.set_ylim(0, .9); a2.set_ylabel("fraction of patches")
            for x, y in enumerate(v): a2.text(x, y + .01, f"{y:.2f}", ha="center", fontsize=10)
            a2.set_title(["most FA-rich", "median", "least FA-rich"][r] + f" {c} cell", fontsize=11)
            rows.append(dict(condition=c, image_idx=int(i), frame_idx=int(fr), role=["max", "median", "min"][r], pc1=float(pc1[i]), **{k: float(x) for k, x in zip(CL, v)}))
        handles = [plt.Rectangle((0, 0), 1, 1, color=h) for h in HEX]
        fig.legend(handles, CL, ncol=5, loc="lower center", frameon=False, fontsize=12)
        fig.tight_layout(rect=(0, .03, 1, 1))
        out = ROOT / "cell_composition" / "overlays"; out.mkdir(exist_ok=True)
        fig.savefig(out / f"{c}_examples.png", dpi=90); plt.close(fig)
pd.DataFrame(rows).to_csv(ROOT / "cell_composition" / "overlays" / "examples.csv", index=False)
print("done", picks)

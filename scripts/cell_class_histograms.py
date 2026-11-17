#!/usr/bin/env python3
"""
cell_class_histograms.py

DS1 (vinc ctrl + ycomp, 91 cells = 91 frames), GRID patches only (no local-max work).
5-class patch prediction (No adhesion / nascent / focal complex / focal adhesion / fibrillar) from the two-stage CP
logistic-regression models (train_cp_two_stage.py), summarised per cell as a 5-bin histogram.

Outputs (results/cell_histograms/):
  cell_histograms.csv        per cell: condition, n patches, counts, fractions
  overview_stacked.png       stacked 5-class fractions, 91 cells
  mean_histograms.png        mean histogram per condition + per-cell dots
  example_{id}.png           mosaic | prediction overlay on grid patches | histogram
  cell_pca.png               PCA of the 5-d histograms (examples marked)
  cell_classifier.csv        control-vs-ycomp CV AUROC on histogram features
  examples.csv, summary.json
NOTE: models were fit on all B1+B2 labels, so labelled patches (about 11% of all) are training predictions.
"""
import json, re, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np, pandas as pd, tifffile
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("/net/projects/CLS/lding/data/fa_data_analysis")
PRED = DATA / "ae_results" / "cp_two_stage" / "predictions.csv"
PATCH = {c: DATA / "ae_results" / "patches" / "cio" / "vinc" / c / "tiff_patches32_mr10" for c in ("control", "ycomp")}
OUT = Path(__file__).resolve().parents[1] / "results" / "cell_histograms"; OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["No adhesion", "Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
SHORT = ["No adhesion", "nascent", "focal complex", "focal adhesion", "fibrillar"]
COL = ["#8e44ad", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
CCOL = {"control": "#34495e", "ycomp": "#e67e22"}

P = pd.read_csv(PRED)
ex = P.filename.str.extract(r"^(control|ycomp)_f(\d+)x(\d+)y(\d+)ps32\.tif$")
P["cond"], P["frame"], P["x"], P["y"] = ex[0], ex[1].astype(int), ex[2].astype(int), ex[3].astype(int)
P["cell"] = P.cond + "_f" + P.frame.map("{:04d}".format)
P["cls"] = P.final_class.map({c: i for i, c in enumerate(CLASSES)})
assert P.cls.notna().all() and P.cell.nunique() == 91, (P.cell.nunique(), P.cls.isna().sum())
lab_frac = P.label.notna().mean()

# ---- per-cell histograms
cnt = P.groupby(["cell", "cls"]).size().unstack(fill_value=0).reindex(columns=range(5), fill_value=0)
H = pd.DataFrame(index=cnt.index); H["cond"] = [c.split("_")[0] for c in H.index]; H["n"] = cnt.sum(1)
for i, s in enumerate(SHORT): H[f"n_{s}"] = cnt[i]
for i, s in enumerate(SHORT): H[f"f_{s}"] = cnt[i] / H.n
H["f_adhesion_any"] = 1 - H["f_No adhesion"]
H.to_csv(OUT / "cell_histograms.csv"); FR = H[[f"f_{s}" for s in SHORT]].values

# ---- cohort figures
o = H.sort_values(["cond", "f_No adhesion"]).index
fig, ax = plt.subplots(figsize=(15, 4.4)); b = np.zeros(len(o)); xs = np.arange(len(o)) + (np.array([c.startswith("ycomp") for c in o]) * 1.5)
for i in range(5):
    v = H.loc[o, f"f_{SHORT[i]}"].values; ax.bar(xs, v, bottom=b, color=COL[i], width=0.85, label=SHORT[i]); b += v
ax.set_xticks([]); ax.set_ylabel("fraction of grid patches"); ax.set_xlim(-1, xs.max() + 1); ax.set_ylim(0, 1)
nc = (H.cond == "control").sum(); ax.text(nc / 2, 1.02, f"control (n={nc})", ha="center", fontsize=11, color=CCOL["control"], fontweight="bold")
ax.text(nc + 1.5 + (91 - nc) / 2, 1.02, f"ycomp (n={91 - nc})", ha="center", fontsize=11, color=CCOL["ycomp"], fontweight="bold")
ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=10)
ax.set_title("Predicted 5-class histogram of every cell (each column = one cell, sorted by no-adhesion fraction within condition)", fontsize=11, pad=22)
fig.tight_layout(); fig.savefig(OUT / "overview_stacked.png", dpi=150); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
rng = np.random.RandomState(0)
for ax, cond in zip(axes, ("control", "ycomp")):
    d = H[H.cond == cond]; m = d[[f"f_{s}" for s in SHORT]].mean().values; sd = d[[f"f_{s}" for s in SHORT]].std().values
    ax.bar(range(5), m, color=COL, alpha=0.85, yerr=sd, capsize=4, error_kw=dict(ecolor="#333", lw=1))
    for i in range(5): ax.scatter(i + rng.uniform(-0.28, 0.28, len(d)), d[f"f_{SHORT[i]}"], s=9, c="k", alpha=0.35, zorder=3)
    ax.set_xticks(range(5)); ax.set_xticklabels(SHORT, fontsize=9); ax.set_title(f"{cond}: mean ± SD over {len(d)} cells (dots = single cells)", fontsize=10)
axes[0].set_ylabel("fraction of grid patches")
fig.tight_layout(); fig.savefig(OUT / "mean_histograms.png", dpi=150); plt.close(fig)

# ---- mosaic helpers
def mosaic(cell):
    d = P[P.cell == cell]; c0, r0 = (d.x.min() - 16) // 32, (d.y.min() - 16) // 32
    nc_, nr_ = (d.x.max() - 16) // 32 - c0 + 1, (d.y.max() - 16) // 32 - r0 + 1
    img = np.zeros((nr_ * 32, nc_ * 32), np.float32); ov = np.zeros((nr_ * 32, nc_ * 32, 4), np.float32)
    for fn, x, y, k in zip(d.filename, d.x, d.y, d.cls):
        c, r = (x - 16) // 32 - c0, (y - 16) // 32 - r0
        img[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = tifffile.imread(PATCH[d.cond.iloc[0]] / fn)
        ov[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32, :3] = matplotlib.colors.to_rgb(COL[k]); ov[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32, 3] = 1
    return img, ov


def seam(img):
    a = np.abs(np.diff(img, axis=1))[:, 31::32].mean() + np.abs(np.diff(img, axis=0))[31::32].mean(); return a


# orientation check: x as column vs x as row (correct one has much smaller tile-seam jumps)
t_cell = H.index[0]; img_a, _ = mosaic(t_cell)
d = P[P.cell == t_cell]; swapped = P.copy(); P_back = P
P = P.assign(x=P.y, y=P.x); img_b, _ = mosaic(t_cell); P = P_back
inner = np.abs(np.diff(img_a, axis=1)).mean()
orient = dict(seam_x_as_col=float(seam(img_a)), seam_x_as_row=float(seam(img_b)), mean_neighbor_diff=float(inner))
print("orientation check", orient, flush=True)
assert orient["seam_x_as_col"] <= orient["seam_x_as_row"], "x/y orientation looks swapped"

# ---- examples: low / median / high adhesion fraction in each condition
ex_rows = []; rs = {}
for cond in ("control", "ycomp"):
    d = H[H.cond == cond].sort_values("f_adhesion_any")
    for tag, q in (("low", 0.1), ("median", 0.5), ("high", 0.9)):
        cell = d.index[int(round(q * (len(d) - 1)))]; ex_rows.append(dict(cell=cell, tag=tag, **H.loc[cell].to_dict()))
E = pd.DataFrame(ex_rows); E["id"] = [f"E{i + 1}" for i in range(len(E))]; E.to_csv(OUT / "examples.csv", index=False)
for _, r in E.iterrows():
    img, ov = mosaic(r.cell); v = np.percentile(img[img != 0], 99.5) if (img != 0).any() else 1
    fig = plt.figure(figsize=(15, 5.2)); gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.78])
    a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
    g = np.clip(img / v, 0, 1)
    a0.imshow(g, cmap="gray", interpolation="nearest"); a0.set_title("whole cell, mosaic of grid patches (vinculin, CIO)", fontsize=10)
    a1.imshow(g, cmap="gray", interpolation="nearest"); ov2 = ov.copy(); ov2[..., 3] *= 0.5; a1.imshow(ov2, interpolation="nearest")
    for k in range(1, img.shape[1] // 32): a1.axvline(k * 32 - 0.5, color="w", lw=0.15, alpha=0.4)
    for k in range(1, img.shape[0] // 32): a1.axhline(k * 32 - 0.5, color="w", lw=0.15, alpha=0.4)
    a1.set_title("predicted class per 32x32 grid patch", fontsize=10); a1.legend(handles=[Patch(color=COL[i], label=SHORT[i]) for i in range(5)], loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=5, fontsize=7, frameon=False, handlelength=1, columnspacing=0.8)
    for a in (a0, a1): a.set_xticks([]); a.set_yticks([])
    f = [r[f"f_{s}"] for s in SHORT]; n = [int(r[f"n_{s}"]) for s in SHORT]
    a2.bar(range(5), f, color=COL)
    for i in range(5): a2.text(i, f[i] + 0.01, f"{f[i]:.0%}\n({n[i]})", ha="center", fontsize=8)
    a2.set_xticks(range(5)); a2.set_xticklabels(SHORT, rotation=30, ha="right", fontsize=8); a2.set_ylim(0, max(f) * 1.25 + 0.03); a2.set_ylabel("fraction of patches")
    a2.set_title("5-class histogram of this cell", fontsize=10)
    fig.suptitle(f"{r.id}: {r.cell} ({r.cond}) — {r.tag} predicted-adhesion cell, {int(r.n)} grid patches, {r.f_adhesion_any:.0%} predicted adhesion", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f"example_{r.id}.png", dpi=130); plt.close(fig); print("saved", r.id, flush=True)

# ---- cell-level representation
pca = PCA(2).fit(FR - FR.mean(0)); Z = pca.transform(FR - FR.mean(0)); H["pc1"], H["pc2"] = Z[:, 0], Z[:, 1]
fig, ax = plt.subplots(figsize=(7.5, 5.8))
for cond in ("control", "ycomp"):
    m = (H.cond == cond).values; ax.scatter(Z[m, 0], Z[m, 1], s=H.n[m] / 4, c=CCOL[cond], alpha=0.65, label=f"{cond} (n={m.sum()})", edgecolors="none")
for _, r in E.iterrows():
    i = list(H.index).index(r.cell); ax.annotate(r.id, (Z[i, 0], Z[i, 1]), fontsize=9, fontweight="bold", xytext=(4, 4), textcoords="offset points")
    ax.scatter(Z[i, 0], Z[i, 1], s=70, facecolors="none", edgecolors="k", lw=1.2)
ld = pca.components_
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%}): " + ", ".join(f"{SHORT[j]} {ld[0, j]:+.2f}" for j in np.argsort(-np.abs(ld[0]))[:2]), fontsize=8)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%}): " + ", ".join(f"{SHORT[j]} {ld[1, j]:+.2f}" for j in np.argsort(-np.abs(ld[1]))[:2]), fontsize=8)
ax.set_title("Each cell as a 5-d histogram, PCA (marker size = number of patches)", fontsize=10); ax.legend(fontsize=9, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "cell_pca.png", dpi=150); plt.close(fig)
pc_load = {SHORT[j]: [float(ld[0, j]), float(ld[1, j])] for j in range(5)}

# ---- toy cell classifier: control vs ycomp from the histogram (illustration only)
y = (H.cond == "ycomp").astype(int).values; cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=0)
feats = {"cell size only (n patches)": H[["n"]].values, "5-class histogram": FR,
         "5-class histogram + size": np.c_[FR, H.n.values], "adhesion fraction only": H[["f_adhesion_any"]].values}
rows = []
for name, X in feats.items():
    au = []
    for tr, te in cv.split(X, y):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X[tr], y[tr]); au.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    rows.append(dict(features=name, auroc_mean=np.mean(au), auroc_sd=np.std(au)))
CL = pd.DataFrame(rows); CL.to_csv(OUT / "cell_classifier.csv", index=False); print(CL.round(3).to_string(index=False))
H.to_csv(OUT / "cell_histograms.csv")
summ = dict(n_cells=91, n_patches=int(len(P)), labelled_patch_fraction=float(lab_frac), orientation_check=orient, pca_var=pca.explained_variance_ratio_.tolist(), pc_loadings=pc_load,
            mean_fractions={c: H[H.cond == c][[f"f_{s}" for s in SHORT]].mean().round(4).to_dict() for c in ("control", "ycomp")},
            n_by_cond=H.cond.value_counts().to_dict(), median_patches=float(H.n.median()))
(OUT / "summary.json").write_text(json.dumps(summ, indent=2)); print(json.dumps(summ, indent=2))

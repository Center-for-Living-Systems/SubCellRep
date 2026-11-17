#!/usr/bin/env python3
"""
analyze_selected_pca_cp.py

Select the CP features that matter for the rank (ordinal) / classes, then run PCA on only those and test how much
of the signal the leading PCs keep.  Selection is done INSIDE each frame-grouped CV fold from the training labels only
(B1+B2 of the other images); PCA is fit on all patches (unsupervised); tested on held-out B1 (and B2) patches.

Feature sets : all50 | texture39 | top10 univariate |rho| | top20 | lasso (non-zero) | one-per-cluster (corr-clustered, best per cluster)
Baselines    : full-feature ridge / logistic, PLS(2) on all features (supervised projection), LDA(3) on all features
For each set and k in {1,2,3,5} PCs: held-out |Spearman(PC1, rank)|, ridge-on-PCs Spearman & MAE, logistic-on-PCs balanced acc & within-one.
Outputs: results/ordinal_cp/selected_pca.csv, selected_features.csv, selected_pca_scatter.png, selected_pca_bars.png
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_common import *

OUT = Path(__file__).resolve().parents[1] / "results" / "ordinal_cp"
RANK = {c: i for i, c in enumerate(FA4)}
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]; NAMES = ["nascent", "f.complex", "f.adhesion", "fibrillar"]
b1, b2 = load_fa4()
cp = pd.read_csv(FEAT / "cellprofiler" / "ds1.csv").set_index("filename"); feats = list(cp.columns)
Xall_std = StandardScaler().fit_transform(cp.values); XS = pd.DataFrame(Xall_std, index=cp.index, columns=feats)   # unsupervised standardisation (all patches)
texture = [f for f in feats if not f.startswith("intensity")]


def select(tr, kind):
    X = XS.loc[tr.fn]; y = tr.label.map(RANK).values
    if kind == "all50": return feats
    if kind == "texture39": return texture
    rho = pd.Series([abs(spearmanr(X[f], y)[0]) for f in feats], index=feats).fillna(0)
    if kind == "top10_univ": return list(rho.sort_values(ascending=False).index[:10])
    if kind == "top20_univ": return list(rho.sort_values(ascending=False).index[:20])
    if kind == "lasso":
        for a in (0.2, 0.1, 0.05, 0.02, 0.01):
            nz = [f for f, c in zip(feats, Lasso(alpha=a, max_iter=5000).fit(X.values, y).coef_) if abs(c) > 1e-8]
            if len(nz) >= 5: return nz[:30]
        return nz
    if kind == "one_per_cluster":                         # correlation-cluster the features (|r| distance), keep the best-rho member per cluster
        C = np.abs(np.corrcoef(XS.loc[tr.fn].values.T)); D = 1 - C; np.fill_diagonal(D, 0)
        cl = fcluster(linkage(squareform(D, checks=False), "average"), t=0.3, criterion="distance")
        return [rho[[f for f, c in zip(feats, cl) if c == k]].idxmax() for k in np.unique(cl)]


def evaluate(Ztr, ytr, Zte, yte):
    rg = Ridge(alpha=1.0).fit(Ztr, ytr).predict(Zte)
    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Ztr, ytr); pred = lr.predict(Zte)
    return dict(pc1_abs_rho=abs(spearmanr(Zte[:, 0], yte)[0]), ridge_spearman=spearmanr(rg, yte)[0], ridge_mae=np.abs(rg - yte).mean(),
                bal_acc=balanced_accuracy_score(yte, pred), within1=(np.abs(pred - yte) <= 1).mean())


KINDS = ["all50", "texture39", "top10_univ", "top20_univ", "lasso", "one_per_cluster"]; KS = [1, 2, 3, 5]
folds = make_folds(0); rng = np.random.RandomState(0); rows = []; chosen = {k: [] for k in KINDS}
for fold in range(N_FOLDS):
    T = {g for g, f in folds.items() if f == fold}
    tr = train_sources(b1, b2, T, rng)["B1+B2"]; tr = tr[tr.fn.isin(cp.index)]; ytr = tr.label.map(RANK).values
    for tname, te in (("B1", b1[b1.g.isin(T)]), ("B2", b2[b2.g.isin(T)])):
        te = te[te.fn.isin(cp.index)]; yte = te.label.map(RANK).values
        for kind in KINDS:
            cols = select(tr, kind)
            if tname == "B1": chosen[kind].append(cols)
            pca = PCA(n_components=5, random_state=0).fit(XS[cols].values)               # unsupervised, all patches
            Ztr, Zte = pca.transform(XS.loc[tr.fn, cols].values), pca.transform(XS.loc[te.fn, cols].values)
            for k in KS:
                if k > len(cols): continue
                rows.append(dict(fold=fold, test=tname, method=f"PCA({kind})", k=k, n_feat=len(cols), n_te=len(te), **evaluate(Ztr[:, :k], ytr, Zte[:, :k], yte)))
        Xtr, Xte = XS.loc[tr.fn].values, XS.loc[te.fn].values
        pls = PLSRegression(n_components=3).fit(Xtr, ytr)
        for k in (1, 2, 3):
            rows.append(dict(fold=fold, test=tname, method="PLS (supervised)", k=k, n_feat=50, n_te=len(te), **evaluate(pls.transform(Xtr)[:, :k], ytr, pls.transform(Xte)[:, :k], yte)))
        lda = LinearDiscriminantAnalysis(n_components=3).fit(Xtr, ytr)
        for k in (1, 2, 3):
            rows.append(dict(fold=fold, test=tname, method="LDA (supervised)", k=k, n_feat=50, n_te=len(te), **evaluate(lda.transform(Xtr)[:, :k], ytr, lda.transform(Xte)[:, :k], yte)))
        rows.append(dict(fold=fold, test=tname, method="full 50 features (no reduction)", k=50, n_feat=50, n_te=len(te), **evaluate(Xtr, ytr, Xte, yte) | {"pc1_abs_rho": np.nan}))
R = pd.DataFrame(rows)
# pooled metric = patch-weighted mean over folds
metrics = ["pc1_abs_rho", "ridge_spearman", "ridge_mae", "bal_acc", "within1"]
S = R.groupby(["test", "method", "k"]).apply(lambda d: pd.Series({m: np.average(d[m], weights=d.n_te) if d[m].notna().all() else np.nan for m in metrics} | {"n_feat": d.n_feat.mean()})).reset_index()
S.to_csv(OUT / "selected_pca.csv", index=False)
cnt = pd.DataFrame({k: pd.Series([f for cols in v for f in cols]).value_counts() for k, v in chosen.items() if k in ("top10_univ", "lasso", "one_per_cluster")}).fillna(0).astype(int)
cnt.to_csv(OUT / "selected_features.csv")
pd.set_option("display.width", 220)
print(S[S.test == "B1"].round(3).to_string(index=False))
print("selected in all 5 folds:"); [print(k, sorted(cnt.index[cnt[k] == 5])) for k in cnt.columns]
print("cluster count per fold:", [len(c) for c in chosen["one_per_cluster"]])

# ---- illustrative scatter: features selected on ALL labelled B1+B2 (not used for the CV numbers) ----
allb = pd.concat([b2, b1[~b1.fn.isin(set(b2.fn))]]); allb = allb[allb.fn.isin(cp.index)]
fig, axes = plt.subplots(1, 4, figsize=(19, 4.9)); r_ = np.random.RandomState(0)
for ax, kind in zip(axes, ["all50", "texture39", "top10_univ", "one_per_cluster"]):
    cols = select(allb, kind); pca = PCA(2, random_state=0).fit(XS[cols].values); Z = pca.transform(XS[cols].values); Zl = pd.DataFrame(Z, index=cp.index)
    sub = r_.choice(len(Z), 6000, replace=False); ax.scatter(Z[sub, 0], Z[sub, 1], s=2, c="#d9d9d9", alpha=0.5, rasterized=True, linewidths=0)
    for c in range(4):
        d = allb[allb.label.map(RANK) == c]; ax.scatter(Zl.loc[d.fn, 0], Zl.loc[d.fn, 1], s=7, c=COL[c], linewidths=0, label=NAMES[c])
    rho = spearmanr(Zl.loc[allb.fn, 0], allb.label.map(RANK))[0]
    ax.set_title(f"PCA({kind}), {len(cols)} features\nPC1 {pca.explained_variance_ratio_[0]:.0%} var, ρ(PC1, rank) = {rho:.2f}", fontsize=9); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
axes[0].legend(frameon=False, fontsize=8, markerscale=2)
fig.tight_layout(); fig.savefig(OUT / "selected_pca_scatter.png", dpi=150); plt.close(fig)

b = S[(S.test == "B1") & (S.k.isin([2, 3]) | S.method.str.startswith("full"))]
meth = ["PCA(all50)", "PCA(texture39)", "PCA(top10_univ)", "PCA(one_per_cluster)", "PCA(lasso)", "PLS (supervised)", "LDA (supervised)"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6)); x = np.arange(len(meth))
for ax, col, ttl in zip(axes, ["ridge_spearman", "bal_acc"], ["held-out Spearman with the rank (B1 test)", "held-out balanced accuracy, 4 classes (B1 test)"]):
    for i, (k, c) in enumerate(((2, "#6fa8dc"), (3, "#0f3d79"))):
        v = [b[(b.method == m_) & (b.k == k)][col].iloc[0] for m_ in meth]; ax.bar(x + (i - 0.5) * 0.38, v, 0.36, color=c, label=f"{k} components")
    full = b[b.method.str.startswith("full")][col].iloc[0]; ax.axhline(full, ls="--", c="k", lw=1, label=f"full 50 features ({full:.2f})")
    ax.set_xticks(x); ax.set_xticklabels(meth, rotation=30, ha="right", fontsize=8); ax.set_title(ttl, fontsize=10); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "selected_pca_bars.png", dpi=150); print("saved")

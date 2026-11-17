#!/usr/bin/env python3
"""
analyze_top10_cp.py

Are the top-10 features by univariate |Spearman| with the rank the most informative / discriminant 10?
Within each frame-grouped CV fold (training labels only: B1+B2 of the other images) pick 10 CP features by
  univ_rho | anova_F (class discriminant, order ignored) | mutual_info | ridge_perm (permutation importance of a ridge on the rank,
  computed on the training fold) | one_per_cluster | random (mean of 20 draws),
then fit ridge (rank) and balanced logistic regression (classes) on those 10 raw features (no PCA) and score on the held-out
B1 test patches.  Also reports redundancy = mean pairwise |corr| among the selected features and the overlap with univ_rho.
Outputs: results/ordinal_cp/top10_compare.csv, top10_sets.csv
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_common import *

OUT = Path(__file__).resolve().parents[1] / "results" / "ordinal_cp"
RANK = {c: i for i, c in enumerate(FA4)}
b1, b2 = load_fa4()
cp = pd.read_csv(FEAT / "cellprofiler" / "ds1.csv").set_index("filename"); feats = list(cp.columns)
XS = pd.DataFrame(StandardScaler().fit_transform(cp.values), index=cp.index, columns=feats)
K = 10


def pick(tr, how, rs):
    X = XS.loc[tr.fn]; y = tr.label.map(RANK).values
    rho = pd.Series([abs(spearmanr(X[f], y)[0]) for f in feats], index=feats).fillna(0)
    if how == "univ_rho": return list(rho.sort_values(ascending=False).index[:K])
    if how == "anova_F": return list(pd.Series(np.nan_to_num(f_classif(X.values, y)[0]), index=feats).sort_values(ascending=False).index[:K])
    if how == "mutual_info": return list(pd.Series(mutual_info_classif(X.values, y, random_state=0), index=feats).sort_values(ascending=False).index[:K])
    if how == "ridge_perm":
        m = Ridge(alpha=10.0).fit(X.values, y); pi = permutation_importance(m, X.values, y, n_repeats=5, random_state=0, scoring="neg_mean_absolute_error")
        return list(pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False).index[:K])
    if how == "one_per_cluster":
        D = 1 - np.abs(np.corrcoef(X.values.T)); np.fill_diagonal(D, 0)
        cl = fcluster(linkage(squareform(D, checks=False), "average"), t=0.3, criterion="distance")
        return [rho[[f for f, c in zip(feats, cl) if c == k]].idxmax() for k in np.unique(cl)][:K]
    if how == "random": return list(rs.choice(feats, K, replace=False))


def evaluate(cols, tr, te):
    ytr, yte = tr.label.map(RANK).values, te.label.map(RANK).values
    sc = StandardScaler().fit(cp.loc[tr.fn, cols]); Xtr, Xte = sc.transform(cp.loc[tr.fn, cols]), sc.transform(cp.loc[te.fn, cols])
    rg = Ridge(alpha=10.0).fit(Xtr, ytr).predict(Xte); pr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr).predict(Xte)
    return dict(spearman=spearmanr(rg, yte)[0], mae=np.abs(rg - yte).mean(), bal_acc=balanced_accuracy_score(yte, pr), within1=(np.abs(pr - yte) <= 1).mean())


HOWS = ["univ_rho", "anova_F", "mutual_info", "ridge_perm", "one_per_cluster", "random"]
folds = make_folds(0); rng = np.random.RandomState(0); rows = []; sets = []
for fold in range(N_FOLDS):
    T = {g for g, f in folds.items() if f == fold}
    tr = train_sources(b1, b2, T, rng)["B1+B2"]; tr = tr[tr.fn.isin(cp.index)]; te = b1[b1.g.isin(T)]; te = te[te.fn.isin(cp.index)]
    base = None
    for how in HOWS:
        reps = 20 if how == "random" else 1; rs = np.random.RandomState(fold); res = []; reds = []; ovl = []
        for _ in range(reps):
            cols = pick(tr, how, rs); res.append(evaluate(cols, tr, te))
            C = np.abs(np.corrcoef(XS.loc[tr.fn, cols].values.T)); reds.append(C[np.triu_indices(len(cols), 1)].mean())
            if how == "univ_rho": base = cols
            ovl.append(len(set(cols) & set(base)) if base else np.nan)
        rows.append(dict(fold=fold, how=how, n_te=len(te), redundancy=np.mean(reds), overlap_with_univ=np.mean(ovl), **{k: np.mean([r[k] for r in res]) for k in res[0]}))
        if how != "random": sets.append(dict(fold=fold, how=how, features=", ".join(cols)))
R = pd.DataFrame(rows)
S = R.groupby("how").apply(lambda d: pd.Series({m: np.average(d[m], weights=d.n_te) for m in ["spearman", "mae", "bal_acc", "within1", "redundancy", "overlap_with_univ"]})).reindex(HOWS)
S.to_csv(OUT / "top10_compare.csv"); pd.DataFrame(sets).to_csv(OUT / "top10_sets.csv", index=False)
# full-feature reference
refs = []
for fold in range(N_FOLDS):
    T = {g for g, f in folds.items() if f == fold}; r2 = np.random.RandomState(0)
    tr = train_sources(b1, b2, T, np.random.RandomState(fold))["B1+B2"]; tr = tr[tr.fn.isin(cp.index)]; te = b1[b1.g.isin(T)]; te = te[te.fn.isin(cp.index)]
    refs.append({**evaluate(feats, tr, te), "n": len(te)})
ref = pd.DataFrame(refs); print("full 50 reference:", {k: round(float(np.average(ref[k], weights=ref.n)), 3) for k in ["spearman", "mae", "bal_acc", "within1"]})
pd.set_option("display.width", 200); print(S.round(3).to_string())
d = pd.DataFrame(sets); print(d[d.fold == 0].to_string(index=False, max_colwidth=160))

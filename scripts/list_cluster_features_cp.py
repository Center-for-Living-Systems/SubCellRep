#!/usr/bin/env python3
"""List the 'one per correlated cluster' CP features: per-fold picks (training labels only), pick frequency, and the final
set + cluster members from all labelled B1+B2 patches. Output: results/ordinal_cp/cluster_features.csv, cluster_members.csv"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from collections import Counter
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_common import *

OUT = Path(__file__).resolve().parents[1] / "results" / "ordinal_cp"; RANK = {c: i for i, c in enumerate(FA4)}
b1, b2 = load_fa4(); cp = pd.read_csv(FEAT / "cellprofiler" / "ds1.csv").set_index("filename"); feats = list(cp.columns)
XS = pd.DataFrame(StandardScaler().fit_transform(cp.values), index=cp.index, columns=feats)


def clusters(tr):
    X = XS.loc[tr.fn]; y = tr.label.map(RANK).values
    rho = pd.Series([abs(spearmanr(X[f], y)[0]) for f in feats], index=feats).fillna(0)
    D = 1 - np.abs(np.corrcoef(X.values.T)); np.fill_diagonal(D, 0)
    cl = fcluster(linkage(squareform(D, checks=False), "average"), t=0.3, criterion="distance")
    out = []
    for k in np.unique(cl):
        mem = [f for f, c in zip(feats, cl) if c == k]; out.append((rho[mem].idxmax(), mem, rho))
    return out


folds = make_folds(0); rng = np.random.RandomState(0); picks = []
for fold in range(N_FOLDS):
    T = {g for g, f in folds.items() if f == fold}
    tr = train_sources(b1, b2, T, rng)["B1+B2"]; tr = tr[tr.fn.isin(cp.index)]
    picks.append([r for r, _, _ in clusters(tr)])
cnt = Counter(f for p in picks for f in p)
allb = pd.concat([b2, b1[~b1.fn.isin(set(b2.fn))]]); allb = allb[allb.fn.isin(cp.index)]
fin = clusters(allb); rho = fin[0][2]
rows = []
for rep, mem, _ in sorted(fin, key=lambda x: -rho[x[0]]):
    rows.append(dict(representative=rep, rho_with_rank=round(rho[rep], 3), picked_in_folds=cnt.get(rep, 0), cluster_size=len(mem), members=", ".join(m for m in mem if m != rep)))
F = pd.DataFrame(rows); F.to_csv(OUT / "cluster_features.csv", index=False)
pd.DataFrame({"feature": list(cnt), "picked_in_folds": list(cnt.values())}).sort_values("picked_in_folds", ascending=False).to_csv(OUT / "cluster_pick_frequency.csv", index=False)
pd.set_option("display.width", 250, "display.max_colwidth", 140)
print(F.to_string(index=False)); print("\nper-fold pick frequency:"); print(sorted(cnt.items(), key=lambda x: -x[1]))

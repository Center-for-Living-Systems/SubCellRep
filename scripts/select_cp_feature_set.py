#!/usr/bin/env python3
"""
select_cp_feature_set.py

Smallest CP feature subset that matches the full 50-feature set, for
  fa4   : 4-class FA classifier (balanced logistic regression), train B1+B2, test B1 (representative) and B2
  adnoad: adhesion vs no-adhesion (balanced logistic regression), train/test on combined B1+B2 labels

Forward selection (criterion = inner grouped-CV log loss, so it is smooth) is run INSIDE each outer frame-grouped
fold on training frames only; the held-out score is recorded for every set size k, so the curve is honest.
Also scores: full 50, one-per-cluster (10, picked on the training fold), random-k.
The final set for each task = forward selection on all labelled data, truncated at the smallest k whose outer
test score is within 1 SE (over folds) of the full set.

Outputs (results/ordinal_cp/): fsel_curve.csv, fsel_final_sets.csv, fsel_curve.png, fsel_pick_freq.csv
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_common import *

OUT = Path(__file__).resolve().parents[1] / "results" / "ordinal_cp"
KMAX = 25
RANK = {c: i for i, c in enumerate(FA4)}
cp = pd.read_csv(FEAT / "cellprofiler" / "ds1.csv").set_index("filename"); feats = list(cp.columns)
b1, b2 = load_fa4()
comb = pd.read_csv(PREP / "combined_labels.csv"); comb = comb[comb.filename.isin(cp.index)].copy()
comb["g"] = comb.filename.map(group_of)


def clf():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))


def inner_ll(cols, X, y, g, nclass):
    ll = []
    for tr, te in GroupKFold(3).split(X, y, g):
        m = clf().fit(X.iloc[tr][cols], y[tr])
        p = m.predict_proba(X.iloc[te][cols])
        ll.append(log_loss(y[te], p, labels=m.classes_) if len(set(y[te])) > 1 else np.nan)
    return np.nanmean(ll)


def forward(X, y, g, kmax=KMAX):
    sel, rem = [], list(X.columns)
    nclass = len(set(y))
    for _ in range(kmax):
        sc = Parallel(n_jobs=8)(delayed(inner_ll)(sel + [f], X, y, g, nclass) for f in rem)
        best = rem[int(np.nanargmin(sc))]; sel.append(best); rem.remove(best)
    return sel


def one_per_cluster(X, y):
    C = X.corr(method="spearman").abs().values
    D = 1 - C; np.fill_diagonal(D, 0)
    cl = fcluster(linkage(squareform(D, checks=False), "average"), t=0.3, criterion="distance")
    rho = pd.Series([abs(spearmanr(X[f], y)[0]) for f in X.columns], index=X.columns).fillna(0)
    return [rho[[f for f, c in zip(X.columns, cl) if c == k]].idxmax() for k in np.unique(cl)]


def score(cols, trX, trY, teX, teY, task):
    m = clf().fit(trX[cols], trY); p = m.predict(teX[cols])
    r = dict(bal_acc=balanced_accuracy_score(teY, p))
    if task == "adnoad": r["auroc"] = roc_auc_score(teY, m.predict_proba(teX[cols])[:, 1])
    else: r["within1"] = (np.abs(p - teY) <= 1).mean()
    return r


def tasks_for_fold(T, rng, seed):
    """yield (task, test_name, trainX, trainY, trainG, teX, teY)"""
    tr = train_sources(b1, b2, T, rng)["B1+B2"]; tr = tr[tr.fn.isin(cp.index)]
    out = [("fa4", tr.fn.values, tr.label.map(RANK).values, tr.g.values,
            {"B1": b1[b1.g.isin(T) & b1.fn.isin(cp.index)], "B2": b2[b2.g.isin(T) & b2.fn.isin(cp.index)]})]
    tc = comb[~comb.g.isin(T)]; te = comb[comb.g.isin(T)]
    out.append(("adnoad", tc.filename.values, tc.binary.values, tc.g.values,
                {"all": pd.DataFrame({"fn": te.filename.values, "label": te.binary.values})}))
    return out


def main():
    folds = make_folds(0); rows = []; picks = []
    for fold in range(N_FOLDS):
        T = {g for g, f in folds.items() if f == fold}
        for task, trf, trY, trG, tests in tasks_for_fold(T, np.random.RandomState(fold), fold):
            trX = cp.loc[trf]
            sel = forward(trX, trY, trG)
            opc = one_per_cluster(trX, trY)
            picks.append(dict(task=task, fold=fold, order=", ".join(sel)))
            rs = np.random.RandomState(fold)
            for tn, te in tests.items():
                teY = te.label.map(RANK).values if task == "fa4" else te.label.values
                teX = cp.loc[te.fn]
                full = score(feats, trX, trY, teX, teY, task)
                rows.append(dict(task=task, test=tn, fold=fold, n_te=len(te), method="full50", k=len(feats), **full))
                rows.append(dict(task=task, test=tn, fold=fold, n_te=len(te), method="one_per_cluster", k=len(opc), **score(opc, trX, trY, teX, teY, task)))
                for k in range(1, KMAX + 1):
                    rows.append(dict(task=task, test=tn, fold=fold, n_te=len(te), method="forward", k=k, **score(sel[:k], trX, trY, teX, teY, task)))
                    rr = [score(list(rs.choice(feats, k, replace=False)), trX, trY, teX, teY, task)["bal_acc"] for _ in range(10)]
                    rows.append(dict(task=task, test=tn, fold=fold, n_te=len(te), method="random", k=k, bal_acc=np.mean(rr)))
            print(f"fold {fold} {task} done: {sel[:8]}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "fsel_curve_folds.csv", index=False)
    pd.DataFrame(picks).to_csv(OUT / "fsel_fold_orders.csv", index=False)

    def agg(d):
        w = d.n_te.values; m = np.average(d.bal_acc, weights=w)
        se = np.sqrt(np.cov(d.bal_acc, aweights=w) / len(d)) if len(d) > 1 else 0
        return pd.Series(dict(bal_acc=m, se=se, **({"auroc": np.average(d.auroc, weights=w)} if "auroc" in d and d.auroc.notna().all() else {})))
    C = R.groupby(["task", "test", "method", "k"]).apply(agg).reset_index(); C.to_csv(OUT / "fsel_curve.csv", index=False)

    # pick frequency over folds, final sets on all labelled data
    freq = []; final = []
    allfa = pd.concat([b2, b1[~b1.fn.isin(set(b2.fn))]]); allfa = allfa[allfa.fn.isin(cp.index)]
    data = {"fa4": (cp.loc[allfa.fn], allfa.label.map(RANK).values, allfa.g.values),
            "adnoad": (cp.loc[comb.filename], comb.binary.values, comb.g.values)}
    for task, (X, y, g) in data.items():
        P = pd.DataFrame(picks); P = P[P.task == task]
        cnt = pd.Series([f for o in P.order for f in o.split(", ")[:10]]).value_counts()
        for f, c in cnt.items(): freq.append(dict(task=task, feature=f, folds_in_top10=int(c)))
        order = forward(X, y, g)
        tests = [t for t in C[C.task == task].test.unique()]
        ref = C[(C.task == task) & (C.method == "full50")].set_index("test")
        fw = C[(C.task == task) & (C.method == "forward")]
        ok = []
        for k in range(1, KMAX + 1):
            good = all(fw[(fw.test == t) & (fw.k == k)].bal_acc.iloc[0] >= ref.loc[t, "bal_acc"] - ref.loc[t, "se"] for t in tests)
            ok.append(good)
        kbest = next((k for k, o in enumerate(ok, 1) if o), KMAX)
        final.append(dict(task=task, k=kbest, features=", ".join(order[:kbest]), full_order=", ".join(order)))
        print(task, "k =", kbest, order[:kbest])
    pd.DataFrame(freq).to_csv(OUT / "fsel_pick_freq.csv", index=False)
    pd.DataFrame(final).to_csv(OUT / "fsel_final_sets.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (task, tn) in zip(axes, [("fa4", "B1"), ("fa4", "B2"), ("adnoad", "all")]):
        d = C[(C.task == task) & (C.test == tn)]
        f = d[d.method == "forward"]; r = d[d.method == "random"]; full = d[d.method == "full50"].iloc[0]; o = d[d.method == "one_per_cluster"].iloc[0]
        ax.errorbar(f.k, f.bal_acc, f.se, label="forward selection", color="#2ca02c", marker="o", ms=3)
        ax.plot(r.k, r.bal_acc, label="random k", color="#999", ls="--")
        ax.axhline(full.bal_acc, color="k", label=f"full 50 ({full.bal_acc:.3f})")
        ax.axhspan(full.bal_acc - full.se, full.bal_acc + full.se, color="k", alpha=0.08)
        ax.scatter([o.k], [o.bal_acc], color="#d62728", zorder=5, label=f"one-per-cluster ({o.bal_acc:.3f})")
        ax.set_title(f"{task}: test {tn}", fontsize=10); ax.set_xlabel("number of features"); ax.set_ylabel("balanced accuracy"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fsel_curve.png", dpi=150)
    pd.set_option("display.width", 200)
    print(C[C.method.isin(["full50", "one_per_cluster"])].round(3).to_string())
    for task in ("fa4", "adnoad"):
        for t in C[C.task == task].test.unique():
            print(task, t); print(C[(C.task == task) & (C.test == t) & (C.method == "forward") & (C.k.isin([1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25]))].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

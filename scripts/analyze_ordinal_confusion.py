#!/usr/bin/env python3
"""
analyze_ordinal_confusion.py

Regression view of the 4 FA classes (rank: nascent 0, focal complex 1, focal adhesion 2, fibrillar 3) on the
B1 test set (representative), frame-grouped 5-fold CV (seed 0), probe trained on B1+B2 labels of the other folds.

Per representation (CP, IL, CP+IL, AE_b12):
  ridge     : ridge regression on the rank (alpha=10) -> Spearman, Pearson, MAE
  soft_rank : expected rank sum_k p_k*k from the logistic-regression probabilities -> Spearman, MAE
  errors    : of the misclassified patches, share that are adjacent-class errors; p(true) rank among classes
  confidence: mean max-prob (correct vs wrong), AUROC of max-prob / margin for predicting correctness
Outputs: results/cv_b1b2/ordinal_analysis.csv, ordinal_confusion.png, ordinal_oof.csv
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_common import *

OUT = Path(__file__).resolve().parents[1] / "results" / "cv_b1b2"
RANK = {c: i for i, c in enumerate(FA4)}
b1, b2 = load_fa4()
cp = pd.read_csv(FEAT / "cellprofiler" / "ds1.csv").set_index("filename")
il = pd.read_csv(FEAT / "ilastik" / "ds1.csv").set_index("filename")
folds = make_folds(0); rng = np.random.RandomState(0)
rows = []
for k in range(N_FOLDS):
    T = {g for g, f in folds.items() if f == k}
    tr = train_sources(b1, b2, T, rng)["B1+B2"]; te = b1[b1.g.isin(T)]
    z = pd.read_csv(RUN / "cv_ae" / f"b12_f{k}" / "latents.csv").set_index("filename")
    z = z[[c for c in z.columns if c.startswith("z_")]]
    for rep, X in {"CP": cp, "IL": il, "CP+IL": pd.concat([cp, il], axis=1), "AE_b12": z}.items():
        t_ = tr[tr.fn.isin(X.index)]; e = te[te.fn.isin(X.index)]
        sc = StandardScaler().fit(X.loc[t_.fn]); Xtr, Xte = sc.transform(X.loc[t_.fn]), sc.transform(X.loc[e.fn])
        ytr = t_.label.map(RANK).values
        rg = Ridge(alpha=10.0).fit(Xtr, ytr).predict(Xte)
        lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        P = np.zeros((len(e), 4)); P[:, lr.classes_] = lr.predict_proba(Xte)
        rows.append(pd.DataFrame({"rep": rep, "fold": k, "fn": e.fn.values, "y": e.label.map(RANK).values, "ridge": rg,
                                  **{f"p{c}": P[:, c] for c in range(4)}}))
oof = pd.concat(rows, ignore_index=True); oof.to_csv(OUT / "ordinal_oof.csv", index=False)

res = []
for rep, d in oof.groupby("rep"):
    P = d[[f"p{c}" for c in range(4)]].values; y = d.y.values
    pred = P.argmax(1); soft = P @ np.arange(4)
    wrong = pred != y; srt = np.sort(P, 1)
    conf, margin = srt[:, -1], srt[:, -1] - srt[:, -2]
    true_rank = (-P).argsort(1).argsort(1)[np.arange(len(y)), y]      # 0 = true class has highest prob
    res.append(dict(rep=rep, n=len(d), acc=1 - wrong.mean(),
                    ridge_spearman=spearmanr(d.ridge, y)[0], ridge_mae=np.abs(d.ridge - y).mean(),
                    soft_spearman=spearmanr(soft, y)[0], soft_mae=np.abs(soft - y).mean(),
                    err_adjacent=(np.abs(pred - y)[wrong] == 1).mean(), err_far=(np.abs(pred - y)[wrong] >= 2).mean(),
                    true_is_2nd=(true_rank[wrong] == 1).mean(), top2_acc=(true_rank <= 1).mean(),
                    conf_correct=conf[~wrong].mean(), conf_wrong=conf[wrong].mean(),
                    auc_maxprob=roc_auc_score(~wrong, conf), auc_margin=roc_auc_score(~wrong, margin),
                    p_true_plus_pred_wrong=np.mean([P[i, y[i]] + P[i, pred[i]] for i in np.where(wrong)[0]])))
R = pd.DataFrame(res).set_index("rep"); R.to_csv(OUT / "ordinal_analysis.csv")
pd.set_option("display.width", 220); print(R.round(3).T.to_string())

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
names = ["nascent", "f.complex", "f.adhesion", "fibrillar"]; cols = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for a, rep in zip(ax[:1], ["IL"]):
    d = oof[oof.rep == rep]
    bp = a.boxplot([d.ridge[d.y == c] for c in range(4)], labels=[f"{n}\n(n={(d.y == c).sum()})" for c, n in enumerate(names)], patch_artist=True, showfliers=False)
    for p_, c in zip(bp["boxes"], cols): p_.set_facecolor(c); p_.set_alpha(0.7)
    a.plot([1, 2, 3, 4], [0, 1, 2, 3], "k--", lw=1); a.set_ylabel("out-of-fold ridge score (ilastik)"); a.set_title("regression score by true class (B1 test)")
d = oof[oof.rep == "IL"]; P = d[[f"p{c}" for c in range(4)]].values; y = d.y.values; wrong = P.argmax(1) != y
ax[1].hist([P.max(1)[~wrong], P.max(1)[wrong]], bins=15, range=(0.25, 1), label=["correct", "misclassified"], color=["#0f3d79", "#d65f0e"], density=True)
ax[1].set_xlabel("max class probability"); ax[1].set_title("ilastik classifier confidence"); ax[1].legend(frameon=False)
x = np.arange(len(R)); ax[2].bar(x - 0.2, R.err_adjacent, 0.4, label="adjacent-class errors", color="#0f3d79"); ax[2].bar(x + 0.2, R.true_is_2nd, 0.4, label="true class is 2nd-most-probable", color="#6fa8dc")
ax[2].set_xticks(x); ax[2].set_xticklabels(R.index); ax[2].set_ylim(0, 1); ax[2].set_title("share of misclassified patches"); ax[2].legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "ordinal_confusion.png", dpi=150); print("saved")

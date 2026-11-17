#!/usr/bin/env python3
"""
analyze_ordinal_cp.py

Ordinal / regression view of the 4 FA classes using CellProfiler (CP) features.
Frame-grouped 5-fold CV (seed 0, same folds as eval_cv_b1b2.py); rank: nascent 0, focal complex 1,
focal adhesion 2, fibrillar 3. Models per fold: ridge regression on the rank (alpha=10) and balanced
multinomial logistic regression (soft rank = sum_k p_k * k), trained on B1 / B2 / B1+B2 labels of the other
folds, tested on held-out images (B1 or B2 labels).

Outputs (results/ordinal_cp/): metrics.csv, oof.csv, feature_axis.csv, summary.json and figures
  ordinal_vs_pc1.png score_by_class.png error_structure.png boundary.png confidence.png sources.png features.png
"""
import json, sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_common import *

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "ordinal_cp"; OUT.mkdir(parents=True, exist_ok=True)
RANK = {c: i for i, c in enumerate(FA4)}
NAMES = ["nascent", "f.complex", "f.adhesion", "fibrillar"]
COL = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

b1, b2 = load_fa4()
cp = pd.read_csv(FEAT / "cellprofiler" / "ds1.csv").set_index("filename")
folds = make_folds(0); rng = np.random.RandomState(0)

rows = []
for k in range(N_FOLDS):
    T = {g for g, f in folds.items() if f == k}
    srcs = train_sources(b1, b2, T, rng)
    tests = {"B1": b1[b1.g.isin(T)], "B2": b2[b2.g.isin(T)]}
    for sname in ("B1", "B2", "B1+B2"):
        tr = srcs[sname]; tr = tr[tr.fn.isin(cp.index)]
        sc = StandardScaler().fit(cp.loc[tr.fn]); Xtr = sc.transform(cp.loc[tr.fn]); ytr = tr.label.map(RANK).values
        rg = Ridge(alpha=10.0).fit(Xtr, ytr)
        lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(Xtr, ytr)
        for tname, te in tests.items():
            te = te[te.fn.isin(cp.index)]; Xte = sc.transform(cp.loc[te.fn])
            P = np.zeros((len(te), 4)); P[:, lr.classes_] = lr.predict_proba(Xte)
            rows.append(pd.DataFrame({"src": sname, "test": tname, "fold": k, "fn": te.fn.values, "y": te.label.map(RANK).values,
                                      "ridge": rg.predict(Xte), **{f"p{c}": P[:, c] for c in range(4)}}))
oof = pd.concat(rows, ignore_index=True); oof.to_csv(OUT / "oof.csv", index=False)


def chance_adjacent(y):
    """share of errors that would be adjacent if errors were uniform over the other classes (given true-class counts)."""
    cnt = np.bincount(y, minlength=4); adj = np.array([1, 2, 2, 1]) / 3.0
    return float((cnt * adj).sum() / cnt.sum())


mets = []
for (src, test), d in oof.groupby(["src", "test"]):
    P = d[[f"p{c}" for c in range(4)]].values; y = d.y.values; pred = P.argmax(1); wrong = pred != y
    soft = P @ np.arange(4); srt = np.sort(P, 1); conf, margin = srt[:, -1], srt[:, -1] - srt[:, -2]
    tr_rank = (-P).argsort(1).argsort(1)[np.arange(len(y)), y]
    cnt = np.bincount(y, minlength=4)
    mets.append(dict(src=src, test=test, n=len(d), acc=1 - wrong.mean(), within1=(np.abs(pred - y) <= 1).mean(),
                     ridge_spearman=spearmanr(d.ridge, y)[0], ridge_mae=np.abs(d.ridge - y).mean(),
                     soft_spearman=spearmanr(soft, y)[0], soft_mae=np.abs(soft - y).mean(),
                     median_mae=np.abs(y - np.median(y)).mean(), err_adjacent=(np.abs(pred - y)[wrong] == 1).mean(),
                     err_adjacent_chance=chance_adjacent(y), true_is_2nd=(tr_rank[wrong] == 1).mean(), top2=(tr_rank <= 1).mean(),
                     conf_correct=conf[~wrong].mean(), conf_wrong=conf[wrong].mean(),
                     auc_maxprob=roc_auc_score(~wrong, conf), auc_margin=roc_auc_score(~wrong, margin),
                     mass_true_pred_wrong=np.mean([P[i, y[i]] + P[i, pred[i]] for i in np.where(wrong)[0]]),
                     **{f"n_{NAMES[c]}": int(cnt[c]) for c in range(4)}))
M = pd.DataFrame(mets); M.to_csv(OUT / "metrics.csv", index=False)

main = oof[(oof.src == "B1+B2") & (oof.test == "B1")].copy()
P = main[[f"p{c}" for c in range(4)]].values; y = main.y.values; pred = P.argmax(1); wrong = pred != y
main["soft"] = P @ np.arange(4); main["pred"] = pred; main["conf"] = P.max(1)
mean_sc = main.groupby("y").ridge.mean().values                       # class mean OOF ridge score
summ = {"class_mean_score": dict(zip(NAMES, mean_sc.round(3).tolist())),
        "class_sd_score": dict(zip(NAMES, main.groupby("y").ridge.std().round(3).tolist()))}

# transition test: are errors displaced toward the predicted class on the score axis?
closest = np.abs(main.ridge.values[:, None] - mean_sc[None, :]).argmin(1)
adj = wrong & (np.abs(pred - y) == 1)
summ["adj_errors_closer_to_pred_mean"] = float((np.abs(main.ridge[adj] - mean_sc[pred[adj]]) < np.abs(main.ridge[adj] - mean_sc[y[adj]])).mean())
summ["correct_closer_to_own_mean"] = float((closest[~wrong] == y[~wrong]).mean())
summ["adj_errors_between_means"] = float(((main.ridge[adj] - np.minimum(mean_sc[pred[adj]], mean_sc[y[adj]])) * (main.ridge[adj] - np.maximum(mean_sc[pred[adj]], mean_sc[y[adj]])) <= 0).mean())
summ["n_errors"] = int(wrong.sum()); summ["n_adjacent_errors"] = int(adj.sum())

# ---------- features: PC1 loadings and ridge coefficients (display fit on all labelled B1+B2) ----------
allb = pd.concat([b2, b1[~b1.fn.isin(set(b2.fn))]]); allb = allb[allb.fn.isin(cp.index)]
sc = StandardScaler().fit(cp.loc[allb.fn]); Xall = sc.transform(cp.loc[allb.fn])
pca = PCA(2, random_state=0).fit(sc.transform(cp.values)); rg = Ridge(alpha=10.0).fit(Xall, allb.label.map(RANK).values)
fa = pd.DataFrame({"feature": cp.columns, "pc1_loading": pca.components_[0], "ridge_coef": rg.coef_})
fa.to_csv(OUT / "feature_axis.csv", index=False)
summ["pc1_var"] = float(pca.explained_variance_ratio_[0]); summ["pc2_var"] = float(pca.explained_variance_ratio_[1])
summ["spearman_ordinal_vs_pc1_all_labelled"] = float(spearmanr(Xall @ rg.coef_, pca.transform(sc.transform(cp.loc[allb.fn]))[:, 0])[0])
summ["cos_pc1_ridge"] = float(np.dot(pca.components_[0], rg.coef_) / np.linalg.norm(rg.coef_))
json.dump(summ, open(OUT / "summary.json", "w"), indent=2)

# ---------- figures ----------
# 1 ordinal axis vs PC1 (from the stage-2 CP embedding coordinates; fit on train split, val edged)
e = pd.read_csv(REPO / "results" / "stage2_rebuild" / "embed_cp_stage2_coords.csv")
fig, ax = plt.subplots(figsize=(7.4, 5.6))
bg = e.label.isna(); ax.scatter(e.pca1[bg], e.ordinal[bg], s=2, c="#d9d9d9", alpha=0.5, rasterized=True, linewidths=0)
for c, col, n in zip(FA4, COL, NAMES):
    m = e.label == c; tr_ = m & (e.split == "train"); va_ = m & (e.split == "val")
    ax.scatter(e.pca1[tr_], e.ordinal[tr_], s=9, c=col, linewidths=0, label=f"{n} (n={m.sum()})")
    ax.scatter(e.pca1[va_], e.ordinal[va_], s=15, c=col, edgecolors="k", linewidths=0.5)
ax.set_xlabel("PC1 (unsupervised, 57% of CP variance)"); ax.set_ylabel("ordinal score (ridge on rank, train-fit)")
ax.legend(frameon=False, fontsize=8); ax.set_title("CP: ordinal axis vs PC1 (black edge = validation split)")
fig.tight_layout(); fig.savefig(OUT / "ordinal_vs_pc1.png", dpi=170); plt.close(fig)

# 2 score by class (ridge and soft rank), OOF, B1 test
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8)); r_ = np.random.RandomState(1)
for ax, col_, ttl in zip(axes, ["ridge", "soft"], ["ridge regression on rank", "soft rank = Σ p_k · k (logistic regression)"]):
    for c in range(4):
        v = main[col_][main.y == c]
        ax.scatter(c + r_.uniform(-0.28, 0.28, len(v)), v, s=7, c=COL[c], alpha=0.45, linewidths=0)
        ax.boxplot([v], positions=[c], widths=0.5, showfliers=False, manage_ticks=False, medianprops=dict(color="k"), boxprops=dict(alpha=0.9), whiskerprops=dict(alpha=0.6), capprops=dict(alpha=0.6))
        ax.text(c, v.max() + 0.05, f"mean {v.mean():.2f}", ha="center", fontsize=8)
    ax.plot([-0.4, 3.4], [-0.4, 3.4], "k--", lw=1)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{n}\n(n={(main.y == c).sum()})" for c, n in enumerate(NAMES)]); ax.set_ylabel("out-of-fold prediction"); ax.set_title(ttl, fontsize=10)
fig.suptitle("CP, B1 test patches, probe trained on B1+B2 of other images (dashed = perfect)", fontsize=11); fig.tight_layout(); fig.savefig(OUT / "score_by_class.png", dpi=170); plt.close(fig)

# 3 error structure
m0 = M[(M.src == "B1+B2") & (M.test == "B1")].iloc[0]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
v = [m0.acc, m0.within1, m0.top2]; axes[0].bar(["exact\naccuracy", "within one\nclass", "true class in\ntop-2 probs"], v, color=["#0f3d79", "#6fa8dc", "#9fc5e8"])
for i, x in enumerate(v): axes[0].text(i, x + 0.015, f"{x:.2f}", ha="center")
axes[0].set_ylim(0, 1.05); axes[0].set_title("how close are the predictions?", fontsize=10)
v = [m0.err_adjacent_chance, m0.err_adjacent, m0.mass_true_pred_wrong]
axes[1].bar(["adjacent share if\nerrors were random", "adjacent share\nobserved", "probability mass on\ntrue+predicted class"], v, color=["#cccccc", "#0f3d79", "#6fa8dc"])
for i, x in enumerate(v): axes[1].text(i, x + 0.015, f"{x:.2f}", ha="center")
axes[1].set_ylim(0, 1.05); axes[1].set_title("structure of the misclassified patches", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "error_structure.png", dpi=170); plt.close(fig)

# 4 boundary: error rate vs score
q = pd.qcut(main.ridge, 8, duplicates="drop"); g = main.assign(wrong=wrong, q=q).groupby("q", observed=True)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
xc = np.arange(len(g)); axes[0].bar(xc, g.wrong.mean().values, color="#d65f0e"); axes[0].set_xticks(xc); axes[0].set_xticklabels([f"{iv.mid:.2f}" for iv in g.wrong.mean().index], fontsize=8)
axes[0].set_xlabel("ridge score (octile midpoint)"); axes[0].set_ylabel("misclassification rate"); axes[0].set_title("misclassification rate by score octile (roughly flat)", fontsize=10)
comp = main.assign(q=q).groupby(["q", "y"], observed=True).size().unstack(fill_value=0); comp = comp.div(comp.sum(1), axis=0)
bot = np.zeros(len(comp))
for c in range(4):
    if c in comp.columns: axes[1].bar(xc, comp[c].values, bottom=bot, color=COL[c], label=NAMES[c]); bot += comp[c].values
axes[1].set_xticks(xc); axes[1].set_xticklabels([f"{iv.mid:.2f}" for iv in comp.index], fontsize=8); axes[1].set_xlabel("ridge score (octile midpoint)"); axes[1].set_ylabel("true class composition")
axes[1].legend(frameon=False, fontsize=8, ncol=2); axes[1].set_title("class mix along the score", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "boundary.png", dpi=170); plt.close(fig)

# 5 confidence
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
axes[0].hist([main.conf[~wrong], main.conf[wrong]], bins=14, range=(0.3, 1), density=True, color=["#0f3d79", "#d65f0e"], label=["correct", "misclassified"])
axes[0].set_xlabel("max class probability"); axes[0].legend(frameon=False); axes[0].set_title(f"CP: AUROC for detecting errors = {m0.auc_maxprob:.2f}", fontsize=10)
bins = np.array([0.3, 0.45, 0.55, 0.65, 0.75, 1.0]); idx = np.digitize(main.conf, bins) - 1
rate = [wrong[idx == b].mean() for b in range(len(bins) - 1)]; cnt = [(idx == b).sum() for b in range(len(bins) - 1)]
axes[1].bar(range(len(rate)), rate, color="#d65f0e"); axes[1].set_xticks(range(len(rate)))
axes[1].set_xticklabels([f"{bins[i]:.2f}–{bins[i + 1]:.2f}\n(n={cnt[i]})" for i in range(len(rate))], fontsize=8)
axes[1].set_xlabel("max class probability"); axes[1].set_ylabel("misclassification rate"); axes[1].set_ylim(0, 1); axes[1].set_title("error rate by confidence bin", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "confidence.png", dpi=170); plt.close(fig)
summ["error_rate_by_conf_bin"] = {f"{bins[i]:.2f}-{bins[i + 1]:.2f}": [round(float(rate[i]), 3), int(cnt[i])] for i in range(len(rate))}
json.dump(summ, open(OUT / "summary.json", "w"), indent=2)

# 6 training source effect
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2)); srcs_ = ["B1", "B2", "B1+B2"]; x = np.arange(3)
for ax, (col_, ttl, lo) in zip(axes, [("ridge_spearman", "ridge rank correlation (Spearman)", 0), ("ridge_mae", "ridge MAE (rank units, lower is better)", 0), ("within1", "within-one-class accuracy", 0)]):
    for i, (test, colr) in enumerate((("B1", "#0f3d79"), ("B2", "#d65f0e"))):
        v = [M[(M.src == s) & (M.test == test)][col_].iloc[0] for s in srcs_]
        ax.bar(x + (i - 0.5) * 0.36, v, 0.34, color=colr, label=f"test on {test}")
        for xx, vv in zip(x + (i - 0.5) * 0.36, v): ax.text(xx, vv + 0.01, f"{vv:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([f"train {s}" for s in srcs_]); ax.set_title(ttl, fontsize=10); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "sources.png", dpi=170); plt.close(fig)

# 7 features
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, col_, ttl in zip(axes, ["pc1_loading", "ridge_coef"], ["PC1 loadings (top 10 by |loading|)", "ordinal ridge coefficients (top 10 by |coef|)"]):
    t = fa.reindex(fa[col_].abs().sort_values(ascending=False).index).head(10)[::-1]
    ax.barh(t.feature, t[col_], color=np.where(t[col_] > 0, "#0f3d79", "#d65f0e")); ax.set_title(ttl, fontsize=10); ax.tick_params(labelsize=8)
fig.tight_layout(); fig.savefig(OUT / "features.png", dpi=170); plt.close(fig)

pd.set_option("display.width", 220)
print(M.round(3).T.to_string()); print(json.dumps(summ, indent=1)); print(fa.reindex(fa.pc1_loading.abs().sort_values(ascending=False).index).head(8).round(2).to_string(index=False)); print(fa.reindex(fa.ridge_coef.abs().sort_values(ascending=False).index).head(8).round(2).to_string(index=False))

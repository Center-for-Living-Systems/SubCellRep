#!/usr/bin/env python3
"""
analyze_ordinal_features_cp.py

Which CP features carry the ordinal (rank) signal? Frame-grouped 5-fold CV (seed 0), ridge on rank (alpha 10),
trained on B1+B2 of the other images, tested on B1 patches of the held-out images (Spearman / MAE).

  1 univariate : |Spearman| of each feature with the rank (B1 labels; B2 labels)
  2 groups     : ablation by feature group (intensity 11 | Haralick d1, d2, d4 | texture all | all | all minus intensity)
  3 permutation: drop in held-out Spearman when a feature (or a whole group) is permuted in the test fold
  4 stability  : ridge coefficient sign / magnitude consistency across the 5 training folds
Outputs: results/ordinal_cp/{feature_importance.csv, group_ablation.csv, features_dominant.png}
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_common import *

OUT = Path(__file__).resolve().parents[1] / "results" / "ordinal_cp"
RANK = {c: i for i, c in enumerate(FA4)}
b1, b2 = load_fa4()
cp = pd.read_csv(FEAT / "cellprofiler" / "ds1.csv").set_index("filename")
feats = list(cp.columns)
GROUPS = {"intensity": [f for f in feats if f.startswith("intensity")],
          "haralick_d1": [f for f in feats if f.endswith("_d1")], "haralick_d2": [f for f in feats if f.endswith("_d2")],
          "haralick_d4": [f for f in feats if f.endswith("_d4")]}
GROUPS["texture_all"] = GROUPS["haralick_d1"] + GROUPS["haralick_d2"] + GROUPS["haralick_d4"]
assert sum(len(GROUPS[g]) for g in ("intensity", "haralick_d1", "haralick_d2", "haralick_d4")) == len(feats)
print({g: len(v) for g, v in GROUPS.items()})

# 1 univariate
uni = []
for nm, d in (("B1", b1), ("B2", b2)):
    d = d[d.fn.isin(cp.index)]; r = d.label.map(RANK).values
    for f in feats: uni.append((nm, f, spearmanr(cp.loc[d.fn, f], r)[0]))
uni = pd.DataFrame(uni, columns=["labels", "feature", "rho"]).pivot(index="feature", columns="labels", values="rho")

folds = make_folds(0); rng = np.random.RandomState(0)
sets = {"all": feats, "intensity_only": GROUPS["intensity"], "texture_only": GROUPS["texture_all"], "all_minus_intensity": GROUPS["texture_all"],
        "haralick_d1_only": GROUPS["haralick_d1"], "haralick_d2_only": GROUPS["haralick_d2"], "haralick_d4_only": GROUPS["haralick_d4"],
        "all_minus_d1": [f for f in feats if f not in GROUPS["haralick_d1"]], "all_minus_d2": [f for f in feats if f not in GROUPS["haralick_d2"]],
        "all_minus_d4": [f for f in feats if f not in GROUPS["haralick_d4"]]}
oof = {k: [] for k in sets}; ytrue = []; coefs = []; perm_feat = {f: [] for f in feats}; perm_grp = {g: [] for g in GROUPS}
for k in range(N_FOLDS):
    T = {g for g, f in folds.items() if f == k}
    tr = train_sources(b1, b2, T, rng)["B1+B2"]; te = b1[b1.g.isin(T)]
    tr, te = tr[tr.fn.isin(cp.index)], te[te.fn.isin(cp.index)]; ytr = tr.label.map(RANK).values; yte = te.label.map(RANK).values; ytrue.append(yte)
    for name, cols in sets.items():
        sc = StandardScaler().fit(cp.loc[tr.fn, cols]); m = Ridge(alpha=10.0).fit(sc.transform(cp.loc[tr.fn, cols]), ytr)
        oof[name].append(m.predict(sc.transform(cp.loc[te.fn, cols])))
        if name == "all": coefs.append(m.coef_)
    sc = StandardScaler().fit(cp.loc[tr.fn]); m = Ridge(alpha=10.0).fit(sc.transform(cp.loc[tr.fn]), ytr); Xte = sc.transform(cp.loc[te.fn]); base = spearmanr(m.predict(Xte), yte)[0]
    pr = np.random.RandomState(k)
    for j, f in enumerate(feats):
        d = []
        for _ in range(5):
            X2 = Xte.copy(); X2[:, j] = pr.permutation(X2[:, j]); d.append(base - spearmanr(m.predict(X2), yte)[0])
        perm_feat[f].append(np.mean(d))
    for g, cols in GROUPS.items():
        idx = [feats.index(c) for c in cols]; d = []
        for _ in range(5):
            X2 = Xte.copy(); X2[:, idx] = X2[np.ix_(pr.permutation(len(X2)), idx)]; d.append(base - spearmanr(m.predict(X2), yte)[0])
        perm_grp[g].append(np.mean(d))

y = np.concatenate(ytrue); ab = []
for name in sets:
    p = np.concatenate(oof[name]); ab.append(dict(model=name, n_features=len(sets[name]), spearman=spearmanr(p, y)[0], mae=np.abs(p - y).mean()))
ab = pd.DataFrame(ab); ab["note"] = ""; ab.loc[ab.model == "all_minus_intensity", "note"] = "= texture only"
ab = ab[ab.model != "all_minus_intensity"]; ab.to_csv(OUT / "group_ablation.csv", index=False)

C = np.array(coefs)                                     # folds x features
imp = pd.DataFrame({"feature": feats, "rho_B1": uni.loc[feats, "B1"].values, "rho_B2": uni.loc[feats, "B2"].values,
                    "perm_drop_mean": [np.mean(perm_feat[f]) for f in feats], "perm_drop_sd": [np.std(perm_feat[f]) for f in feats],
                    "coef_mean": C.mean(0), "coef_sd": C.std(0), "coef_sign_consistency": (np.sign(C) == np.sign(C.mean(0))).mean(0)})
imp["group"] = [next(g for g in ("intensity", "haralick_d1", "haralick_d2", "haralick_d4") if f in GROUPS[g]) for f in feats]
imp.to_csv(OUT / "feature_importance.csv", index=False)
grp = pd.DataFrame({"group": list(GROUPS), "perm_drop_mean": [np.mean(perm_grp[g]) for g in GROUPS], "perm_drop_sd": [np.std(perm_grp[g]) for g in GROUPS]})
grp.to_csv(OUT / "group_permutation.csv", index=False)
pd.set_option("display.width", 220)
print(ab.round(3).to_string(index=False)); print(grp.round(3).to_string(index=False))
print("top univariate |rho| (B1):"); print(imp.reindex(imp.rho_B1.abs().sort_values(ascending=False).index).head(10)[["feature", "group", "rho_B1", "rho_B2"]].round(2).to_string(index=False))
print("top permutation drop:"); print(imp.sort_values("perm_drop_mean", ascending=False).head(10)[["feature", "group", "perm_drop_mean", "perm_drop_sd", "rho_B1", "coef_mean", "coef_sign_consistency"]].round(3).to_string(index=False))
print("intensity_min:"); print(imp[imp.feature.isin(["intensity_min", "intensity_p10", "intensity_mean", "correlation_d2"])].round(3).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2)); gcol = {"intensity": "#7b3294", "haralick_d1": "#1f77b4", "haralick_d2": "#2ca02c", "haralick_d4": "#ff7f0e"}
t = imp.reindex(imp.rho_B1.abs().sort_values(ascending=False).index).head(10)[::-1]
axes[0].barh(t.feature, t.rho_B1, color=[gcol[g] for g in t.group]); axes[0].set_title("univariate Spearman with rank (B1 labels)", fontsize=10); axes[0].tick_params(labelsize=8)
t = imp.sort_values("perm_drop_mean", ascending=False).head(10)[::-1]
axes[1].barh(t.feature, t.perm_drop_mean, xerr=t.perm_drop_sd, color=[gcol[g] for g in t.group]); axes[1].set_title("held-out Spearman drop when permuted", fontsize=10); axes[1].tick_params(labelsize=8)
axes[2].bar(range(len(ab)), ab.spearman, color="#0f3d79"); axes[2].set_xticks(range(len(ab))); axes[2].set_xticklabels(ab.model, rotation=40, ha="right", fontsize=8)
for i, v in enumerate(ab.spearman): axes[2].text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
axes[2].set_title("ridge on feature subsets (held-out Spearman)", fontsize=10); axes[2].set_ylim(0, 0.8)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=c, label=g) for g, c in gcol.items()], loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(OUT / "features_dominant.png", dpi=160); print("saved")

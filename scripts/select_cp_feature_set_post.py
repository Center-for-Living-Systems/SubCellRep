#!/usr/bin/env python3
"""Aggregate fsel_curve_folds.csv (from select_cp_feature_set.py), pick final sets, plot."""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_cp_feature_set as S
from cv_common import *

OUT = S.OUT
R = pd.read_csv(OUT / "fsel_curve_folds.csv")
rows = []
for key, d in R.groupby(["task", "test", "method", "k"]):
    w = d.n_te.values; m = np.average(d.bal_acc, weights=w); n = len(d)
    se = d.bal_acc.std(ddof=1) / np.sqrt(n) if n > 1 else 0
    rows.append(dict(task=key[0], test=key[1], method=key[2], k=key[3], bal_acc=m, se=se,
                     auroc=np.average(d.auroc, weights=w) if d.auroc.notna().all() else np.nan,
                     within1=np.average(d.within1, weights=w) if d.within1.notna().all() else np.nan))
C = pd.DataFrame(rows); C.to_csv(OUT / "fsel_curve.csv", index=False)

P = pd.read_csv(OUT / "fsel_fold_orders.csv")
freq = []
for task, d in P.groupby("task"):
    cnt = pd.Series([f for o in d.order for f in o.split(", ")[:10]]).value_counts()
    freq += [dict(task=task, feature=f, folds_in_top10=int(c)) for f, c in cnt.items()]
pd.DataFrame(freq).to_csv(OUT / "fsel_pick_freq.csv", index=False)

b1, b2 = S.b1, S.b2; cp = S.cp; comb = S.comb; RANK = S.RANK
allfa = pd.concat([b2, b1[~b1.fn.isin(set(b2.fn))]]); allfa = allfa[allfa.fn.isin(cp.index)]
data = {"fa4": (cp.loc[allfa.fn], allfa.label.map(RANK).values, allfa.g.values),
        "adnoad": (cp.loc[comb.filename], comb.binary.values, comb.g.values)}
final = []
for task, (X, y, g) in data.items():
    order = S.forward(X, y, g)
    ref = C[(C.task == task) & (C.method == "full50")].set_index("test")
    fw = C[(C.task == task) & (C.method == "forward")]
    k_ok = next((k for k in range(1, S.KMAX + 1) if all(
        fw[(fw.test == t) & (fw.k == k)].bal_acc.iloc[0] >= ref.loc[t, "bal_acc"] - ref.loc[t, "se"] for t in ref.index)), S.KMAX)
    final.append(dict(task=task, k=k_ok, features=", ".join(order[:k_ok]), full_order=", ".join(order)))
    print(task, "k =", k_ok, order[:k_ok], flush=True)
pd.DataFrame(final).to_csv(OUT / "fsel_final_sets.csv", index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (task, tn) in zip(axes, [("fa4", "B1"), ("fa4", "B2"), ("adnoad", "all")]):
    d = C[(C.task == task) & (C.test == tn)]
    f = d[d.method == "forward"]; r = d[d.method == "random"]; full = d[d.method == "full50"].iloc[0]; o = d[d.method == "one_per_cluster"].iloc[0]
    ax.errorbar(f.k, f.bal_acc, f.se, label="forward selection", color="#2ca02c", marker="o", ms=3)
    ax.plot(r.k, r.bal_acc, label="random k", color="#999", ls="--")
    ax.axhline(full.bal_acc, color="k", label=f"full 50 ({full.bal_acc:.3f})")
    ax.axhspan(full.bal_acc - full.se, full.bal_acc + full.se, color="k", alpha=0.08)
    ax.scatter([o.k], [o.bal_acc], color="#d62728", zorder=5, label=f"one-per-cluster k={int(o.k)} ({o.bal_acc:.3f})")
    ax.set_title(f"{task}: test {tn}", fontsize=10); ax.set_xlabel("number of features"); ax.set_ylabel("balanced accuracy"); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fsel_curve.png", dpi=150)
pd.set_option("display.width", 200)
print(C[C.method.isin(["full50", "one_per_cluster"])].round(3).to_string(index=False))
for task in ("fa4", "adnoad"):
    for t in C[C.task == task].test.unique():
        print(task, t); print(C[(C.task == task) & (C.test == t) & (C.method.isin(["forward", "random"])) & (C.k.isin([1, 2, 3, 4, 5, 6, 8, 10, 15, 20, 25]))].pivot(index="k", columns="method", values="bal_acc").round(3).to_string())

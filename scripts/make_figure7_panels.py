#!/usr/bin/env python3
"""
make_figure7_panels.py

Panels for Figure 7: four FA subclasses and binary-vs-subtype features (CP features, DS1 vinc).
Writes PNG/CSV to results/figure7/:
  class_counts.png/.csv        label counts per class and source
  gate_per_subclass.png/.csv   ad vs no-ad gate (CP + balanced logreg), frame-grouped OOF (5 fold assignments), per subclass
  fa4_metrics.png              balanced accuracy / macro-F1 for CP, IL, CP+IL, AE (CV oof from results/cv_b1b2)
  fa4_per_class.png/.csv       per-class precision / recall from the same OOF predictions
  feature_stability.png        forward-selection rank of the top features in each CV fold, binary vs FA4
  feature_distributions.png    class-specific distributions of the top features
  patch_examples.png           random labelled patches per class
"""
import sys, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, tifffile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_common import *

REPO = Path(__file__).resolve().parents[1]; OUT = REPO / "results" / "figure7"; OUT.mkdir(parents=True, exist_ok=True)
CV = REPO / "results" / "cv_b1b2"; OD = REPO / "results" / "ordinal_cp"
ALL5 = ["No adhesion"] + FA4; SH = ["No adhesion", "nascent", "focal complex", "focal adhesion", "fibrillar"]
COL = ["#8e44ad", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]; CC = dict(zip(ALL5, COL)); SHC = dict(zip(FA4, SH[1:]))
cp = pd.read_csv(FEAT / "cellprofiler" / "ds1.csv").set_index("filename"); feats = list(cp.columns)
comb = pd.read_csv(PREP / "combined_labels.csv"); comb = comb[comb.filename.isin(cp.index)].reset_index(drop=True); comb["g"] = comb.filename.map(group_of)
rng = np.random.RandomState(0)

# ---------- 1. class counts per source
b1 = pd.read_csv(B1_FILE); b1 = b1[b1.classification != "Uncertain"].classification
b2 = pd.read_csv(B2_FILE).label
cnt = pd.DataFrame({"B1 (random)": b1.value_counts(), "B2 (prototype)": b2.value_counts(), "B1+B2 combined": comb.label.value_counts()}).reindex(ALL5).fillna(0).astype(int)
cnt.to_csv(OUT / "class_counts.csv")
fig, ax = plt.subplots(figsize=(9.5, 4.2)); w = 0.26
for j, c in enumerate(cnt.columns):
    for i, k in enumerate(ALL5):
        ax.bar(i + (j - 1) * w, cnt.loc[k, c], w * 0.95, color=COL[i], alpha=[0.45, 0.7, 1.0][j], edgecolor="k", lw=0.4)
        ax.text(i + (j - 1) * w, cnt.loc[k, c] + 15, str(cnt.loc[k, c]), ha="center", fontsize=7)
ax.set_xticks(range(5)); ax.set_xticklabels([f"{s}" for s in SH], fontsize=9); ax.set_ylabel("labelled patches")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="#888", alpha=a, ec="k", lw=0.4) for a in (0.45, 0.7, 1.0)], labels=list(cnt.columns), frameon=False, fontsize=9)
ax.set_title("Labels per class and annotator set (no adhesion is the binary negative; the four FA subclasses are the positives)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "class_counts.png", dpi=150); plt.close(fig)

# ---------- 2. gate per subclass (frame-grouped OOF over 5 fold assignments)
def make():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42))
X = cp.loc[comb.filename, feats].values; y = comb.binary.values; P = np.zeros(len(comb))
for seed in range(5):
    fo = make_folds(seed); f = comb.g.map(fo).values
    for k in range(N_FOLDS):
        te = f == k; P[te] += make().fit(X[~te], y[~te]).predict_proba(X[te])[:, 1] / 5
comb["p"] = P; comb["pred_ad"] = (P >= 0.5).astype(int)
def rate(d, c):
    d = d[d.label == c]; return (d.pred_ad == 0).mean() if c == "No adhesion" else d.pred_ad.mean()
frames = comb.g.unique(); idx = {g: np.where(comb.g.values == g)[0] for g in frames}; bs = []
for _ in range(1000):
    s = rng.choice(frames, len(frames)); d = comb.iloc[np.concatenate([idx[g] for g in s])]; bs.append([rate(d, c) for c in ALL5])
bs = np.array(bs); rows = []
for i, c in enumerate(ALL5):
    d = comb[comb.label == c]; rows.append(dict(cls=SH[i], n=len(d), correct_rate=rate(comb, c), ci_lo=np.percentile(bs[:, i], 2.5), ci_hi=np.percentile(bs[:, i], 97.5),
                                               n_wrong=int(round(len(d) * (1 - rate(comb, c)))), median_p_adhesion=d.p.median()))
G = pd.DataFrame(rows); G.to_csv(OUT / "gate_per_subclass.csv", index=False)
from sklearn.metrics import balanced_accuracy_score
gate_ba = balanced_accuracy_score(y, comb.pred_ad.values)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1, 1.15]))
ax = axs[0]; ax.bar(range(5), G.correct_rate, color=COL, yerr=[G.correct_rate - G.ci_lo, G.ci_hi - G.correct_rate], capsize=3)
for i, r in G.iterrows(): ax.text(i, 1.02, f"{r.correct_rate:.0%}\n{r.n_wrong}/{r.n}", ha="center", fontsize=8)
ax.set_ylim(0, 1.2); ax.set_xticks(range(5)); ax.set_xticklabels(SH, fontsize=8, rotation=15); ax.set_ylabel("fraction called correctly")
ax.set_title("Gate: correct call per class (label: wrong calls / n)\nno adhesion = called no adhesion; FA classes = called adhesion", fontsize=9)
ax = axs[1]; data = [comb[comb.label == c].p.values for c in ALL5]
bp = ax.boxplot(data, patch_artist=True, showfliers=False, medianprops=dict(color="k"))
for p_, c in zip(bp["boxes"], COL): p_.set_facecolor(c); p_.set_alpha(0.75)
ax.axhline(0.5, color="k", ls="--", lw=0.8); ax.set_xticklabels(SH, fontsize=8, rotation=15); ax.set_ylabel("held-out p(adhesion)")
ax.set_title("Gate score per class (frame-grouped CV, mean of 5 fold assignments)", fontsize=9)
fig.suptitle(f"Binary adhesion / no-adhesion classifier on the subclass labels: CP features + class-balanced logistic regression, balanced accuracy {gate_ba:.3f}", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "gate_per_subclass.png", dpi=150); plt.close(fig)

# ---------- 3. FA4 classifier: CP / IL / CP+IL / AE
S = pd.read_csv(CV / "summary.csv"); O = pd.read_csv(CV / "oof.csv")
REPS = [("CP", "CP"), ("IL", "ilastik"), ("CP+IL", "CP+IL"), ("AE_b12", "SupCon-AE")]; RC = ["#34495e", "#16a085", "#7f8c8d", "#c0392b"]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
for ax, test in zip(axs, ("B1", "B2")):
    for j, (rep, nm) in enumerate(REPS):
        r = S[(S.rep == rep) & (S.src == "B1+B2") & (S.test == test)].iloc[0]
        ax.bar(j - 0.19, r.bal_acc, 0.36, color=RC[j], yerr=[[r.bal_acc - r.ci_lo], [r.ci_hi - r.bal_acc]], capsize=3)
        ax.bar(j + 0.19, r.macro_f1, 0.36, color=RC[j], alpha=0.45)
        ax.text(j - 0.19, r.bal_acc + 0.025, f"{r.bal_acc:.2f}", ha="center", fontsize=8); ax.text(j + 0.19, r.macro_f1 + 0.01, f"{r.macro_f1:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(4)); ax.set_xticklabels([n for _, n in REPS]); ax.set_title(f"trained on B1+B2 (other cells), tested on {test} (n={int(r.n) if r.n < 1000 else int(r.n)} patch predictions)", fontsize=9)
    ax.set_ylim(0, 0.9)
axs[0].set_ylabel("solid: balanced accuracy (95% CI)   faded: macro-F1")
fig.suptitle("Four-class FA classification, frame-grouped 5-fold CV, same splits for all features", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "fa4_metrics.png", dpi=150); plt.close(fig)

rows = []
for rep, nm in REPS:
    for test in ("B1", "B2"):
        d = O[(O.rep == rep) & (O.src == "B1+B2") & (O.test == test)]
        p, r, f, s = precision_recall_fscore_support(d.y, d.pred, labels=FA4, zero_division=0)
        for i, c in enumerate(FA4): rows.append(dict(method=nm, test=test, cls=SH[i + 1], precision=p[i], recall=r[i], f1=f[i], support=int(s[i])))
PC = pd.DataFrame(rows); PC.to_csv(OUT / "fa4_per_class.csv", index=False)
fig, axs = plt.subplots(2, 2, figsize=(12, 6.4), sharey=True)
for ri, test in enumerate(("B1", "B2")):
    for ci_, met in enumerate(("recall", "precision")):
        ax = axs[ri, ci_]
        for j, (rep, nm) in enumerate(REPS):
            d = PC[(PC.method == nm) & (PC.test == test)]; ax.bar(np.arange(4) + (j - 1.5) * 0.2, d[met], 0.19, color=RC[j], label=nm)
        ax.set_xticks(range(4)); ax.set_xticklabels(SH[1:], fontsize=9); ax.set_ylim(0, 1); ax.set_title(f"{met} - test {test} (support: " + ", ".join(str(int(v)) for v in d.support) + ")", fontsize=9)
axs[0, 0].legend(fontsize=8, ncol=4, frameon=False, loc="upper center")
fig.tight_layout(); fig.savefig(OUT / "fa4_per_class.png", dpi=150); plt.close(fig)

# ---------- 4. feature stability + distributions + examples
FS = pd.read_csv(OD / "fsel_fold_orders.csv"); FQ = pd.read_csv(OD / "fsel_pick_freq.csv")
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
for ax, (task, ttl) in zip(axs, (("adnoad", "Adhesion vs no adhesion"), ("fa4", "Four FA subclasses"))):
    top = FQ[FQ.task == task].sort_values("folds_in_top10", ascending=False).feature.tolist()[:12]
    M = np.full((len(top), N_FOLDS), np.nan)
    for _, r in FS[FS.task == task].iterrows():
        order = r.order.split(", ")
        for i, f in enumerate(top):
            if f in order: M[i, r.fold] = order.index(f) + 1
    im = ax.imshow(M, cmap="viridis_r", vmin=1, vmax=25, aspect="auto")
    for i in range(len(top)):
        for j in range(N_FOLDS): ax.text(j, i, "-" if np.isnan(M[i, j]) else int(M[i, j]), ha="center", va="center", fontsize=8, color="w" if (not np.isnan(M[i, j]) and M[i, j] < 14) else "k")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=8); ax.set_xticks(range(N_FOLDS)); ax.set_xticklabels([f"fold {i}" for i in range(N_FOLDS)], fontsize=8)
    ax.set_title(f"{ttl}\nforward-selection rank per CV fold (1 = first pick, - = beyond 25)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "feature_stability.png", dpi=150); plt.close(fig)

FIN = pd.read_csv(OD / "fsel_final_sets.csv").set_index("task")
fa_f = FIN.loc["fa4", "full_order"].split(", ")[:4]; bi_f = FIN.loc["adnoad", "full_order"].split(", ")[:4]
Z = (cp[feats] - cp.loc[comb.filename, feats].mean()) / cp.loc[comb.filename, feats].std()
lab = comb.set_index("filename").label
fig, axs = plt.subplots(2, 4, figsize=(15, 6.2))
for r_, (fl, ttl) in enumerate(((bi_f, "adhesion vs no adhesion"), (fa_f, "four FA subclasses"))):
    for c_, f in enumerate(fl):
        ax = axs[r_, c_]; data = [Z.loc[lab[lab == k].index, f].values for k in ALL5]
        bp = ax.boxplot(data, patch_artist=True, showfliers=False, medianprops=dict(color="k"))
        for p_, c in zip(bp["boxes"], COL): p_.set_facecolor(c); p_.set_alpha(0.8)
        ax.set_xticklabels(["no\nadh.", "nasc.", "f.\ncompl.", "f.\nadh.", "fibr."], fontsize=8); ax.set_title(f, fontsize=9); ax.set_ylabel("z-score" if c_ == 0 else "")
    axs[r_, 0].text(-0.38, 0.5, f"top features:\n{ttl}", transform=axs[r_, 0].transAxes, rotation=90, va="center", ha="center", fontsize=10, fontweight="bold")
fig.suptitle("Class-specific distributions of the first four forward-selected CP features (all B1+B2 labelled patches; z-scored)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "feature_distributions.png", dpi=150); plt.close(fig)

PD_ = DATA / "ae_results" / "patches" / "cio" / "vinc"; NEX = 12; sel = {}
for k in ALL5: sel[k] = comb[comb.label == k].sample(NEX, random_state=3).filename.tolist()
imgs = {k: [tifffile.imread(PD_ / ("control" if fn.startswith("control") else "ycomp") / "tiff_patches32_mr10" / fn) for fn in v] for k, v in sel.items()}
vmax = np.percentile(np.concatenate([i.ravel() for v in imgs.values() for i in v]), 99.5)
fig, axs = plt.subplots(5, NEX, figsize=(NEX * 1.05, 5 * 1.1))
for r_, k in enumerate(ALL5):
    for c_ in range(NEX):
        ax = axs[r_, c_]; ax.imshow(np.clip(imgs[k][c_], 0, vmax), cmap="gray", vmin=0, vmax=vmax); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor(COL[r_]); sp.set_linewidth(2.5)
    axs[r_, 0].set_ylabel(SH[r_], fontsize=9, color=COL[r_], fontweight="bold", rotation=0, ha="right", va="center", labelpad=6)
fig.suptitle("Random labelled patches per class (32x32, common brightness scale)", fontsize=11)
fig.tight_layout(); fig.savefig(OUT / "patch_examples.png", dpi=150, bbox_inches="tight"); plt.close(fig)

json.dump(dict(gate_bal_acc=float(gate_ba), top_fa4=fa_f, top_binary=bi_f), open(OUT / "summary.json", "w"), indent=2)
print(cnt.to_string()); print(G.round(3).to_string(index=False)); print(PC.round(2).to_string(index=False)); print(fa_f, bi_f, gate_ba)

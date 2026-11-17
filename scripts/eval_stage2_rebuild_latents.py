#!/usr/bin/env python3
"""Held-out check for the stage-2 rebuild runs: logreg on latents fit on train-split
labelled patches, scored on val-split labelled patches (labels = labels_used.csv).

Outputs (results/stage2_rebuild/):
  val_metrics.csv          per-run balanced accuracy + per-class recall
  cm_{run}.png             row-normalised val confusion matrix (counts in cells)
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, recall_score
from sklearn.preprocessing import StandardScaler

import argparse, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_stage2_rebuild import load_run

ap = argparse.ArgumentParser()
ap.add_argument("--feat", choices=["latent", "conae", "cp", "ilastik"], default="latent")
FEAT = ap.parse_args().feat
PFX = "" if FEAT == "latent" else f"{FEAT}_"

R   = Path("/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run")
OUT = Path(__file__).resolve().parents[1] / "results" / "stage2_rebuild"
RUNS = ["stage2", "stage2_conf90", "stage2_b1", "stage2_b1_conf90", "stage2_b2", "stage2_b2_conf90", "whole"]
ORDER = ["No adhesion", "Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
SHORT = {"No adhesion": "no-ad", "Nascent Adhesion": "nascent", "focal complex": "f.complex",
         "focal adhesion": "f.adhesion", "fibrillar adhesion": "fibrillar"}
KEY = {"No adhesion": "noad", "Nascent Adhesion": "nasc", "focal complex": "fcomp",
       "focal adhesion": "fadh", "fibrillar adhesion": "fibr"}

OUT.mkdir(parents=True, exist_ok=True)
rows = []
for m in RUNS:
    lat, z = load_run(m, FEAT)
    lat = lat[lat.label.notna()]
    tr, va = lat[lat.split == "train"], lat[lat.split == "val"]
    sc = StandardScaler().fit(tr[z])
    clf = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(tr[z]), tr.label)
    pv = clf.predict(sc.transform(va[z]))
    classes = [c for c in ORDER if c in set(lat.label)]
    r = dict(run=m, n_train=len(tr), n_val=len(va),
             bal_acc_val=balanced_accuracy_score(va.label, pv),
             bal_acc_train=balanced_accuracy_score(tr.label, clf.predict(sc.transform(tr[z]))))
    for c in classes:
        r["rec_" + KEY[c]] = recall_score(va.label == c, pv == c)
        r["nval_" + KEY[c]] = int((va.label == c).sum())
    rows.append(r)

    cm = confusion_matrix(va.label, pv, labels=classes)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n(n={cm[i, j]})", ha="center", va="center", fontsize=16,
                    color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels([SHORT[c] for c in classes], rotation=30, ha="right", fontsize=18)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels([SHORT[c] for c in classes], fontsize=18)
    ax.set_xlabel("predicted"); ax.set_ylabel("true (val split)")
    ax.set_title(f"{FEAT} {m}: val bal. acc {r['bal_acc_val']:.3f}", fontsize=20)
    fig.tight_layout(); fig.savefig(OUT / f"cm_{PFX}{m}.png", dpi=200); plt.close(fig)

df = pd.DataFrame(rows); df.to_csv(OUT / f"val_metrics{'' if FEAT == 'latent' else '_' + FEAT}.csv", index=False)
print(df.round(3).to_string(index=False))

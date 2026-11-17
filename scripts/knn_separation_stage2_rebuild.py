#!/usr/bin/env python3
"""
knn_separation_stage2_rebuild.py

Is "the AE looks better separated in UMAP" a generalisation effect or a training-set effect?
For each representation and condition, a k-NN classifier (k=15, standardised features):
  train_LOO : leave-one-out balanced accuracy among train-split labelled patches
              (how separated the classes LOOK on the points the encoder was trained on)
  val       : train-split labelled patches as reference, val-split labelled patches as queries
Output: results/stage2_rebuild/knn_separation.csv
"""
import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embed_stage2_rebuild import load_run, R

OUT  = Path(__file__).resolve().parents[1] / "results" / "stage2_rebuild"
RUNS = ["stage2", "stage2_conf90", "stage2_b1", "stage2_b1_conf90", "stage2_b2", "stage2_b2_conf90", "whole"]
REPS = {"SupCon-AE": "latent", "ConAE": "conae", "CP": "cp", "ilastik": "ilastik"}
K = 15
rows = []
for run in RUNS:
    for name, feat in REPS.items():
        if feat == "conae" and not (R / "stage2_rebuild_conae" / "latents.csv").exists():
            continue
        lat, z = load_run(run, feat)
        lat = lat[lat.label.notna()]
        tr, va = lat[lat.split == "train"], lat[lat.split == "val"]
        sc = StandardScaler().fit(tr[z]); Xtr, Xva = sc.transform(tr[z]), sc.transform(va[z])
        knn = KNeighborsClassifier(n_neighbors=K + 1).fit(Xtr, tr.label)
        idx = knn.kneighbors(Xtr, return_distance=False)[:, 1:]            # drop self
        ytr = tr.label.values
        loo = [pd.Series(ytr[i]).mode().iloc[0] for i in idx]
        pv = KNeighborsClassifier(n_neighbors=K).fit(Xtr, tr.label).predict(Xva)
        rows.append(dict(run=run, rep=name, train_loo=balanced_accuracy_score(ytr, loo), val=balanced_accuracy_score(va.label, pv)))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "knn_separation.csv", index=False)
print(df.pivot(index="run", columns="rep", values=["train_loo", "val"]).round(3).to_string())

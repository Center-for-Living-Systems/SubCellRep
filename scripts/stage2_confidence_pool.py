#!/usr/bin/env python3
"""
stage2_confidence_pool.py

High-confidence stage-2 pools. Uses adnoad_predictions.csv (p_adhesion from the
CP+logreg ad/no-ad model) and grouped-CV out-of-fold probabilities on the B1+B2
labels to calibrate thresholds.

Outputs (labelling/stage2_rebuild/):
  confidence_threshold_table.csv   OOF precision/recall + pool size per threshold
  stage2_pool_conf{T}.csv          predicted-ad with p>=T  U  all FA4-labelled
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

DATA = Path("/net/projects/CLS/lding/data/fa_data_analysis")
PREP = DATA / "labelling" / "stage2_rebuild"
CP   = DATA / "ae_results/features/cellprofiler/ds1.csv"
FA4 = ["Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
THRESH = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98]

comb = pd.read_csv(PREP / "combined_labels.csv")
pred = pd.read_csv(PREP / "adnoad_predictions.csv")
cp = pd.read_csv(CP).set_index("filename")
lab = comb[comb["filename"].isin(cp.index)].reset_index(drop=True)
X = cp.loc[lab["filename"]].values; y = lab["binary"].values
g = lab["filename"].str.split("x").str[0].values
oof = np.zeros(len(lab))
for tr, te in GroupKFold(5).split(X, y, g):
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)).fit(X[tr], y[tr])
    oof[te] = m.predict_proba(X[te])[:, 1]

fa4 = set(lab.loc[lab["label"].isin(FA4), "filename"])
rows = []
for t in THRESH:
    sel = oof >= t
    prec = y[sel].mean() if sel.any() else np.nan
    rec = sel[y == 1].mean()
    fa4_rec = np.mean([o >= t for o, f in zip(oof, lab["filename"]) if f in fa4])
    hi = pred[pred["p_adhesion"] >= t]
    pool = set(hi["filename"]) | fa4
    rows.append(dict(threshold=t, oof_precision_ad=prec, oof_recall_ad=rec, oof_recall_fa4=fa4_rec,
                     n_pred_ad=len(hi), n_pool=len(pool)))
    pd.DataFrame({"filename": sorted(pool)}).assign(
        condition=lambda d: d["filename"].map(lambda f: "control" if f.startswith("control") else "ycomp"),
        in_pred_ad=lambda d: d["filename"].isin(set(hi["filename"])).astype(int),
        is_fa4_label=lambda d: d["filename"].isin(fa4).astype(int),
    ).to_csv(PREP / f"stage2_pool_conf{t}.csv", index=False)
tab = pd.DataFrame(rows); tab.to_csv(PREP / "confidence_threshold_table.csv", index=False)
print(tab.round(3).to_string(index=False))
print("p_adhesion quantiles (all patches):", pred["p_adhesion"].quantile([.1,.25,.5,.75,.9]).round(3).to_dict())

#!/usr/bin/env python3
"""
rebuild_stage2_prep.py

Steps 1-2 of the stage-2 rebuild.

1. Ad vs no-ad classifier: CellProfiler features (DS1 = vinc ctrl+ycomp) +
   logistic regression, trained on combined B1+B2 labels (full sets, not the
   matched ones; B2 takes priority on overlap; "Uncertain" dropped).
   Grouped (per image) 5-fold CV is reported, the final model is fit on all
   labels and applied to every patch.
2. Stage-2 patch pool = predicted-adhesion patches  U  all FA4-labelled patches.

Outputs (labelling/stage2_rebuild/):
  combined_labels.csv     filename, condition, label (5-class), binary
  adnoad_predictions.csv  filename, p_adhesion, pred_adhesion
  stage2_pool.csv         filename, condition, in_pred_ad, is_fa4_label
  adnoad_cv.json          grouped-CV metrics

Usage:
  python scripts/rebuild_stage2_prep.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA   = Path("/net/projects/CLS/lding/data/fa_data_analysis")
OUT    = DATA / "labelling" / "stage2_rebuild"
CP_CSV = DATA / "ae_results" / "features" / "cellprofiler" / "ds1.csv"
B1_FILE = DATA / "labelling" / "labels_vinc_20260521.csv"
B2_FILE = DATA / "labelling" / "vinc_combined_label_Annabel_20260911.csv"

FA4 = ["Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]


def _cond(fn: str) -> str:
    return "control" if fn.startswith("control") else "ycomp"


def _group(fn: str) -> str:
    return fn.split("x", 1)[0]          # control_f0001


def build_combined_labels() -> pd.DataFrame:
    b1 = pd.read_csv(B1_FILE)
    b1 = b1[b1["classification"] != "Uncertain"].copy()
    b1["filename"] = b1["unique_ID"].str.replace("-f", "_f", n=1, regex=False)
    b1 = b1.rename(columns={"classification": "label"})[["filename", "label"]]
    b2 = pd.read_csv(B2_FILE)[["filename", "label"]]
    comb = pd.concat([b2, b1[~b1["filename"].isin(set(b2["filename"]))]], ignore_index=True)
    comb["condition"] = comb["filename"].map(_cond)
    comb["binary"] = np.where(comb["label"] == "No adhesion", 0, 1)
    print(f"B1 {len(b1)}  B2 {len(b2)}  combined (B2 priority) {len(comb)}")
    print(comb["label"].value_counts().to_string())
    return comb


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    comb = build_combined_labels()
    comb.to_csv(OUT / "combined_labels.csv", index=False)

    cp = pd.read_csv(CP_CSV)
    feat_cols = [c for c in cp.columns if c != "filename"]
    cp = cp.set_index("filename")
    lab = comb[comb["filename"].isin(cp.index)].reset_index(drop=True)
    print(f"labelled patches with CP features: {len(lab)} / {len(comb)}")

    X = cp.loc[lab["filename"], feat_cols].values
    y = lab["binary"].values
    groups = lab["filename"].map(_group).values

    def make_clf():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        )

    accs = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        accs.append(balanced_accuracy_score(y[te], make_clf().fit(X[tr], y[tr]).predict(X[te])))
    print(f"grouped 5-fold CV bal_acc: {np.mean(accs):.3f} +- {np.std(accs):.3f}  {np.round(accs, 3)}")

    clf = make_clf().fit(X, y)
    p_all = clf.predict_proba(cp[feat_cols].values)[:, 1]
    pred = pd.DataFrame({"filename": cp.index, "p_adhesion": p_all, "pred_adhesion": (p_all >= 0.5).astype(int)})
    pred.to_csv(OUT / "adnoad_predictions.csv", index=False)
    print(f"predicted adhesion: {pred['pred_adhesion'].sum()} / {len(pred)}")

    fa4_fns = set(comb.loc[comb["label"].isin(FA4), "filename"]) & set(cp.index)
    pool = pred[["filename"]].copy()
    pool["condition"] = pool["filename"].map(_cond)
    pool["in_pred_ad"] = pred["pred_adhesion"].values
    pool["is_fa4_label"] = pool["filename"].isin(fa4_fns).astype(int)
    pool = pool[(pool["in_pred_ad"] == 1) | (pool["is_fa4_label"] == 1)].reset_index(drop=True)
    pool.to_csv(OUT / "stage2_pool.csv", index=False)

    n_fa4_missed = int(((pool["is_fa4_label"] == 1) & (pool["in_pred_ad"] == 0)).sum())
    summary = {
        "n_labelled": int(len(lab)),
        "cv_bal_acc_mean": float(np.mean(accs)), "cv_bal_acc_std": float(np.std(accs)),
        "n_pred_adhesion": int(pred["pred_adhesion"].sum()),
        "n_fa4_labelled": len(fa4_fns),
        "n_fa4_labelled_not_predicted_ad": n_fa4_missed,
        "stage2_pool": int(len(pool)),
        "stage2_pool_by_condition": pool["condition"].value_counts().to_dict(),
    }
    (OUT / "adnoad_cv.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

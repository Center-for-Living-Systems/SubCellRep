#!/usr/bin/env python3
"""
train_cp_two_stage.py

Final (no hold-out) two-stage CP + logistic-regression models, trained on ALL B1+B2 labels (B2 priority, file 0911).
  stage 1: adhesion vs no adhesion      (all labelled patches with CP features)
  stage 2: FA4 (nascent / focal complex / focal adhesion / fibrillar)  (all FA4-labelled patches)
Both: StandardScaler + class-balanced LogisticRegression on the 50 CP features (DS1 vinc ctrl+ycomp).
Applies both to every DS1 patch. Reported metrics are TRAINING-set (resubstitution) numbers, not generalisation.

Outputs -> ae_results/cp_two_stage/ : stage1.joblib, stage2.joblib, predictions.csv, summary.json,
          stage1_coefficients.csv, stage2_coefficients.csv
Usage: python scripts/train_cp_two_stage.py
"""
import json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("/net/projects/CLS/lding/data/fa_data_analysis")
OUT = DATA / "ae_results" / "cp_two_stage"; OUT.mkdir(parents=True, exist_ok=True)
CP = DATA / "ae_results" / "features" / "cellprofiler" / "ds1.csv"
FA4 = ["Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]


def make():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42))


cp = pd.read_csv(CP).set_index("filename"); feats = list(cp.columns)
comb = pd.read_csv(DATA / "labelling" / "stage2_rebuild" / "combined_labels.csv")
comb = comb[comb.filename.isin(cp.index)].reset_index(drop=True)

# stage 1
X1, y1 = cp.loc[comb.filename, feats].values, comb.binary.values
m1 = make().fit(X1, y1); p1tr = m1.predict_proba(X1)[:, 1]
# stage 2
fa = comb[comb.label.isin(FA4)].reset_index(drop=True)
X2, y2 = cp.loc[fa.filename, feats].values, fa.label.map({c: i for i, c in enumerate(FA4)}).values
m2 = make().fit(X2, y2); p2tr = m2.predict(X2)

joblib.dump({"model": m1, "features": feats, "task": "adhesion(1) vs no adhesion(0)"}, OUT / "stage1.joblib")
joblib.dump({"model": m2, "features": feats, "classes": FA4}, OUT / "stage2.joblib")
for name, m, cls in [("stage1", m1, ["adhesion_vs_noad"]), ("stage2", m2, FA4)]:
    co = m[-1].coef_; idx = cls if co.shape[0] > 1 else cls
    pd.DataFrame(co.T, index=feats, columns=idx if co.shape[0] > 1 else cls).to_csv(OUT / f"{name}_coefficients.csv")

# apply to all patches
P = pd.DataFrame({"filename": cp.index})
P["p_adhesion"] = m1.predict_proba(cp[feats].values)[:, 1]; P["pred_adhesion"] = (P.p_adhesion >= 0.5).astype(int)
pr2 = m2.predict_proba(cp[feats].values)
for i, c in enumerate(FA4): P[f"p_{c.split()[0].lower()}_{c.split()[-1][:3]}"] = pr2[:, i]
P["fa4_class"] = [FA4[i] for i in pr2.argmax(1)]
P["final_class"] = np.where(P.pred_adhesion == 1, P.fa4_class, "No adhesion")
lab = comb.set_index("filename").label
P["label"] = P.filename.map(lab)
P.to_csv(OUT / "predictions.csv", index=False)

cm2 = confusion_matrix(y2, p2tr, labels=range(4))
summ = {
    "note": "training-set (resubstitution) metrics only; no hold-out",
    "n_features": len(feats),
    "stage1": {"n_train": int(len(comb)), "n_adhesion": int(y1.sum()), "n_noad": int((1 - y1).sum()),
               "train_bal_acc": float(balanced_accuracy_score(y1, p1tr >= 0.5)), "train_auroc": float(roc_auc_score(y1, p1tr))},
    "stage2": {"n_train": int(len(fa)), "per_class": {c: int((y2 == i).sum()) for i, c in enumerate(FA4)},
               "train_bal_acc": float(balanced_accuracy_score(y2, p2tr)), "train_confusion_rows_true_cols_pred": cm2.tolist()},
    "all_patches": {"n": int(len(P)), "pred_adhesion": int(P.pred_adhesion.sum()),
                    "final_class_counts": P.final_class.value_counts().to_dict()},
    "cv_reference_not_from_this_fit": "grouped 5-fold CV: stage 1 bal acc 0.922 +- 0.022; stage 2 FA4 about 0.56 (B1 test) / 0.63 (B2 test)",
}
(OUT / "summary.json").write_text(json.dumps(summ, indent=2)); print(json.dumps(summ, indent=2))

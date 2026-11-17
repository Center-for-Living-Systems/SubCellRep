#!/usr/bin/env python3
"""
eval_le_b1b2v2_cross.py

Cross-batch evaluation for B1/B2v2 matched set.
CP and ilastik only — no SupCon.

Combos: b1→b1, b1→b2v2, b2v2→b1, b2v2→b2v2

Outputs (EVAL_DIR):
  le_b1b2v2_cross_{feat}_{clf}.csv
  le_b1b2v2_cross_errors_{feat}_{clf}.json

CSV columns:
  train_batch, test_batch, feat, clf, fold, n_train, n_test,
  bal_acc, recall_ad, recall_noad, prec_ad, prec_noad

Usage:
  python scripts/eval_le_b1b2v2_cross.py [--feats cp ilastik] [--clf lgbm logreg]
"""
from __future__ import annotations
import argparse, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler

REPO     = Path(__file__).resolve().parent.parent
DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
EVAL_DIR = REPO / "eval_results" / "b1b2v2"
ANN_DIR  = DATA / "labelling" / "le_b1b2v2_le"
FS_DIR   = DATA / "labelling" / "le_b1b2v2_matched"
CP_CSV   = DATA / "ae_results" / "features" / "cellprofiler" / "ds1.csv"
IL_CSV   = DATA / "ae_results" / "features" / "ilastik" / "ds1.csv"

FOLDS    = list(range(5))
BATCHES  = ["b1", "b2v2"]
PRED_DIR = REPO / "eval_results" / "patch_preds"


def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def _build_clf(clf_name: str):
    if clf_name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            min_child_samples=1, class_weight="balanced",
            n_jobs=4, random_state=42, verbose=-1,
        )
    elif clf_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            solver="lbfgs", random_state=42,
        )
    raise ValueError(f"Unknown clf: {clf_name}")


def _load_handcrafted(feat: str) -> pd.DataFrame | None:
    csv = CP_CSV if feat == "cp" else IL_CSV
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    feat_cols = [c for c in df.columns if c != "filename"]
    return df[["filename"] + feat_cols].set_index("filename")


def _run_combo(train_batch: str, test_batch: str, feat: str, clf_name: str,
               feats_all: pd.DataFrame, save_preds: bool = False):
    results     = []
    errors      = {}
    patch_rows  = []

    fs_test = pd.read_csv(FS_DIR / f"fold_splits_{test_batch}.csv")
    fs_test["fn"] = fs_test["unique_ID"].apply(_h2u)

    for fold in FOLDS:
        ann_path = ANN_DIR / f"le_b1b2v2_{train_batch}_fv{fold}_nball_r0.csv"
        if not ann_path.exists():
            print(f"    skip fold {fold} — missing {ann_path.name}")
            continue
        ann = pd.read_csv(ann_path)
        ann["fn"] = ann["unique_ID"].apply(_h2u)

        te = fs_test[fs_test["fold"] == fold].copy()
        train_fns = set(ann["fn"])
        te = te[~te["fn"].isin(train_fns)]

        X_tr_raw = feats_all.reindex(ann["fn"]).values
        X_te_raw = feats_all.reindex(te["fn"]).values

        tr_ok = ~np.isnan(X_tr_raw).any(axis=1)
        te_ok = ~np.isnan(X_te_raw).any(axis=1)
        X_tr, X_te = X_tr_raw[tr_ok], X_te_raw[te_ok]

        y_tr = np.array([1 if l == "adhesion" else 0
                         for l in ann["label"].tolist()])[tr_ok]
        y_te = np.array([1 if l == "adhesion" else 0
                         for l in te["label"].values])[te_ok]

        if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
            continue

        sc = StandardScaler().fit(X_tr)
        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

        clf = _build_clf(clf_name)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        prob = clf.predict_proba(X_te)[:, 1]

        te_ok_fns = te["fn"].values[te_ok]
        if save_preds:
            for fn, yt, yp, pb in zip(te_ok_fns, y_te, y_pred, prob):
                patch_rows.append(dict(
                    fn=fn, y_true=int(yt), y_pred=int(yp), prob_adhesion=float(pb),
                    train_batch=train_batch, test_batch=test_batch,
                    feat=feat, clf=clf_name, fold=fold,
                ))
        errors[fold] = {
            "ad2noad": [fn for fn, yt, yp in zip(te_ok_fns, y_te, y_pred) if yt == 1 and yp == 0],
            "noad2ad": [fn for fn, yt, yp in zip(te_ok_fns, y_te, y_pred) if yt == 0 and yp == 1],
        }

        results.append(dict(
            train_batch=train_batch, test_batch=test_batch, feat=feat, clf=clf_name, fold=fold,
            n_train=int(tr_ok.sum()), n_test=int(te_ok.sum()),
            bal_acc     = round(balanced_accuracy_score(y_te, y_pred), 6),
            recall_ad   = round(recall_score(y_te, y_pred, pos_label=1, zero_division=0), 6),
            recall_noad = round(recall_score(y_te, y_pred, pos_label=0, zero_division=0), 6),
            prec_ad     = round(precision_score(y_te, y_pred, pos_label=1, zero_division=0), 6),
            prec_noad   = round(precision_score(y_te, y_pred, pos_label=0, zero_division=0), 6),
        ))

    return pd.DataFrame(results), errors, patch_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", default=["cp", "ilastik"])
    ap.add_argument("--clf",   nargs="+", default=["lgbm"], choices=["lgbm", "logreg"])
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    combos = [(tr, te) for tr in BATCHES for te in BATCHES]

    for feat in args.feats:
        feats_all = _load_handcrafted(feat)
        if feats_all is None:
            print(f"Missing {feat} CSV — skipping"); continue

        for clf_name in args.clf:
            all_rows, all_errors, all_patch_rows = [], {}, []
            for train_b, test_b in combos:
                print(f"\n=== {train_b}→{test_b}  feat={feat}  clf={clf_name} ===")
                df, errs, patch_rows = _run_combo(
                    train_b, test_b, feat, clf_name, feats_all,
                    save_preds=args.save_preds,
                )
                if df.empty:
                    continue
                all_rows.append(df)
                all_errors[f"{train_b}_to_{test_b}"] = errs
                all_patch_rows.extend(patch_rows)
                s = df[["bal_acc","recall_ad","recall_noad","prec_ad","prec_noad"]].mean().round(3)
                print(f"  Mean: {s.to_dict()}")

            if all_rows:
                out = EVAL_DIR / f"le_b1b2v2_cross_{feat}_{clf_name}.csv"
                pd.concat(all_rows, ignore_index=True).to_csv(out, index=False)
                print(f"\nSaved → {out}")

            out_err = EVAL_DIR / f"le_b1b2v2_cross_errors_{feat}_{clf_name}.json"
            with open(out_err, "w") as f:
                json.dump(all_errors, f, indent=2)
            print(f"Saved → {out_err}")

            if args.save_preds and all_patch_rows:
                PRED_DIR.mkdir(parents=True, exist_ok=True)
                pred_df = pd.DataFrame(all_patch_rows)
                pred_out = PRED_DIR / f"le_cross_{feat}_{clf_name}.csv"
                pred_df.to_csv(pred_out, index=False)
                print(f"Saved preds → {pred_out}  ({len(pred_df)} rows)")


if __name__ == "__main__":
    main()

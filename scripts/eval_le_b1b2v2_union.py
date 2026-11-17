#!/usr/bin/env python3
"""
eval_le_b1b2v2_union.py

Train on B1∪B2v2 (conflict-excluded), test on B1 and B2v2 test sets.
CP and ilastik only — no SupCon.

Outputs (EVAL_DIR):
  le_b1b2v2_union_{feat}_{clf}.csv
  le_b1b2v2_union_errors_{feat}_{clf}.json

CSV columns: train_batch, test_batch, feat, clf, fold, n_train, n_conflict, n_test,
             bal_acc, recall_ad, recall_noad, prec_ad, prec_noad

Usage:
  python scripts/eval_le_b1b2v2_union.py [--feats cp ilastik] [--clf lgbm logreg]
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


def _run_union(feat: str, clf_name: str, feats_all: pd.DataFrame, save_preds: bool = False):
    results    = []
    errors     = {}
    patch_rows = []

    fs_b1   = pd.read_csv(FS_DIR / "fold_splits_b1.csv")
    fs_b2v2 = pd.read_csv(FS_DIR / "fold_splits_b2v2.csv")
    fs_b1["fn"]   = fs_b1["unique_ID"].apply(_h2u)
    fs_b2v2["fn"] = fs_b2v2["unique_ID"].apply(_h2u)

    for fold in FOLDS:
        ann_b1_path   = ANN_DIR / f"le_b1b2v2_b1_fv{fold}_nball_r0.csv"
        ann_b2v2_path = ANN_DIR / f"le_b1b2v2_b2v2_fv{fold}_nball_r0.csv"
        if not ann_b1_path.exists() or not ann_b2v2_path.exists():
            print(f"    skip fold {fold} — missing annotation file")
            continue

        ann_b1   = pd.read_csv(ann_b1_path)
        ann_b2v2 = pd.read_csv(ann_b2v2_path)
        ann_b1["fn"]   = ann_b1["unique_ID"].apply(_h2u)
        ann_b2v2["fn"] = ann_b2v2["unique_ID"].apply(_h2u)

        # Build union: exclude conflict patches
        b1_label   = dict(zip(ann_b1["fn"],   ann_b1["label"]))
        b2v2_label = dict(zip(ann_b2v2["fn"], ann_b2v2["label"]))
        conflicts  = {fn for fn in set(b1_label) & set(b2v2_label)
                      if b1_label[fn] != b2v2_label[fn]}

        b1_rows   = ann_b1[~ann_b1["fn"].isin(conflicts)][["fn", "label"]].copy()
        b2v2_only = ann_b2v2[~ann_b2v2["fn"].isin(set(b1_label) | conflicts)][["fn", "label"]].copy()
        union_ann = pd.concat([b1_rows, b2v2_only], ignore_index=True)

        if len(union_ann) == 0 or len(np.unique(
                [1 if l == "adhesion" else 0 for l in union_ann["label"]])) < 2:
            print(f"    skip fold {fold} — degenerate union")
            continue

        train_fns = set(union_ann["fn"])
        X_tr_raw  = feats_all.reindex(union_ann["fn"]).values
        y_tr_raw  = np.array([1 if l == "adhesion" else 0 for l in union_ann["label"]])
        tr_ok     = ~np.isnan(X_tr_raw).any(axis=1)
        X_tr, y_tr = X_tr_raw[tr_ok], y_tr_raw[tr_ok]

        if len(X_tr) == 0 or len(np.unique(y_tr)) < 2:
            continue

        sc = StandardScaler().fit(X_tr)
        X_tr_fit = sc.transform(X_tr)

        clf = _build_clf(clf_name)
        clf.fit(X_tr_fit, y_tr)

        n_train    = int(tr_ok.sum())
        n_conflict = len(conflicts)
        print(f"    fold {fold}: n_train={n_train}  conflicts={n_conflict}")

        for test_batch, fs_te in [("b1", fs_b1), ("b2v2", fs_b2v2)]:
            te = fs_te[fs_te["fold"] == fold].copy()
            te = te[~te["fn"].isin(train_fns)]
            if len(te) == 0:
                continue

            X_te_raw = feats_all.reindex(te["fn"]).values
            y_te_raw = np.array([1 if l == "adhesion" else 0 for l in te["label"].values])
            te_ok    = ~np.isnan(X_te_raw).any(axis=1)
            X_te, y_te = sc.transform(X_te_raw[te_ok]), y_te_raw[te_ok]

            if len(X_te) == 0:
                continue

            y_pred = clf.predict(X_te)
            prob   = clf.predict_proba(X_te)[:, 1]
            te_ok_fns = te["fn"].values[te_ok]
            if save_preds:
                patch_rows.extend(dict(
                    fn=fn, y_true=yt, y_pred=yp, prob_adhesion=pr,
                    test_batch=test_batch, feat=feat, clf=clf_name, fold=fold,
                ) for fn, yt, yp, pr in zip(te_ok_fns, y_te, y_pred, prob))
            errors.setdefault(test_batch, {})[fold] = {
                "ad2noad": [fn for fn, yt, yp in zip(te_ok_fns, y_te, y_pred) if yt == 1 and yp == 0],
                "noad2ad": [fn for fn, yt, yp in zip(te_ok_fns, y_te, y_pred) if yt == 0 and yp == 1],
            }

            results.append(dict(
                train_batch="union", test_batch=test_batch, feat=feat, clf=clf_name, fold=fold,
                n_train=n_train, n_conflict=n_conflict, n_test=int(te_ok.sum()),
                bal_acc     = round(balanced_accuracy_score(y_te, y_pred), 6),
                recall_ad   = round(recall_score(y_te, y_pred, pos_label=1, zero_division=0), 6),
                recall_noad = round(recall_score(y_te, y_pred, pos_label=0, zero_division=0), 6),
                prec_ad     = round(precision_score(y_te, y_pred, pos_label=1, zero_division=0), 6),
                prec_noad   = round(precision_score(y_te, y_pred, pos_label=0, zero_division=0), 6),
            ))

    return pd.DataFrame(results), errors, patch_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats",      nargs="+", default=["cp", "ilastik"])
    ap.add_argument("--clf",        nargs="+", default=["lgbm"], choices=["lgbm", "logreg"])
    ap.add_argument("--save_preds", action="store_true")
    args = ap.parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    for feat in args.feats:
        feats_all = _load_handcrafted(feat)
        if feats_all is None:
            print(f"Missing {feat} CSV — skipping"); continue

        for clf_name in args.clf:
            print(f"\n=== Union  feat={feat}  clf={clf_name} ===")
            df, errs, patch_rows = _run_union(feat, clf_name, feats_all, save_preds=args.save_preds)
            if df.empty:
                print("  No results"); continue

            for tb in ["b1", "b2v2"]:
                sub = df[df["test_batch"] == tb]
                if sub.empty: continue
                s = sub[["bal_acc","recall_ad","recall_noad","prec_ad","prec_noad"]].mean().round(3)
                print(f"  Union→{tb}: {s.to_dict()}")

            out = EVAL_DIR / f"le_b1b2v2_union_{feat}_{clf_name}.csv"
            df.to_csv(out, index=False)
            print(f"Saved → {out}")

            out_err = EVAL_DIR / f"le_b1b2v2_union_errors_{feat}_{clf_name}.json"
            with open(out_err, "w") as f:
                json.dump(errs, f, indent=2)
            print(f"Saved → {out_err}")

            if args.save_preds and patch_rows:
                PRED_DIR.mkdir(parents=True, exist_ok=True)
                out_pred = PRED_DIR / f"le_union_{feat}_{clf_name}.csv"
                pd.DataFrame(patch_rows).to_csv(out_pred, index=False)
                print(f"Saved → {out_pred}")


if __name__ == "__main__":
    main()

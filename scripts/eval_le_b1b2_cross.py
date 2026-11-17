#!/usr/bin/env python3
"""
eval_le_b1b2_cross.py

Cross-batch evaluation for multiple feature types (SupCon latents, CellProfiler,
ilastik): train LGBM on B1 annotations, test on B2 test patches (and vice versa).

Outputs (EVAL_DIR):
  le_b1b2_cross_lat12p8.csv / le_b1b2_cross_lat64p32.csv  — SupCon
  le_b1b2_cross_cp.csv                                      — CellProfiler
  le_b1b2_cross_ilastik.csv                                 — ilastik
  le_b1b2_cross_errors_{feat}.json  — error patch filenames (fold 0)

CSV columns:
  train_batch, test_batch, feat, fold, n_train, n_test,
  bal_acc, recall_ad, recall_noad, prec_ad, prec_noad

Usage
-----
  python scripts/eval_le_b1b2_cross.py [--lats lat12p8 lat64p32] [--feats cp ilastik supcon]
"""
from __future__ import annotations
import argparse, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler

DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
EVAL_DIR = DATA / "ae_results" / "features" / "eval_results"
RUN_ROOT = DATA / "ae_results" / "contrastive_run"
ANN_DIR  = DATA / "labelling" / "le_b1b2_le"
FS_DIR   = DATA / "labelling" / "le_b1b2_matched"
CP_CSV   = DATA / "ae_results" / "features" / "cellprofiler" / "ds1.csv"
IL_CSV   = DATA / "ae_results" / "features" / "ilastik" / "ds1.csv"

LGBM_PARAMS = dict(
    n_estimators=200, num_leaves=31, learning_rate=0.05,
    min_child_samples=1, class_weight="balanced",
    n_jobs=4, random_state=42, verbose=-1,
)

FOLDS   = list(range(5))


def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def _load_latents(batch: str, lat: str, fold: int) -> pd.DataFrame | None:
    name = f"le_b1b2_{batch}_{lat}_fv{fold}_nball_r0"
    p = RUN_ROOT / f"le_b1b2_{batch}_{lat}" / name / "latents.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    z_cols = [c for c in df.columns if c.startswith("z_") and "_proj" not in c]
    return df[["filename"] + z_cols].set_index("filename")


def _load_handcrafted(feat: str) -> pd.DataFrame | None:
    csv = CP_CSV if feat == "cp" else IL_CSV
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    feat_cols = [c for c in df.columns if c != "filename"]
    return df[["filename"] + feat_cols].set_index("filename")


def _run_combo(train_batch: str, test_batch: str, feat: str, lat: str = "lat12p8"):
    """feat: 'cp', 'ilastik', or a latent spec like 'lat12p8'."""
    results = []
    errors  = {}

    fs_test = pd.read_csv(FS_DIR / f"fold_splits_{test_batch}.csv")
    fs_test["fn"] = fs_test["unique_ID"].apply(_h2u)

    # For handcrafted features, load once (same for all folds)
    hc_feats = None
    if feat in ("cp", "ilastik"):
        hc_feats = _load_handcrafted(feat)
        if hc_feats is None:
            print(f"  Missing {feat} CSV")
            return pd.DataFrame(), {}

    for fold in FOLDS:
        if feat in ("cp", "ilastik"):
            feats_all = hc_feats
        else:
            feats_all = _load_latents(train_batch, feat, fold)
            if feats_all is None:
                print(f"    skip fold {fold} — missing {train_batch} {feat} latents")
                continue

        ann_path = ANN_DIR / f"le_b1b2_{train_batch}_fv{fold}_nball_r0.csv"
        if not ann_path.exists():
            print(f"    skip fold {fold} — missing ann")
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

        if feat in ("cp", "ilastik"):
            sc = StandardScaler().fit(X_tr)
            X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

        clf = LGBMClassifier(**LGBM_PARAMS)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

        te_ok_fns = te["fn"].values[te_ok]
        ad2noad = [fn for fn, yt, yp in zip(te_ok_fns, y_te, y_pred) if yt == 1 and yp == 0]
        noad2ad = [fn for fn, yt, yp in zip(te_ok_fns, y_te, y_pred) if yt == 0 and yp == 1]
        errors[fold] = {"ad2noad": ad2noad, "noad2ad": noad2ad}

        results.append(dict(
            train_batch=train_batch, test_batch=test_batch, feat=feat, fold=fold,
            n_train=int(tr_ok.sum()), n_test=int(te_ok.sum()),
            bal_acc     = round(balanced_accuracy_score(y_te, y_pred), 6),
            recall_ad   = round(recall_score(y_te, y_pred, pos_label=1, zero_division=0), 6),
            recall_noad = round(recall_score(y_te, y_pred, pos_label=0, zero_division=0), 6),
            prec_ad     = round(precision_score(y_te, y_pred, pos_label=1, zero_division=0), 6),
            prec_noad   = round(precision_score(y_te, y_pred, pos_label=0, zero_division=0), 6),
        ))

    return pd.DataFrame(results), errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lats",  nargs="+", default=["lat12p8", "lat64p32"])
    ap.add_argument("--feats", nargs="+", default=["supcon", "cp", "ilastik"],
                    help="Feature types to evaluate")
    args = ap.parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    combos = [("b1","b1"), ("b1","b2"), ("b2","b1"), ("b2","b2")]

    # --- handcrafted features (CP, ilastik) ---
    for feat in args.feats:
        if feat == "supcon":
            continue
        all_rows, all_errors = [], {}
        for train_b, test_b in combos:
            print(f"\n=== {train_b}→{test_b}  feat={feat} ===")
            df, errs = _run_combo(train_b, test_b, feat)
            if df.empty:
                continue
            all_rows.append(df)
            all_errors[f"{train_b}_to_{test_b}"] = errs
            s = df[["bal_acc","recall_ad","recall_noad","prec_ad","prec_noad"]].mean().round(3)
            print(f"  Mean:\n{s.to_string()}")
        if all_rows:
            out = EVAL_DIR / f"le_b1b2_cross_{feat}.csv"
            pd.concat(all_rows, ignore_index=True).to_csv(out, index=False)
            print(f"\nSaved → {out}")
        out_err = EVAL_DIR / f"le_b1b2_cross_errors_{feat}.json"
        with open(out_err, "w") as f:
            json.dump(all_errors, f, indent=2)
        print(f"Saved → {out_err}")

    # --- SupCon latents ---
    if "supcon" in args.feats:
        for lat in args.lats:
            all_rows, all_errors = [], {}
            for train_b, test_b in combos:
                print(f"\n=== {train_b}→{test_b}  lat={lat} ===")
                df, errs = _run_combo(train_b, test_b, lat)
                if df.empty:
                    continue
                all_rows.append(df)
                all_errors[f"{train_b}_to_{test_b}"] = errs
                s = df[["bal_acc","recall_ad","recall_noad","prec_ad","prec_noad"]].mean().round(3)
                print(f"  Mean:\n{s.to_string()}")
            if all_rows:
                out = EVAL_DIR / f"le_b1b2_cross_{lat}.csv"
                pd.concat(all_rows, ignore_index=True).to_csv(out, index=False)
                print(f"\nSaved → {out}")
            out_err = EVAL_DIR / f"le_b1b2_cross_errors_{lat}.json"
            with open(out_err, "w") as f:
                json.dump(all_errors, f, indent=2)
            print(f"Saved → {out_err}")


if __name__ == "__main__":
    main()

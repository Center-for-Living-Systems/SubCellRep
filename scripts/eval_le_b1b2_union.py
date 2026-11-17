#!/usr/bin/env python3
"""
eval_le_b1b2_union.py

Train on the union of B1+B2 annotations (conflict-excluded), test on B1 and B2 test sets.

For each fold f:
  union_train = (B1 train fold-f) ∪ (B2 train fold-f) – patches with conflicting labels
  B1_test     = B1 patches (fold == f) NOT in union_train
  B2_test     = B2 patches (fold == f) NOT in union_train

Outputs (EVAL_DIR):
  le_b1b2_union_lat12p8.csv / le_b1b2_union_lat64p32.csv  — SupCon
  le_b1b2_union_cp.csv                                      — CellProfiler
  le_b1b2_union_ilastik.csv                                 — ilastik
  le_b1b2_union_errors_{feat}.json

CSV columns: test_batch, feat, fold, n_train, n_test, bal_acc, recall_ad, recall_noad, prec_ad, prec_noad

Usage
-----
  python scripts/eval_le_b1b2_union.py [--lats lat12p8 lat64p32] [--feats cp ilastik supcon]
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

FOLDS = list(range(5))


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


def _run_union(feat: str, lat: str = "lat12p8"):
    """
    Train on B1∪B2 (per fold), test on B1 test and B2 test separately.
    feat: 'cp', 'ilastik', or latent spec like 'lat12p8'
    For SupCon, uses B1 latents (consistent embedding space across all patches).
    """
    results = []
    errors  = {}  # {test_batch: {fold: {ad2noad, noad2ad}}}

    fs_b1 = pd.read_csv(FS_DIR / "fold_splits_b1.csv")
    fs_b2 = pd.read_csv(FS_DIR / "fold_splits_b2.csv")
    fs_b1["fn"] = fs_b1["unique_ID"].apply(_h2u)
    fs_b2["fn"] = fs_b2["unique_ID"].apply(_h2u)

    hc_feats = None
    if feat in ("cp", "ilastik"):
        hc_feats = _load_handcrafted(feat)
        if hc_feats is None:
            print(f"  Missing {feat} CSV")
            return pd.DataFrame(), {}

    for fold in FOLDS:
        # Load per-fold SupCon latents (B1 embedding space for all patches)
        if feat not in ("cp", "ilastik"):
            feats_all = _load_latents("b1", feat, fold)
            if feats_all is None:
                print(f"    skip fold {fold} — missing b1 {feat} latents")
                continue
        else:
            feats_all = hc_feats

        # Load B1 and B2 nball training annotations
        ann_b1_path = ANN_DIR / f"le_b1b2_b1_fv{fold}_nball_r0.csv"
        ann_b2_path = ANN_DIR / f"le_b1b2_b2_fv{fold}_nball_r0.csv"
        if not ann_b1_path.exists() or not ann_b2_path.exists():
            print(f"    skip fold {fold} — missing annotation file")
            continue

        ann_b1 = pd.read_csv(ann_b1_path)
        ann_b2 = pd.read_csv(ann_b2_path)
        ann_b1["fn"] = ann_b1["unique_ID"].apply(_h2u)
        ann_b2["fn"] = ann_b2["unique_ID"].apply(_h2u)

        # Build union: exclude patches where B1 and B2 disagree
        b1_label = dict(zip(ann_b1["fn"], ann_b1["label"]))
        b2_label = dict(zip(ann_b2["fn"], ann_b2["label"]))
        conflicts = {fn for fn in set(b1_label) & set(b2_label)
                     if b1_label[fn] != b2_label[fn]}

        # Merge: B1 rows first, then B2 rows not already in B1 and not conflicting
        b1_rows = ann_b1[~ann_b1["fn"].isin(conflicts)][["fn", "label"]].copy()
        b2_only = ann_b2[~ann_b2["fn"].isin(set(b1_label) | conflicts)][["fn", "label"]].copy()
        union_ann = pd.concat([b1_rows, b2_only], ignore_index=True)

        if len(union_ann) == 0 or len(np.unique(
                [1 if l == "adhesion" else 0 for l in union_ann["label"]])) < 2:
            print(f"    skip fold {fold} — degenerate union training set")
            continue

        train_fns = set(union_ann["fn"])

        # Gather features for training
        X_tr_raw = feats_all.reindex(union_ann["fn"]).values
        y_tr_raw = np.array([1 if l == "adhesion" else 0 for l in union_ann["label"]])
        tr_ok = ~np.isnan(X_tr_raw).any(axis=1)
        X_tr, y_tr = X_tr_raw[tr_ok], y_tr_raw[tr_ok]

        if len(X_tr) == 0 or len(np.unique(y_tr)) < 2:
            continue

        if feat in ("cp", "ilastik"):
            sc = StandardScaler().fit(X_tr)
            X_tr_fit = sc.transform(X_tr)
        else:
            sc = None
            X_tr_fit = X_tr

        clf = LGBMClassifier(**LGBM_PARAMS)
        clf.fit(X_tr_fit, y_tr)

        n_train = int(tr_ok.sum())
        n_conflict = len(conflicts)

        # Test on B1 and B2 separately
        for test_batch, fs_test in [("b1", fs_b1), ("b2", fs_b2)]:
            te = fs_test[fs_test["fold"] == fold].copy()
            te = te[~te["fn"].isin(train_fns)]  # exclude train patches

            if len(te) == 0:
                continue

            X_te_raw = feats_all.reindex(te["fn"]).values
            y_te_raw = np.array([1 if l == "adhesion" else 0 for l in te["label"].values])
            te_ok = ~np.isnan(X_te_raw).any(axis=1)
            X_te, y_te = X_te_raw[te_ok], y_te_raw[te_ok]

            if len(X_te) == 0:
                continue

            if sc is not None:
                X_te = sc.transform(X_te)

            y_pred = clf.predict(X_te)

            te_ok_fns = te["fn"].values[te_ok]
            ad2noad = [fn for fn, yt, yp in zip(te_ok_fns, y_te, y_pred) if yt == 1 and yp == 0]
            noad2ad = [fn for fn, yt, yp in zip(te_ok_fns, y_te, y_pred) if yt == 0 and yp == 1]
            errors.setdefault(test_batch, {})[fold] = {"ad2noad": ad2noad, "noad2ad": noad2ad}

            results.append(dict(
                train_batch="union", test_batch=test_batch, feat=feat, fold=fold,
                n_train=n_train, n_conflict=n_conflict, n_test=int(te_ok.sum()),
                bal_acc     = round(balanced_accuracy_score(y_te, y_pred), 6),
                recall_ad   = round(recall_score(y_te, y_pred, pos_label=1, zero_division=0), 6),
                recall_noad = round(recall_score(y_te, y_pred, pos_label=0, zero_division=0), 6),
                prec_ad     = round(precision_score(y_te, y_pred, pos_label=1, zero_division=0), 6),
                prec_noad   = round(precision_score(y_te, y_pred, pos_label=0, zero_division=0), 6),
            ))

        print(f"    fold {fold}: n_train={n_train} (conflicts excluded={n_conflict})")

    return pd.DataFrame(results), errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lats",  nargs="+", default=["lat12p8", "lat64p32"])
    ap.add_argument("--feats", nargs="+", default=["supcon", "cp", "ilastik"])
    args = ap.parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    for feat in args.feats:
        if feat == "supcon":
            for lat in args.lats:
                print(f"\n=== Union  feat=SupCon/{lat} ===")
                df, errs = _run_union(lat)
                if df.empty:
                    continue
                for tb in ["b1", "b2"]:
                    sub = df[df["test_batch"] == tb]
                    if sub.empty:
                        continue
                    s = sub[["bal_acc","recall_ad","recall_noad","prec_ad","prec_noad"]].mean().round(3)
                    print(f"  Union→{tb}:\n{s.to_string()}")
                out = EVAL_DIR / f"le_b1b2_union_{lat}.csv"
                df.to_csv(out, index=False)
                print(f"Saved → {out}")
                out_err = EVAL_DIR / f"le_b1b2_union_errors_{lat}.json"
                with open(out_err, "w") as f:
                    json.dump(errs, f, indent=2)
                print(f"Saved → {out_err}")
        else:
            print(f"\n=== Union  feat={feat} ===")
            df, errs = _run_union(feat)
            if df.empty:
                continue
            for tb in ["b1", "b2"]:
                sub = df[df["test_batch"] == tb]
                if sub.empty:
                    continue
                s = sub[["bal_acc","recall_ad","recall_noad","prec_ad","prec_noad"]].mean().round(3)
                print(f"  Union→{tb}:\n{s.to_string()}")
            out = EVAL_DIR / f"le_b1b2_union_{feat}.csv"
            df.to_csv(out, index=False)
            print(f"Saved → {out}")
            out_err = EVAL_DIR / f"le_b1b2_union_errors_{feat}.json"
            with open(out_err, "w") as f:
                json.dump(errs, f, indent=2)
            print(f"Saved → {out_err}")


if __name__ == "__main__":
    main()

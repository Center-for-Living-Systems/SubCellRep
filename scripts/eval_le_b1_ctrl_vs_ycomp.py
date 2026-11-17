#!/usr/bin/env python3
"""
eval_le_b1_ctrl_vs_ycomp.py

For every DS1 B1 LE run (le_b1b2_b1_lat12p8), train the binary
(adhesion vs No adhesion) LGBM classifier and evaluate recall/precision
on 4 subgroups:
  ctrl + adhesion, ctrl + no-adhesion, ycomp + adhesion, ycomp + no-adhesion

Outputs (EVAL_DIR):
  le_b1_ctrlvy_supcon_ds1.csv
  le_b1_ctrlvy_cp_ds1.csv
  le_b1_ctrlvy_il_ds1.csv

Columns: fold, budget, repeat, n_train, n_test, bal_acc,
         acc_ctrl_ad, prec_ctrl_ad, acc_ctrl_noad, prec_ctrl_noad,
         acc_ycomp_ad, prec_ycomp_ad, acc_ycomp_noad, prec_ycomp_noad

Usage
-----
  python scripts/eval_le_b1_ctrl_vs_ycomp.py [--classifier lgbm|logreg]
"""
from __future__ import annotations

import argparse
import re
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

DATA       = Path("/net/projects/CLS/lding/data/fa_data_analysis")
EVAL_DIR   = DATA / "ae_results" / "features" / "eval_results"
RUN_ROOT   = DATA / "ae_results" / "contrastive_run" / "le_b1b2_b1_lat12p8"
ANN_DIR    = DATA / "labelling" / "le_b1b2_le"
FOLD_SPLITS = DATA / "labelling" / "le_b1b2_matched" / "fold_splits_b1.csv"
CONFIG_DIR  = Path(__file__).resolve().parents[1] / "config" / "le_b1b2_b1_lat12p8"

CP_CSV = DATA / "ae_results" / "features" / "cellprofiler" / "ds1.csv"
IL_CSV = DATA / "ae_results" / "features" / "ilastik" / "ds1.csv"

LGBM_PARAMS = dict(
    n_estimators=200, num_leaves=31, learning_rate=0.05,
    min_child_samples=1, class_weight="balanced",
    n_jobs=4, random_state=42, verbose=-1,
)

SUBGROUPS = [
    ("ctrl_ad",    "control", "adhesion"),
    ("ctrl_noad",  "control", "No adhesion"),
    ("ycomp_ad",   "ycomp",   "adhesion"),
    ("ycomp_noad", "ycomp",   "No adhesion"),
]


def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def _condition(uid: str) -> str:
    return uid.split("-f")[0]


def _load_fold_splits() -> pd.DataFrame:
    fs = pd.read_csv(FOLD_SPLITS)
    fs["fn"]        = fs["unique_ID"].apply(_h2u)
    fs["condition"] = fs["unique_ID"].apply(_condition)
    return fs


def _load_job_list() -> list[tuple]:
    job_re = re.compile(r"le_b1b2_b1_lat12p8_fv(\d)_nb(\w+)_r(\d)")
    names  = []
    jl = CONFIG_DIR / "job_list.txt"
    if not jl.exists():
        print(f"WARNING: job_list not found at {jl}")
        return names
    for line in jl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        name = Path(line).stem
        m = job_re.search(name)
        if m:
            names.append((name, int(m.group(1)), m.group(2), int(m.group(3))))
    return names


def _run_eval(fold_splits, feat_df, job_names, classifier,
              latent_loader=False) -> pd.DataFrame:
    results   = []
    n_skipped = 0

    for i, (name, fold, budget, repeat) in enumerate(job_names):
        if i % 50 == 0:
            print(f"  {i}/{len(job_names)} ...", flush=True)

        if latent_loader:
            lat_path = RUN_ROOT / name / "latents.csv"
            if not lat_path.exists():
                n_skipped += 1
                continue
            lat_raw = pd.read_csv(lat_path)
            z_cols  = [c for c in lat_raw.columns if c.startswith("z_") and "_proj" not in c]
            feats   = lat_raw[["filename"] + z_cols].set_index("filename")
        else:
            feats = feat_df

        ann_name = name.replace("_lat12p8", "")
        ann_path = ANN_DIR / f"{ann_name}.csv"
        if not ann_path.exists():
            n_skipped += 1
            continue

        ann = pd.read_csv(ann_path)
        train_fns = set(ann["unique_ID"].apply(_h2u))

        test_fs  = fold_splits[fold_splits["fold"] == fold].copy()
        test_fs  = test_fs[~test_fs["fn"].isin(train_fns)]
        train_fs = fold_splits[fold_splits["fn"].isin(train_fns)].copy()

        X_tr_raw = feats.reindex(train_fs["fn"]).values
        X_te_raw = feats.reindex(test_fs["fn"]).values

        tr_ok = ~np.isnan(X_tr_raw).any(axis=1)
        te_ok = ~np.isnan(X_te_raw).any(axis=1)
        X_tr = X_tr_raw[tr_ok]
        X_te = X_te_raw[te_ok]

        y_tr = np.array([1 if l == "adhesion" else 0
                         for l in train_fs["label"].values])[tr_ok]
        y_te = np.array([1 if l == "adhesion" else 0
                         for l in test_fs["label"].values])[te_ok]
        cond_te  = test_fs["condition"].values[te_ok]
        label_te = test_fs["label"].values[te_ok]

        if len(np.unique(y_tr)) < 2:
            n_skipped += 1
            continue

        if classifier == "logreg":
            sc = StandardScaler().fit(X_tr)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=42)
            clf.fit(sc.transform(X_tr), y_tr)
            y_pred = clf.predict(sc.transform(X_te))
        else:
            clf = LGBMClassifier(**LGBM_PARAMS)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

        bal_acc = balanced_accuracy_score(y_te, y_pred)

        row = dict(fold=fold, budget=budget, repeat=repeat,
                   n_train=int(tr_ok.sum()), n_test=int(te_ok.sum()),
                   bal_acc=round(bal_acc, 6))

        n_pred_ad   = int((y_pred == 1).sum())
        n_pred_noad = int((y_pred == 0).sum())

        for key, cond, label_str in SUBGROUPS:
            mask = (cond_te == cond) & (label_te == label_str)
            if mask.sum() == 0:
                row[f"acc_{key}"]  = np.nan
                row[f"prec_{key}"] = np.nan
                continue
            expected = 1 if label_str == "adhesion" else 0
            row[f"acc_{key}"] = round(float((y_pred[mask] == expected).mean()), 6)
            n_denom = n_pred_ad if expected == 1 else n_pred_noad
            tp_sg   = int((y_pred[mask] == expected).sum())
            row[f"prec_{key}"] = round(tp_sg / n_denom, 6) if n_denom > 0 else np.nan

        results.append(row)

    print(f"  Done: {len(results)} rows, {n_skipped} skipped")
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", choices=["lgbm", "logreg"], default="lgbm")
    args = ap.parse_args()
    clf = args.classifier

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    fold_splits = _load_fold_splits()
    job_names   = _load_job_list()
    tag         = f"_{clf}" if clf != "lgbm" else ""
    print(f"Total jobs: {len(job_names)}")

    cp_feat = pd.read_csv(CP_CSV).set_index("filename")
    il_feat = pd.read_csv(IL_CSV).set_index("filename")

    print("\nEvaluating SupCon latents …")
    df_sc = _run_eval(fold_splits, None, job_names, clf, latent_loader=True)
    sc_out = EVAL_DIR / f"le_b1_ctrlvy_supcon{tag}_ds1.csv"
    df_sc.to_csv(sc_out, index=False)
    print(f"  Saved → {sc_out}")

    print("\nEvaluating CP features …")
    df_cp = _run_eval(fold_splits, cp_feat, job_names, clf)
    cp_out = EVAL_DIR / f"le_b1_ctrlvy_cp{tag}_ds1.csv"
    df_cp.to_csv(cp_out, index=False)
    print(f"  Saved → {cp_out}")

    print("\nEvaluating Ilastik features …")
    df_il = _run_eval(fold_splits, il_feat, job_names, clf)
    il_out = EVAL_DIR / f"le_b1_ctrlvy_il{tag}_ds1.csv"
    df_il.to_csv(il_out, index=False)
    print(f"  Saved → {il_out}")

    for label, df in [("SupCon", df_sc), ("CP", df_cp), ("Ilastik", df_il)]:
        if df.empty:
            continue
        num = df[df["budget"] != "all"].copy()
        num["b"] = num["budget"].astype(int)
        acc_cols = [f"acc_{k}" for k, _, _ in SUBGROUPS]
        s = num.groupby("b")[acc_cols].mean()
        s.columns = [c.replace("acc_", "") for c in s.columns]
        print(f"\n{label} mean recall by subgroup:\n{s.round(3).to_string()}")


if __name__ == "__main__":
    main()

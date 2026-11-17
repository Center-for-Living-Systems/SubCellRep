#!/usr/bin/env python3
"""
eval_le_b1b2_supcon.py

Evaluate SupCon-AE latents on the B1/B2 label-efficiency benchmark
(le_b1b2_b1_lat12p8, le_b1b2_b1_lat64p32).

Reads job lists from config/le_b1b2_{batch}_{lat}/job_list.txt,
annotation CSVs from labelling/le_b1b2_le/,
fold splits from labelling/le_b1b2_matched/fold_splits_{batch}.csv.

Outputs (EVAL_DIR):
  supcon_le_b1b2_b1_lat12p8_ds1.csv
  supcon_le_b1b2_b1_lat64p32_ds1.csv
  (and b2 variants if runs exist)

Columns: name, batch, fold, budget, repeat, n_train, n_test, bal_acc

Usage
-----
  python scripts/eval_le_b1b2_supcon.py [--classifier lgbm|logreg]
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

DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
REPO     = Path(__file__).resolve().parents[1]
EVAL_DIR = DATA / "ae_results" / "features" / "eval_results"
RUN_ROOT = DATA / "ae_results" / "contrastive_run"
ANN_DIR  = DATA / "labelling" / "le_b1b2_le"
FS_DIR   = DATA / "labelling" / "le_b1b2_matched"

JOB_RE = re.compile(r"le_b1b2_(b\d)_(lat\w+)_fv(\d)_nb(\w+)_r(\d)")

LGBM_PARAMS = dict(
    n_estimators=200, num_leaves=31, learning_rate=0.05,
    min_child_samples=1, class_weight="balanced",
    n_jobs=4, random_state=42, verbose=-1,
)


def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def _run_batch(batch, lat, classifier):
    run_root = RUN_ROOT / f"le_b1b2_{batch}_{lat}"
    cfg_dir  = REPO / "config" / f"le_b1b2_{batch}_{lat}"
    jl       = cfg_dir / "job_list.txt"
    if not jl.exists():
        print(f"  No job list: {jl}")
        return pd.DataFrame()

    fs = pd.read_csv(FS_DIR / f"fold_splits_{batch}.csv")
    fs["fn"] = fs["unique_ID"].apply(_h2u)

    jobs    = [l.strip() for l in jl.read_text().splitlines() if l.strip()]
    results = []
    n_skip  = 0

    print(f"  {batch} {lat}: {len(jobs)} jobs", flush=True)
    for i, job_path in enumerate(jobs):
        if i % 50 == 0:
            print(f"    {i}/{len(jobs)} ...", flush=True)

        name = Path(job_path).stem
        m = JOB_RE.search(name)
        if not m:
            n_skip += 1
            continue

        b, l, fold, budget, repeat = (
            m.group(1), m.group(2), int(m.group(3)), m.group(4), int(m.group(5))
        )

        lat_path = run_root / name / "latents.csv"
        if not lat_path.exists():
            n_skip += 1
            continue

        ann_name = f"le_b1b2_{batch}_fv{fold}_nb{budget}_r{repeat}"
        ann_path = ANN_DIR / f"{ann_name}.csv"
        if not ann_path.exists():
            n_skip += 1
            continue

        lat_raw = pd.read_csv(lat_path)
        z_cols  = [c for c in lat_raw.columns if c.startswith("z_") and "_proj" not in c]
        feats   = lat_raw[["filename"] + z_cols].set_index("filename")

        ann = pd.read_csv(ann_path)
        ann["fn"] = ann["unique_ID"].apply(_h2u)

        test_fs  = fs[fs["fold"] == fold].copy()
        train_fns = set(ann["fn"])
        test_fs  = test_fs[~test_fs["fn"].isin(train_fns)]

        X_tr_raw = feats.reindex(ann["fn"]).values
        X_te_raw = feats.reindex(test_fs["fn"]).values

        tr_ok = ~np.isnan(X_tr_raw).any(axis=1)
        te_ok = ~np.isnan(X_te_raw).any(axis=1)
        X_tr  = X_tr_raw[tr_ok]
        X_te  = X_te_raw[te_ok]

        y_tr = np.array([1 if l == "adhesion" else 0
                         for l in ann["label"].tolist()])[tr_ok]
        y_te = np.array([1 if l == "adhesion" else 0
                         for l in test_fs["label"].values])[te_ok]

        if len(X_tr) == 0 or len(X_te) == 0 or len(np.unique(y_tr)) < 2:
            n_skip += 1
            continue

        if classifier == "logreg":
            sc  = StandardScaler().fit(X_tr)
            clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=42)
            clf.fit(sc.transform(X_tr), y_tr)
            y_pred = clf.predict(sc.transform(X_te))
        else:
            clf = LGBMClassifier(**LGBM_PARAMS)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

        results.append(dict(
            name=name, batch=batch, fold=fold, budget=budget, repeat=repeat,
            n_train=int(tr_ok.sum()), n_test=int(te_ok.sum()),
            bal_acc=round(balanced_accuracy_score(y_te, y_pred), 6),
        ))

    print(f"    Done: {len(results)} rows, {n_skip} skipped")
    return pd.DataFrame(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", choices=["lgbm", "logreg"], default="lgbm")
    ap.add_argument("--batches", nargs="+", default=["b1", "b2"],
                    help="Batches to evaluate (default: b1 b2)")
    ap.add_argument("--lats", nargs="+", default=["lat12p8", "lat64p32"],
                    help="Latent configs to evaluate")
    args = ap.parse_args()
    clf  = args.classifier
    tag  = f"_logreg" if clf == "logreg" else ""

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    for batch in args.batches:
        for lat in args.lats:
            run_dir = RUN_ROOT / f"le_b1b2_{batch}_{lat}"
            if not run_dir.exists():
                print(f"\nSkipping {batch} {lat} — no run dir")
                continue
            print(f"\n=== SupCon {batch} {lat} ({clf}) ===")
            df = _run_batch(batch, lat, clf)
            if df.empty:
                print("  No results.")
                continue
            out = EVAL_DIR / f"supcon_le_b1b2_{batch}_{lat}{tag}_ds1.csv"
            df.to_csv(out, index=False)
            print(f"  Saved → {out}")

            num = df[df["budget"] != "all"].copy()
            num["b"] = num["budget"].astype(int)
            s = num.groupby("b")["bal_acc"].mean()
            print(f"  Mean bal_acc by budget:\n{s.round(3).to_string()}")


if __name__ == "__main__":
    main()

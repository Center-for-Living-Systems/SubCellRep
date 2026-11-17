#!/usr/bin/env python3
"""
setup_le_b1b2v2_matched.py

Generate 5-fold annotation CSVs for the B1/B2v2 matched-label experiment.
B1 = Margaret (vinc_b1_matched_ds1_v2.csv, 403 patches)
B2v2 = Annabel Sep-2026 (vinc_b2v2_matched_ds1.csv, 403 patches)

No SupCon training — CP and ilastik features are pre-extracted.

Outputs:
  labelling/le_b1b2v2_matched/fold_splits_b1.csv
  labelling/le_b1b2v2_matched/fold_splits_b2v2.csv
  labelling/le_b1b2v2_le/le_b1b2v2_{b1|b2v2}_fv{fold}_nball_r0.csv

Usage:
  python scripts/setup_le_b1b2v2_matched.py [--dry-run]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LABEL_DIR  = DATA_ROOT / "labelling"
RUN_TAG    = "le_b1b2v2_matched"
OUT_SPLITS = LABEL_DIR / RUN_TAG
OUT_LE     = LABEL_DIR / "le_b1b2v2_le"

ANN_FILES = {
    "b1":   LABEL_DIR / "vinc_b1_matched_ds1_v2.csv",
    "b2v2": LABEL_DIR / "vinc_b2v2_matched_ds1.csv",
}

N_FOLDS = 5
CV_SEED = 42

# Images with too few patches (≤3) — excluded from both annotators
EXCLUDE_FRAME_PREFIXES = [
    "control-f0007",   # control-f7  (2 patches)
    "ycomp-f0000",     # ycomp-f0    (2 patches)
    "control-f0019",   # control-f19 (3 patches)
    "control-f0006",   # control-f6  (3 patches)
]


def _exclude(df: pd.DataFrame) -> pd.DataFrame:
    mask = df["unique_ID"].apply(
        lambda uid: any(uid.startswith(p) for p in EXCLUDE_FRAME_PREFIXES)
    )
    n_excluded = mask.sum()
    if n_excluded:
        print(f"  Excluding {n_excluded} patches from low-patch-count images: "
              + ", ".join(EXCLUDE_FRAME_PREFIXES))
    return df[~mask].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    dry = args.dry_run

    if not dry:
        OUT_SPLITS.mkdir(parents=True, exist_ok=True)
        OUT_LE.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    for batch, ann_file in ANN_FILES.items():
        if not ann_file.exists():
            print(f"[MISSING] {ann_file}")
            continue

        df = pd.read_csv(ann_file)
        df = _exclude(df)
        print(f"\n{'='*60}")
        print(f"Batch: {batch}  total={len(df)}  "
              f"adh={(df['label']=='adhesion').sum()}  "
              f"noad={(df['label']=='No adhesion').sum()}")

        fold_labels = np.empty(len(df), dtype=int)
        for fold, (_, test_idx) in enumerate(skf.split(np.arange(len(df)), df["label"].values)):
            fold_labels[test_idx] = fold

        splits_df = df[["unique_ID", "label"]].copy()
        splits_df["fold"] = fold_labels
        splits_path = OUT_SPLITS / f"fold_splits_{batch}.csv"
        if not dry:
            splits_df.to_csv(splits_path, index=False)
        print(f"  fold splits → {splits_path}")

        for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(df)), df["label"].values)):
            train_pool = df.iloc[train_idx][["unique_ID", "label"]].copy()
            n_adh  = (train_pool["label"] == "adhesion").sum()
            n_noad = (train_pool["label"] == "No adhesion").sum()
            print(f"  fold {fold}: train={len(train_pool)} (adh={n_adh} noad={n_noad})"
                  f"  test={len(test_idx)}")

            nball_path = OUT_LE / f"le_b1b2v2_{batch}_fv{fold}_nball_r0.csv"
            if not dry:
                train_pool.to_csv(nball_path, index=False)

    print(f"\nDone.  Fold splits → {OUT_SPLITS}")
    print(f"       Nball files → {OUT_LE}")
    print(f"\nNext: submit eval jobs:")
    print(f"  sbatch scripts/sbatch_b1b2v2_eval.sh")


if __name__ == "__main__":
    main()

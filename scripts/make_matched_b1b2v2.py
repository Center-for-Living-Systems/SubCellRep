#!/usr/bin/env python3
"""
make_matched_b1b2v2.py

Build the B1 / B2v2 matched label set using the same methodology as the
original B1/B2 matched set:

  1. Find frames annotated by both B1 (Margaret, labels_vinc_20260521.csv)
     and B2v2 (Annabel, vinc_combined_label_Annabel_20260911.csv).
  2. For each common frame take min(n_B1, n_B2v2) patches from each annotator
     (sampled independently, different patch locations).
  3. Save:
       labelling/vinc_b1_matched_ds1_v2.csv   — B1 matched subset (binary label)
       labelling/vinc_b2v2_matched_ds1.csv     — B2v2 matched subset (binary label)

Usage
-----
  python scripts/make_matched_b1b2v2.py [--dry-run] [--seed 42]
"""
from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

LAB  = Path("/net/projects/CLS/lding/data/fa_data_analysis/labelling")
SEED = 42


def parse_frame(df: pd.DataFrame, uid_col: str, sep: str) -> pd.DataFrame:
    """Add cond and frame columns parsed from a filename/uid column."""
    pat = rf'^(control|ycomp){sep}f(\d+)'
    df[["cond", "frame"]] = df[uid_col].str.extract(pat, expand=True)
    df["frame"] = df["frame"].astype(int)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # ── Load B1 (Margaret) ───────────────────────────────────────────────────
    b1_raw = pd.read_csv(LAB / "labels_vinc_20260521.csv")
    b1_raw = b1_raw[b1_raw["dataset"] == "vinc"].copy()
    b1_raw["label"] = b1_raw["classification"].map(
        lambda x: "No adhesion" if x == "No adhesion" else "adhesion")
    b1_raw = parse_frame(b1_raw, "unique_ID", "-")

    # ── Load B2v2 (Annabel Sep 2026) ─────────────────────────────────────────
    b2v2_raw = pd.read_csv(LAB / "vinc_combined_label_Annabel_20260911.csv")
    b2v2_raw["unique_ID"] = b2v2_raw["filename"].str.replace("_f", "-f", n=1)
    b2v2_raw["label"] = b2v2_raw["label"].map(
        lambda x: "No adhesion" if x == "No adhesion" else "adhesion")
    b2v2_raw = parse_frame(b2v2_raw, "filename", "_")

    # ── Per-frame counts and common frames ───────────────────────────────────
    b1_fc   = b1_raw.groupby(["cond", "frame"]).size().rename("n_b1")
    b2v2_fc = b2v2_raw.groupby(["cond", "frame"]).size().rename("n_b2v2")
    fc = pd.concat([b1_fc, b2v2_fc], axis=1).dropna().astype(int)
    fc["n_matched"] = fc[["n_b1", "n_b2v2"]].min(axis=1)

    print(f"B1 total:   {len(b1_raw):4d} patches  {b1_raw.groupby(['cond','frame']).ngroups} frames")
    print(f"B2v2 total: {len(b2v2_raw):4d} patches  {b2v2_raw.groupby(['cond','frame']).ngroups} frames")
    print(f"Common frames: {len(fc)}")
    print(f"Matched patches per annotator: {fc['n_matched'].sum()}")
    print()
    print(fc.to_string())
    print()

    # ── Sample matched patches ────────────────────────────────────────────────
    b1_rows, b2v2_rows = [], []

    for (cond, frame), row in fc.iterrows():
        n = int(row["n_matched"])

        b1_pool   = b1_raw[(b1_raw["cond"] == cond) & (b1_raw["frame"] == frame)]
        b2v2_pool = b2v2_raw[(b2v2_raw["cond"] == cond) & (b2v2_raw["frame"] == frame)]

        b1_sel   = b1_pool.sample(n=n, random_state=int(rng.integers(1 << 31)))
        b2v2_sel = b2v2_pool.sample(n=n, random_state=int(rng.integers(1 << 31)))

        b1_rows.append(b1_sel[["unique_ID", "label"]])
        b2v2_rows.append(b2v2_sel[["unique_ID", "label"]])

    b1_out   = pd.concat(b1_rows).reset_index(drop=True)
    b2v2_out = pd.concat(b2v2_rows).reset_index(drop=True)

    # ── Summary ──────────────────────────────────────────────────────────────
    for name, df in [("B1 matched v2", b1_out), ("B2v2 matched", b2v2_out)]:
        vc = df["label"].value_counts()
        print(f"{name}: n={len(df)}  adhesion={vc.get('adhesion',0)}  "
              f"No adhesion={vc.get('No adhesion',0)}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_b1   = LAB / "vinc_b1_matched_ds1_v2.csv"
    out_b2v2 = LAB / "vinc_b2v2_matched_ds1.csv"

    if not args.dry_run:
        b1_out.to_csv(out_b1,   index=False)
        b2v2_out.to_csv(out_b2v2, index=False)
        print(f"\nSaved: {out_b1}")
        print(f"Saved: {out_b2v2}")
    else:
        print("\n[dry-run] files not written")


if __name__ == "__main__":
    main()

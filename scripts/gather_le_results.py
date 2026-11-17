#!/usr/bin/env python3
"""
gather_le_results.py

Consolidate all LE benchmark eval CSVs into clearly-named summary files
in results/le_summary/.

For each method × dataset × batch × classifier, outputs one CSV with columns:
  fold, budget, repeat
  n_train           : training budget (labelled patches used)
  n_test            : test set size
  bal_acc_adnoad    : 2-class (adhesion vs no-adhesion) balanced accuracy
  --- B12 DS1 LGBM only ---
  bal_acc_ctrlvy_all   : ctrl vs ycomp balanced acc, all patches
  bal_acc_ctrlvy_ad    : ctrl vs ycomp balanced acc, adhesion patches only
  bal_acc_ctrlvy_noad  : ctrl vs ycomp balanced acc, no-adhesion patches only
  n_train_fa4       : B2-labelled patches in training set (may be < budget)
  n_test_fa4        : B2-labelled patches in test set
  bal_acc_5class    : 5-class FA-type balanced accuracy
  macro_f1_5class   : 5-class macro F1
  f1_NoAd           : F1 for "No adhesion"
  f1_NA             : F1 for "Nascent Adhesion"
  f1_FC             : F1 for "focal complex"
  f1_FA             : F1 for "focal adhesion"
  f1_FiBA           : F1 for "fibrillar adhesion"
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
EVAL_DIR = DATA / "ae_results" / "features" / "eval_results"
OUT_DIR  = Path(__file__).resolve().parents[1] / "results" / "le_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load(filename: str) -> pd.DataFrame | None:
    path = EVAL_DIR / filename
    if not path.exists():
        print(f"  [skip] not found: {filename}")
        return None
    return pd.read_csv(path)


def _pivot_ctrlvy(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot subset (all/ad/noad) → separate columns, keyed by fold/budget/repeat."""
    out = {}
    for subset, col in [("all", "bal_acc_ctrlvy_all"),
                        ("ad",  "bal_acc_ctrlvy_ad"),
                        ("noad","bal_acc_ctrlvy_noad")]:
        sub = df[df["subset"] == subset][["fold", "budget", "repeat", "bal_acc"]].copy()
        sub = sub.rename(columns={"bal_acc": col})
        out[subset] = sub
    merged = out["all"]
    for sub in ("ad", "noad"):
        merged = merged.merge(out[sub], on=["fold", "budget", "repeat"], how="left")
    return merged


def _rename_fa4(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "bal_acc":               "bal_acc_5class",
        "macro_f1":              "macro_f1_5class",
        "f1_No adhesion":        "f1_NoAd",
        "f1_Nascent Adhesion":   "f1_NA",
        "f1_focal complex":      "f1_FC",
        "f1_focal adhesion":     "f1_FA",
        "f1_fibrillar adhesion": "f1_FiBA",
        "n_train":               "n_train_fa4",
        "n_test":                "n_test_fa4",
    })
    return df


def build(
    out_name: str,
    main_file: str,
    ctrlvy_file: str | None = None,
    fa4_file: str | None = None,
):
    main = _load(main_file)
    if main is None:
        return

    # rename main bal_acc → bal_acc_adnoad
    main = main.rename(columns={"bal_acc": "bal_acc_adnoad"})
    # handle combined files with n_b2_train / n_b1_train
    if "n_b2_train" in main.columns and "n_train" not in main.columns:
        main = main.rename(columns={"n_b2_train": "n_train_b2", "n_b1_train": "n_train_b1"})
        keep_cols = ["fold", "budget", "repeat", "n_train_b2", "n_train_b1", "n_test", "bal_acc_adnoad"]
    else:
        keep_cols = ["fold", "budget", "repeat", "n_train", "n_test", "bal_acc_adnoad"]
    # normalise budget to str for consistent merging
    main["budget"] = main["budget"].astype(str)
    # sort
    main = main.sort_values(["fold", "budget", "repeat"]).reset_index(drop=True)
    result = main[keep_cols].copy()

    if ctrlvy_file:
        ctrlvy_raw = _load(ctrlvy_file)
        if ctrlvy_raw is not None:
            # budget may contain 'all' for full-set runs — keep as string for merge
            ctrlvy_raw["budget"] = ctrlvy_raw["budget"].astype(str)
            result["budget"] = result["budget"].astype(str)
            ctrlvy = _pivot_ctrlvy(ctrlvy_raw)
            result = result.merge(ctrlvy, on=["fold", "budget", "repeat"], how="left")

    if fa4_file:
        fa4_raw = _load(fa4_file)
        if fa4_raw is not None:
            fa4_raw["budget"] = fa4_raw["budget"].astype(str)
            result["budget"] = result["budget"].astype(str)
            fa4 = _rename_fa4(fa4_raw)[
                ["fold", "budget", "repeat",
                 "n_train_fa4", "n_test_fa4",
                 "bal_acc_5class", "macro_f1_5class",
                 "f1_NoAd", "f1_NA", "f1_FC", "f1_FA", "f1_FiBA"]
            ]
            result = result.merge(fa4, on=["fold", "budget", "repeat"], how="left")

    out_path = OUT_DIR / out_name
    result.to_csv(out_path, index=False)
    print(f"  saved {out_path.name}  ({len(result)} rows, {len(result.columns)} cols)")


# ---------------------------------------------------------------------------
# B12 DS1  —  main LE benchmark, full metrics
# ---------------------------------------------------------------------------
print("\n=== B12 DS1 ===")
build(
    "b12_ds1__supcon_lat12__lgbm.csv",
    main_file   = "supcon_le_b12_ds1_lat12p8_ds1.csv",
    ctrlvy_file = "le_b12_ctrlvy_supcon_ds1.csv",
    fa4_file    = "le_b12_fa4_perclass_supcon_ds1.csv",
)
build(
    "b12_ds1__supcon_lat12__logreg.csv",
    main_file   = "supcon_le_b12_ds1_lat12p8_logreg_ds1.csv",
    ctrlvy_file = None,   # only LGBM ctrlvy/fa4 were generated
    fa4_file    = None,
)
build(
    "b12_ds1__supcon_lat64__lgbm.csv",
    main_file   = "supcon_le_b12_ds1_lat64p32_ds1.csv",
    ctrlvy_file = None,
    fa4_file    = None,
)
build(
    "b12_ds1__supcon_lat64__logreg.csv",
    main_file   = "supcon_le_b12_ds1_lat64p32_logreg_ds1.csv",
    ctrlvy_file = None,
    fa4_file    = None,
)
build(
    "b12_ds1__supcon_supcon__logreg.csv",
    main_file   = "supcon_le_b12_supcon_logreg_ds1.csv",
    ctrlvy_file = None,
    fa4_file    = None,
)
build(
    "b12_ds1__cp__lgbm.csv",
    main_file   = "cp_b12_ds1.csv",
    ctrlvy_file = "le_b12_ctrlvy_cp_ds1.csv",
    fa4_file    = "le_b12_fa4_perclass_cp_ds1.csv",
)
build(
    "b12_ds1__cp__logreg.csv",
    main_file   = "cp_b12_logreg_ds1.csv",
    ctrlvy_file = None,
    fa4_file    = None,
)
build(
    "b12_ds1__ilastik__lgbm.csv",
    main_file   = "ilastik_b12_ds1.csv",
    ctrlvy_file = "le_b12_ctrlvy_il_ds1.csv",
    fa4_file    = "le_b12_fa4_perclass_il_ds1.csv",
)
build(
    "b12_ds1__ilastik__logreg.csv",
    main_file   = "ilastik_b12_logreg_ds1.csv",
    ctrlvy_file = None,
    fa4_file    = None,
)

# ---------------------------------------------------------------------------
# B12 DS2
# ---------------------------------------------------------------------------
print("\n=== B12 DS2 ===")
build(
    "b12_ds2__supcon_lat12__lgbm.csv",
    main_file = "supcon_le_b12_ds2_lat12p8_ds2.csv",
)
build(
    "b12_ds2__supcon_lat64__lgbm.csv",
    main_file = "supcon_le_b12_ds2_lat64p32_ds2.csv",
)
build(
    "b12_ds2__supcon_lat64__logreg.csv",
    main_file = "supcon_le_b12_ds2_lat64p32_logreg_ds2.csv",
)
build(
    "b12_ds2__supcon_supcon__logreg.csv",
    main_file = "supcon_le_b12_supcon_logreg_ds2.csv",
)
build(
    "b12_ds2__cp__lgbm.csv",
    main_file = "cp_b12_ds2.csv",
)
build(
    "b12_ds2__cp__logreg.csv",
    main_file = "cp_b12_logreg_ds2.csv",
)
build(
    "b12_ds2__ilastik__lgbm.csv",
    main_file = "ilastik_b12_ds2.csv",
)
build(
    "b12_ds2__ilastik__logreg.csv",
    main_file = "ilastik_b12_logreg_ds2.csv",
)

# ---------------------------------------------------------------------------
# B2 DS1
# ---------------------------------------------------------------------------
print("\n=== B2 DS1 ===")
build(
    "b2_ds1__supcon_lat12__lgbm.csv",
    main_file = "supcon_le_b2_lat12p8_ds1.csv",
)
build(
    "b2_ds1__supcon_lat12__logreg.csv",
    main_file = "supcon_le_b2_lat12p8_logreg_ds1.csv",
)
build(
    "b2_ds1__supcon_supcon__lgbm.csv",
    main_file = "supcon_le_b2_supcon_ds1.csv",
)
build(
    "b2_ds1__supcon_supcon__logreg.csv",
    main_file = "supcon_le_b2_supcon_logreg_ds1.csv",
)
build(
    "b2_ds1__cp__lgbm.csv",
    main_file = "cp_ds1.csv",
)
build(
    "b2_ds1__cp__logreg.csv",
    main_file = "cp_b2_logreg_ds1.csv",
)
build(
    "b2_ds1__ilastik__lgbm.csv",
    main_file = "ilastik_ds1.csv",
)
build(
    "b2_ds1__ilastik__logreg.csv",
    main_file = "ilastik_b2_logreg_ds1.csv",
)

# ---------------------------------------------------------------------------
# B2 DS2
# ---------------------------------------------------------------------------
print("\n=== B2 DS2 ===")
build(
    "b2_ds2__cp__logreg.csv",
    main_file = "cp_b2_logreg_ds2.csv",
)
build(
    "b2_ds2__ilastik__lgbm.csv",
    main_file = "ilastik_ds2.csv",
)
build(
    "b2_ds2__ilastik__logreg.csv",
    main_file = "ilastik_b2_logreg_ds2.csv",
)

# ---------------------------------------------------------------------------
# B2 DS3
# ---------------------------------------------------------------------------
print("\n=== B2 DS3 ===")
build(
    "b2_ds3__cp__lgbm.csv",
    main_file = "cp_ds3.csv",
)
build(
    "b2_ds3__ilastik__lgbm.csv",
    main_file = "ilastik_ds3.csv",
)

# ---------------------------------------------------------------------------
# B1B2  —  label quality comparison (B1 vs B2 annotator)
# ---------------------------------------------------------------------------
print("\n=== B1B2 DS1 (label quality) ===")
build(
    "b1b2_ds1__b1__cp__lgbm.csv",
    main_file = "cp_b1b2_b1_ds1.csv",
)
build(
    "b1b2_ds1__b1__cp__logreg.csv",
    main_file = "cp_b1b2_b1_logreg_ds1.csv",
)
build(
    "b1b2_ds1__b2__cp__lgbm.csv",
    main_file = "cp_b1b2_b2_ds1.csv",
)
build(
    "b1b2_ds1__b2__cp__logreg.csv",
    main_file = "cp_b1b2_b2_logreg_ds1.csv",
)
build(
    "b1b2_ds1__b1__ilastik__lgbm.csv",
    main_file = "il_b1b2_b1_ds1.csv",
)
build(
    "b1b2_ds1__b1__ilastik__logreg.csv",
    main_file = "il_b1b2_b1_logreg_ds1.csv",
)
build(
    "b1b2_ds1__b2__ilastik__lgbm.csv",
    main_file = "il_b1b2_b2_ds1.csv",
)
build(
    "b1b2_ds1__b2__ilastik__logreg.csv",
    main_file = "il_b1b2_b2_logreg_ds1.csv",
)

# CNN — may be partial
build(
    "b1b2_ds1__b1__cnn__direct.csv",
    main_file = "cnn_b1b2_b1_ds1.csv",
)

# ---------------------------------------------------------------------------
# Vinculin ctrl-only
# ---------------------------------------------------------------------------
print("\n=== Vinc ctrl-only ===")
build(
    "b2_vinc_ctrl__cp__lgbm.csv",
    main_file = "cp_b2_vinc_ctrl_vc.csv",
)
build(
    "b2_vinc_ctrl__cp__logreg.csv",
    main_file = "cp_b2_vinc_ctrl_logreg_vc.csv",
)
build(
    "b2_vinc_ctrl__supcon_lat12__lgbm.csv",
    main_file = "supcon_le_b2_vinc_ctrl_vc.csv",
)
build(
    "b2_vinc_ctrl__supcon_lat12__logreg.csv",
    main_file = "supcon_le_b2_vinc_ctrl_logreg_vc.csv",
)
build(
    "b2_vinc_ctrl__ilastik__lgbm.csv",
    main_file = "ilastik_b2_vinc_ctrl_vc.csv",
)
build(
    "b2_vinc_ctrl__ilastik__logreg.csv",
    main_file = "ilastik_b2_vinc_ctrl_logreg_vc.csv",
)

# ---------------------------------------------------------------------------
# B12 DS1 combined (B2 labels + B1 unlabelled as extra training)
# ---------------------------------------------------------------------------
print("\n=== B12 DS1 combined (B2+B1 training) ===")
build(
    "b12_combined_ds1__cp__lgbm.csv",
    main_file = "b12_combined_cp_ds1.csv",
)
build(
    "b12_combined_ds1__ilastik__lgbm.csv",
    main_file = "b12_combined_ilastik_ds1.csv",
)
build(
    "b12_combined_ds1__supcon_lat12__lgbm.csv",
    main_file = "b12_combined_supcon_ds1.csv",
)

print("\nDone. Output in:", OUT_DIR)

#!/usr/bin/env python3
"""
run_stage2_rebuild_ae.py

Step 3 of the stage-2 rebuild: SupCon AE (lat=32, proj=16, batch 256 = 2x the
usual 128) on DS1 vinc ctrl+ycomp.

  --mode stage2 : pool = stage2_pool.csv (predicted-adhesion U FA4-labelled),
                  supervised with the 4 FA classes.
  --mode whole  : pool = all patches, supervised with No adhesion + 4 FA classes.
  --mode stage2_b1 / stage2_b2 : same pool as stage2, but supervised with only
                  the B1 (Margaret) or only the B2 (Annabel) FA4 labels.
  --mode {stage2,stage2_b1,stage2_b2}_conf90 : same, but pool restricted to
                  p(adhesion) >= 0.9 (stage2_pool_conf0.9.csv).

Inputs come from rebuild_stage2_prep.py (labelling/stage2_rebuild/).
Output: contrastive_run/stage2_rebuild_{mode}/  (model_*.pt, latents.csv, ...)

Usage:
  python scripts/run_stage2_rebuild_ae.py --mode stage2
  python scripts/run_stage2_rebuild_ae.py --mode whole
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
PREP     = DATA / "labelling" / "stage2_rebuild"
RUN_DIR  = DATA / "ae_results" / "contrastive_run"
PATCHES  = DATA / "ae_results" / "patches" / "cio" / "vinc"
PATCH_DIR = {"control": PATCHES / "control" / "tiff_patches32_mr10",
             "ycomp":   PATCHES / "ycomp"   / "tiff_patches32_mr10"}

FA4 = ["Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
LABEL_ORDER = {"stage2": FA4, "whole": ["No adhesion"] + FA4, "stage2_b1": FA4, "stage2_b2": FA4}
SUFFIX = "_conf90"          # e.g. stage2_b1_conf90 -> stage2_b1 labels, p>=0.9 pool
POOL_FILE = {"": "stage2_pool.csv", SUFFIX: "stage2_pool_conf0.9.csv"}
for _m in ("stage2", "stage2_b1", "stage2_b2"):
    LABEL_ORDER[_m + SUFFIX] = FA4
B1_FILE = DATA / "labelling" / "labels_vinc_20260521.csv"
B2_FILE = DATA / "labelling" / "vinc_combined_label_Annabel_20260911.csv"


def load_labels(mode: str) -> pd.DataFrame:
    """filename (underscore), condition, label for the given mode."""
    mode = mode.removesuffix(SUFFIX)
    if mode == "stage2_b1":
        b1 = pd.read_csv(B1_FILE)
        out = pd.DataFrame({"filename": b1["unique_ID"].str.replace("-f", "_f", n=1, regex=False),
                            "label": b1["classification"]})
    elif mode == "stage2_b2":
        out = pd.read_csv(B2_FILE)[["filename", "label"]]
    else:
        out = pd.read_csv(PREP / "combined_labels.csv")[["filename", "label"]]
    out["condition"] = out["filename"].map(lambda f: "control" if f.startswith("control") else "ycomp")
    return out


def _h(fn: str) -> str:
    """underscore patch filename -> hyphen unique_ID (control_f.. -> control-f..)"""
    return fn.replace("_f", "-f", 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(LABEL_ORDER), required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline

    order = LABEL_ORDER[args.mode]
    labels = load_labels(args.mode)
    labels = labels[labels["label"].isin(order)]
    pool = None
    if args.mode != "whole":
        pool = pd.read_csv(PREP / POOL_FILE[SUFFIX if args.mode.endswith(SUFFIX) else ""])
    result_dir = RUN_DIR / f"stage2_rebuild_{args.mode}"
    tmp = result_dir / "annotations"
    tmp.mkdir(parents=True, exist_ok=True)

    labels[["filename", "label"]].to_csv(tmp / "labels_used.csv", index=False)
    patch_dirs = []
    for cond_id, cond in enumerate(["control", "ycomp"]):
        ann = labels[labels["condition"] == cond].copy()
        ann["unique_ID"] = ann["filename"].map(_h)
        ann_csv = tmp / f"{cond}_labels.csv"
        ann[["unique_ID", "label"]].to_csv(ann_csv, index=False)
        entry = {
            "path": str(PATCH_DIR[cond]),
            "condition": cond_id,
            "condition_name": f"vinc_{cond}",
            "annotation_file": str(ann_csv),
            "label_col": "label",
            "filename_col": "unique_ID",
            "label_order": order,
        }
        if pool is not None:
            entry["patch_include"] = pool.loc[pool["condition"] == cond, "filename"].tolist()
        n_in = len(entry.get("patch_include", [])) or len(list(PATCH_DIR[cond].glob("*.tif")))
        print(f"[{args.mode}] {cond}: {n_in} patches, {len(ann)} labelled")
        patch_dirs.append(entry)

    cfg = AEConfig(
        result_dir=str(result_dir),
        patch_dirs=patch_dirs,
        model_type="supcon",
        latent_dim=32, proj_dim=16, input_ps=32, no_ch=1,
        BN_flag=False, dropout_flag=False, output_sigmoid=False,
        recon_loss_type="nl1",
        noise_prob=0.0, temperature=0.5,
        lambda_recon=1.0, lambda_contrast=0.5,
        intensity_scale_range=(0.8, 1.2),
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
        num_workers=0, val_split=0.2, group_split=True,
        weight_decay=1e-4, warmup_epochs=0, lr_scheduler="none",
        early_stopping_patience=0, min_epochs_for_best=0,
        save_recon=False, device="auto",
    )
    run_ae_pipeline(cfg)
    print(f"Done: {result_dir}")


if __name__ == "__main__":
    main()

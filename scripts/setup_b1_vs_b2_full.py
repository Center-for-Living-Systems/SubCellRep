#!/usr/bin/env python3
"""
setup_b1_vs_b2_full.py

Full-label B1 vs full-label B2 comparison on DS1 (vinc ctrl + ycomp),
for two supervised models:
  - supcon  : SupCon-AE (recon + contrastive + supervised contrastive)
  - semisup : SemiSup AE (recon + classification head)

Design
------
- B1 = labels_vinc_20260521.csv (Margaret; Uncertain dropped), binarized
- B2 = vinc_combined_label_Annabel_20260816.csv (Annabel), binarized
- ALL labels of each batch are used (no budget subsampling).
- Image-grouped 80/20 train/val split (seed 42, set by the pipeline). The split
  depends only on the patch dirs, not on the labels, so B1 and B2 runs share
  identical train/val frames.
- Only the final model (model_final.pt) + latents.csv are kept; model_best.pt
  and per-epoch PNGs are removed by sbatch_b1_vs_b2_full.sh.

Outputs
-------
  labelling/b1_vs_b2_full/{b1,b2}_full.csv        (unique_ID, label)
  config/b1_vs_b2_full/b{1,2}full_{supcon,semisup}.yaml
  config/b1_vs_b2_full/job_list.txt
  ae_results/contrastive_run/b1_vs_b2_full/<name>/

Usage
-----
  python scripts/setup_b1_vs_b2_full.py [--dry-run]
  sbatch --array=0-3 scripts/sbatch_b1_vs_b2_full.sh
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT  = Path(__file__).resolve().parents[1]
DATA_ROOT  = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LABEL_DIR  = DATA_ROOT / "labelling"
OUT_ANN    = LABEL_DIR / "b1_vs_b2_full"
CONFIG_DIR = REPO_ROOT / "config" / "b1_vs_b2_full"
RUN_TAG    = "b1_vs_b2_full"

B1_FILE = LABEL_DIR / "labels_vinc_20260521.csv"
B2_FILE = LABEL_DIR / "vinc_combined_label_Annabel_20260816.csv"

MODELS = ["supcon", "semisup"]
BATCHES = ["b1", "b2"]

PATCH_DIRS = """\
    - path           : root_folder + "/ae_results/patches/cio/vinc/control/tiff_patches32_mr10"
      frame_dir      : root_folder + "/ae_results/source_frames/cio_mode_prt/vinc/control"
      condition      : 0
      condition_name : "control"
    - path           : root_folder + "/ae_results/patches/cio/vinc/ycomp/tiff_patches32_mr10"
      frame_dir      : root_folder + "/ae_results/source_frames/cio_mode_prt/vinc/ycomp"
      condition      : 1
      condition_name : "ycomp"
"""

COMMON_HEAD = """\
# =============================================================================
# {batch_upper} full labels — {model} AE  (DS1 vinc ctrl+ycomp, binary adhesion)
# image-grouped 80/20 train/val split; only model_final.pt kept
# =============================================================================
root_folder : "/net/projects/CLS/lding/data/fa_data_analysis"

data:
  patch_dirs:
{patch_dirs}
{enlcrop}output:
  result_dir : root_folder + "/ae_results/contrastive_run/{run_tag}/{name}"

"""

ENLCROP = """\
enlarged_crop:
  enabled       : true
  channel       : "pax"
  context_size  : 58
  max_shift_px  : 4
  max_angle_deg : 15.0
  pad_size      : 64
  input_divisor : 2.0

"""

SUPCON_BODY = """\
model:
  model_type      : "supcon"
  latent_dim      : 12
  input_ps        : 32
  no_ch           : 1
  BN_flag         : false
  dropout_flag    : false
  output_sigmoid  : false
  recon_loss_type : "nl1"

  proj_dim              : 8
  noise_prob            : 0.0
  temperature           : 0.5
  lambda_recon          : 1.0
  lambda_contrast       : 0.5
  lambda_supcon         : 5.0
  intensity_scale_range : [0.8, 1.2]

annotation:
  annotation_file : root_folder + "/labelling/b1_vs_b2_full/{batch}_full.csv"
  label_col       : "label"
  filename_col    : "unique_ID"
  label_order:
    - "No adhesion"
    - "adhesion"

training:
  epochs                  : 500
  lr                      : 0.001
  batch_size              : 128
  num_workers             : 6
  val_split               : 0.2
  group_split             : true
  loss_norm_flag          : false
  weight_decay            : 0.0001
  warmup_epochs           : 0
  lr_scheduler            : "none"
  early_stopping_patience : 0
  min_epochs_for_best     : 0
"""

SEMISUP_BODY = """\
model:
  model_type   : "semisup"
  latent_dim   : 12
  input_ps     : 32
  no_ch        : 1
  BN_flag      : false
  dropout_flag : false

  num_classes  : 2
  lambda_recon : 1.0
  lambda_cls   : 1.0

annotation:
  annotation_file : root_folder + "/labelling/b1_vs_b2_full/{batch}_full.csv"
  label_col       : "label"
  filename_col    : "unique_ID"
  label_order:
    - "No adhesion"
    - "adhesion"

training:
  epochs                  : 500
  lr                      : 0.001
  batch_size              : 128
  val_split               : 0.2
  group_split             : true
  loss_norm_flag          : false
  weight_decay            : 0.0001
  early_stopping_patience : 0
  min_epochs_for_best     : 200
  warmup_epochs           : 200
  lr_scheduler            : "cosine"
  lr_min                  : 1e-5
"""

TAIL = """
reconstruction:
  save_recon : false

misc:
  device    : "auto"
  log_level : "INFO"
"""


def _binarize(label: str) -> str:
    return "No adhesion" if label == "No adhesion" else "adhesion"


def load_labels() -> dict[str, pd.DataFrame]:
    b1 = pd.read_csv(B1_FILE)
    b1 = b1[b1["classification"] != "Uncertain"].copy()
    b1["label"] = b1["classification"].apply(_binarize)
    b1 = b1[["unique_ID", "label"]]

    b2 = pd.read_csv(B2_FILE)
    b2["unique_ID"] = b2["filename"].str.replace("_f", "-f", n=1, regex=False)
    b2["label"] = b2["label"].apply(_binarize)
    b2 = b2[["unique_ID", "label"]]
    return {"b1": b1, "b2": b2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    dry = ap.parse_args().dry_run

    labels = load_labels()
    for b, df in labels.items():
        print(f"{b.upper()}: n={len(df)}  adh={(df['label']=='adhesion').sum()}  "
              f"noad={(df['label']=='No adhesion').sum()}  dup_ids={df['unique_ID'].duplicated().sum()}")

    if not dry:
        OUT_ANN.mkdir(parents=True, exist_ok=True)
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        for b, df in labels.items():
            df.to_csv(OUT_ANN / f"{b}_full.csv", index=False)

    job_list = []
    for batch in BATCHES:
        for model in MODELS:
            name = f"{batch}full_{model}"
            head = COMMON_HEAD.format(
                batch_upper=batch.upper(), model=model, patch_dirs=PATCH_DIRS.rstrip("\n"),
                enlcrop=ENLCROP if model == "supcon" else "",
                run_tag=RUN_TAG, name=name)
            body = (SUPCON_BODY if model == "supcon" else SEMISUP_BODY).format(batch=batch)
            cfg = CONFIG_DIR / f"{name}.yaml"
            if not dry:
                cfg.write_text(head + body + TAIL)
            job_list.append(str(cfg.relative_to(REPO_ROOT)))

    if not dry:
        (CONFIG_DIR / "job_list.txt").write_text("\n".join(job_list) + "\n")
    print(f"{len(job_list)} jobs{' (dry-run)' if dry else ''}: " + ", ".join(job_list))


if __name__ == "__main__":
    main()

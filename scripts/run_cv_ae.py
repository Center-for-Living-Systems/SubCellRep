#!/usr/bin/env python3
"""
run_cv_ae.py  --arm {b1,b2,b12} --fold K

One fold of the frame-grouped CV study: SupCon-AE (lat 32, proj 16, batch 256) on the stage-2 pool
(p>=0.5, 12,571 patches), supervised with the FA4 labels of ONE annotator arm (B1 / B2 / B1+B2),
with all labels in the held-out fold's frames hidden from the encoder.
Output: contrastive_run/cv_ae/{arm}_f{K}/latents.csv
"""
import argparse, sys, warnings
from pathlib import Path
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_common import *

PATCHES = DATA / "ae_results" / "patches" / "cio" / "vinc"
ap = argparse.ArgumentParser()
ap.add_argument("--arm", choices=["b1", "b2", "b12"], required=True)
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline
b1, b2 = load_fa4()
folds = make_folds(a.seed)
test_frames = {g for g, f in folds.items() if f == a.fold}
t1, t2 = b1[~b1.g.isin(test_frames)], b2[~b2.g.isin(test_frames)]
lab = {"b1": t1, "b2": t2, "b12": pd.concat([t2, t1[~t1.fn.isin(set(t2.fn))]])}[a.arm]
pool = pd.read_csv(PREP / "stage2_pool.csv")
out = RUN / "cv_ae" / f"{a.arm}_f{a.fold}"; (out / "annotations").mkdir(parents=True, exist_ok=True)
print(f"arm={a.arm} fold={a.fold}: test frames {len(test_frames)}, training labels {len(lab)}, pool {len(pool)}", flush=True)

dirs = []
for cid, cond in enumerate(["control", "ycomp"]):
    ann = lab[lab.fn.str.startswith(cond)].copy()
    ann["unique_ID"] = ann.fn.str.replace("_f", "-f", n=1, regex=False)
    csv = out / "annotations" / f"{cond}.csv"; ann[["unique_ID", "label"]].to_csv(csv, index=False)
    dirs.append({"path": str(PATCHES / cond / "tiff_patches32_mr10"), "condition": cid, "condition_name": f"vinc_{cond}",
                 "annotation_file": str(csv), "label_col": "label", "filename_col": "unique_ID", "label_order": FA4,
                 "patch_include": pool.loc[pool.condition == cond, "filename"].tolist()})
cfg = AEConfig(result_dir=str(out), patch_dirs=dirs, model_type="supcon", latent_dim=32, proj_dim=16, input_ps=32, no_ch=1,
               BN_flag=False, dropout_flag=False, output_sigmoid=False, recon_loss_type="nl1", noise_prob=0.0, temperature=0.5,
               lambda_recon=1.0, lambda_contrast=0.5, intensity_scale_range=(0.8, 1.2), epochs=a.epochs, lr=1e-3, batch_size=256,
               num_workers=0, val_split=0.2, group_split=True, weight_decay=1e-4, warmup_epochs=0, lr_scheduler="none",
               early_stopping_patience=0, min_epochs_for_best=0, save_recon=False, device="auto")
run_ae_pipeline(cfg); print("Done", out)

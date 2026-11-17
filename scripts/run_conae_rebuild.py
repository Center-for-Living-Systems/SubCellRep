#!/usr/bin/env python3
"""
run_conae_rebuild.py

Unsupervised control for the stage-2 rebuild: ConAE (model_type="contrastive",
NT-Xent on augmented views, NO labels) on all 27,637 DS1 vinc ctrl+ycomp patches.
Same architecture / hyper-parameters as the supervised runs (latent 32, proj 16,
batch 256, 300 epochs, lr 1e-3, 80/20 image-grouped split); only the labels are gone.

Output: contrastive_run/stage2_rebuild_conae/  (latents.csv, model_best.pt, ...)
Probe it on every condition with:
  python scripts/eval_stage2_rebuild_latents.py --feat conae
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA    = Path("/net/projects/CLS/lding/data/fa_data_analysis")
RUN_DIR = DATA / "ae_results" / "contrastive_run"
PATCHES = DATA / "ae_results" / "patches" / "cio" / "vinc"


def main():
    from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline
    result_dir = RUN_DIR / "stage2_rebuild_conae"
    cfg = AEConfig(
        result_dir=str(result_dir),
        patch_dirs=[
            {"path": str(PATCHES / "control" / "tiff_patches32_mr10"), "condition": 0, "condition_name": "vinc_control"},
            {"path": str(PATCHES / "ycomp" / "tiff_patches32_mr10"),   "condition": 1, "condition_name": "vinc_ycomp"},
        ],
        model_type="contrastive",
        latent_dim=32, proj_dim=16, input_ps=32, no_ch=1,
        BN_flag=False, dropout_flag=False, output_sigmoid=False, recon_loss_type="nl1",
        noise_prob=0.0, temperature=0.5, lambda_recon=1.0, lambda_contrast=0.5,
        intensity_scale_range=(0.8, 1.2),
        epochs=300, lr=1e-3, batch_size=256, num_workers=0,
        val_split=0.2, group_split=True, weight_decay=1e-4, warmup_epochs=0,
        lr_scheduler="none", early_stopping_patience=0, min_epochs_for_best=0,
        save_recon=False, device="auto",
    )
    print(f"ConAE (unsupervised) on all vinc ctrl+ycomp patches -> {result_dir}")
    run_ae_pipeline(cfg)
    print(f"Done: {result_dir}")


if __name__ == "__main__":
    main()

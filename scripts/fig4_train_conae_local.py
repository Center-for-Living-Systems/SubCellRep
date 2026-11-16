#!/usr/bin/env python3
"""
fig4_train_conae_local.py <modality> [--epochs N]

Train a plain ConAE (NT-Xent, no labels; same architecture/hyper-parameters as
run_conae_baseline.py: latent 12, proj 8, nl1, lambda_contrast 0.5) on the 32 px
patches of one modality (actin | zyxin | mt), on the local GPU.

1. writes the patches as float32 TIFFs named {mod}_f{frame:04d}x{x:04d}y{y:04d}ps32.tif
   (so the pipeline's group-aware split keeps patches of one image together);
   values clipped to [0, 1] like the FA patches.
2. trains via subcellae.pipeline.ae_pipeline.run_ae_pipeline
3. writes features_ae32.csv (name, z_0..z_11) in the same row order as patches32_index.csv
   to the modality's patch directory, then deletes the temporary TIFFs.
"""
import argparse
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

AE = Path("/home/lding/lding/fa_data_analysis/ae_results")
DIRS = {"actin": AE / "ds1_actin_patches", "zyxin": AE / "ds1_zyxin_patches",
        "mt": Path("/home/lding/lding/data/Ernest/mt_patches")}
WORK = AE / "fig4_conae"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mod", choices=DIRS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--prep-only", action="store_true")
    a = ap.parse_args()
    d = DIRS[a.mod]
    idx = pd.read_csv(d / "patches32_index.csv")
    P = np.load(d / "patches32.npy", mmap_mode="r")
    tdir = WORK / a.mod / "tiff_patches32"
    tdir.mkdir(parents=True, exist_ok=True)

    frame = idx["name"].str.rsplit("_y", n=1).str[0]
    fid = pd.factorize(frame)[0]
    names = [f"{a.mod}_f{f:04d}x{x:04d}y{y:04d}ps32.tif" for f, x, y in zip(fid, idx.x0, idx.y0)]
    assert len(set(names)) == len(names)
    for i, n in enumerate(names):
        if not (tdir / n).exists():
            tifffile.imwrite(tdir / n, np.clip(np.asarray(P[i], dtype=np.float32), 0, 1))
    pd.DataFrame({"idx": idx.idx, "tif": names}).to_csv(WORK / a.mod / "tif_map.csv", index=False)
    print(f"{a.mod}: {len(names)} tiffs in {tdir}", flush=True)
    if a.prep_only:
        return

    from subcellae.pipeline.ae_pipeline import AEConfig, run_ae_pipeline
    res = WORK / a.mod / "conae"
    cfg = AEConfig(
        result_dir=res,
        patch_dirs=[{"path": str(tdir), "condition": 0, "condition_name": a.mod}],
        model_type="contrastive", latent_dim=12, proj_dim=8, input_ps=32, no_ch=1,
        BN_flag=False, dropout_flag=False, output_sigmoid=False, recon_loss_type="nl1",
        noise_prob=0.0, temperature=0.5, lambda_recon=1.0, lambda_contrast=0.5,
        intensity_scale_range=(0.8, 1.2),
        epochs=a.epochs, lr=1e-3, batch_size=128, num_workers=0, val_split=a.val_split,
        group_split=True, weight_decay=1e-4, warmup_epochs=0, lr_scheduler="none",
        early_stopping_patience=0, min_epochs_for_best=0,
        save_recon=False, device="auto",
    )
    run_ae_pipeline(cfg)

    lat = pd.read_csv(res / "latents.csv")
    zc = [c for c in lat.columns if c.startswith("z_")]
    m = pd.read_csv(WORK / a.mod / "tif_map.csv").merge(lat[["filename"] + zc], left_on="tif", right_on="filename", how="left")
    assert m[zc[0]].notna().all(), "some patches missing from latents.csv"
    out = pd.DataFrame(m[zc].values, columns=zc)
    out.insert(0, "name", idx["name"])
    out.to_csv(d / "features_ae32.csv", index=False)
    print(f"saved {d / 'features_ae32.csv'} {out.shape}; train/val: {lat.split.value_counts().to_dict()}")
    shutil.rmtree(tdir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
extract_supcon_features.py

Average the 3 random-seed SupCon latent CSVs (v0/v1/v2) and save two clean
feature files for use in LE/IE eval:

  supcon_z.csv  — filename + z_0..z_31  (32-dim encoder latent, avg across seeds)
  supcon_p.csv  — filename + p_0..p_15  (16-dim projection, avg across seeds)

Source model: annabel_vinc_supcon2_stage2_b12_lat32p16
"""
from pathlib import Path
import pandas as pd
import numpy as np

AE   = Path("/net/spaces/clinic/CLS/lding/data/fa_data_analysis/ae_results")
RUNS = [AE / f"contrastive_run/annabel_vinc_supcon2_stage2_b12_lat32p16_v{v}/latents.csv"
        for v in [0, 1, 2]]
OUT  = Path("/net/spaces/clinic/CLS/lding/data/fa_data_analysis/ae_results/features/supcon")
OUT.mkdir(parents=True, exist_ok=True)

Z_COLS = [f"z_{i}" for i in range(32)]
P_COLS = [f"p_{i}" for i in range(16)]

dfs = [pd.read_csv(p).set_index("filename") for p in RUNS]
print(f"Loaded {len(dfs)} seeds, each shape: {dfs[0].shape}")

# Average z and p dims across seeds
z_avg = sum(df[Z_COLS] for df in dfs) / len(dfs)
p_avg = sum(df[P_COLS] for df in dfs) / len(dfs)

z_out = z_avg.reset_index()[["filename"] + Z_COLS]
p_out = p_avg.reset_index()[["filename"] + P_COLS]

z_out.to_csv(OUT / "ds1_z.csv", index=False)
p_out.to_csv(OUT / "ds1_p.csv", index=False)
print(f"Saved {OUT}/ds1_z.csv  shape={z_out.shape}")
print(f"Saved {OUT}/ds1_p.csv  shape={p_out.shape}")

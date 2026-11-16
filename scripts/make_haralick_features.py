#!/usr/bin/env python3
"""features_hc{S}.csv = Haralick-only columns (drop the 11 intensity_* ones) of features_cp{S}.csv.
Haralick is computed after per-patch min-max quantisation to 16 levels, so it ignores intensity scale/offset."""
import sys
from pathlib import Path
import pandas as pd
for d in sys.argv[1:]:
    for S in (32, 64):
        f = pd.read_csv(Path(d) / f"features_cp{S}.csv")
        keep = ["name"] + [c for c in f.columns[1:] if not c.startswith("intensity_")]
        f[keep].to_csv(Path(d) / f"features_hc{S}.csv", index=False); print(d, S, len(keep) - 1)

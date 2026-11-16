#!/usr/bin/env python3
"""Stack DS1 paxillin (ch1) 32px mr10 patches (control+ycomp) + B1 FA5 labels into
the same layout as extract_mt_patches.py output (patches32.npy / patches32_index.csv)."""
from pathlib import Path
import numpy as np, pandas as pd, tifffile

ROOT = Path("/home/lding/lding/dsicluster_CLS_rsync_folder/data/fa_data_analysis")
OUT = Path("/home/lding/lding/fa_data_analysis/ae_results/fig4_fa_patches")
OUT.mkdir(parents=True, exist_ok=True)
lab = pd.read_csv(ROOT / "labelling/labels_vinc_20260521.csv")
lab["name"] = lab.condition + "_" + lab.crop_img_filename
lab = lab.drop_duplicates("name", keep="last").set_index("name")
rows, arr = [], []
for cond, label in (("control", "Control"), ("ycomp", "Ycomp")):
    d = ROOT / f"ae_results/patches/cio/vinc/{cond}/tiff_patches32_mr10"
    for f in sorted(d.glob("*.tif")):
        arr.append(tifffile.imread(f).astype(np.float16))
        rows.append({"name": f.name, "file": label,
                     "fa_label": lab.annotator.get(f.name) and lab.classification[f.name] or ""})
df = pd.DataFrame(rows); df.insert(0, "idx", range(len(df)))
np.save(OUT / "patches32.npy", np.stack(arr)); df.to_csv(OUT / "patches32_index.csv", index=False)
print(len(df), (df.fa_label != "").sum()); print(df.fa_label.value_counts())

"""Shared helpers for the B1-vs-B2 cross-validated embedding study (frame-grouped folds)."""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd

DATA = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LAB  = DATA / "labelling"
PREP = LAB / "stage2_rebuild"
RUN  = DATA / "ae_results" / "contrastive_run"
FEAT = DATA / "ae_results" / "features"
B1_FILE = LAB / "labels_vinc_20260521.csv"
B2_FILE = LAB / "vinc_combined_label_Annabel_20260911.csv"
FA4 = ["Nascent Adhesion", "focal complex", "focal adhesion", "fibrillar adhesion"]
N_FOLDS = 5


def group_of(fn: str) -> str:
    return fn.split("x", 1)[0]                      # control_f0001


def load_fa4():
    """B1 / B2 FA4 label tables: columns fn, label, g."""
    b1 = pd.read_csv(B1_FILE)
    b1 = pd.DataFrame({"fn": b1["unique_ID"].str.replace("-f", "_f", n=1, regex=False), "label": b1["classification"]})
    b2 = pd.read_csv(B2_FILE).rename(columns={"filename": "fn"})[["fn", "label"]]
    out = []
    for d in (b1, b2):
        d = d[d["label"].isin(FA4)].drop_duplicates("fn").copy()
        d["g"] = d["fn"].map(group_of)
        out.append(d.reset_index(drop=True))
    return out


def all_label_frames():
    b1 = pd.read_csv(B1_FILE); b2 = pd.read_csv(B2_FILE)
    fns = set(b1["unique_ID"].str.replace("-f", "_f", n=1, regex=False)) | set(b2["filename"])
    return sorted({group_of(f) for f in fns})


def make_folds(seed: int) -> dict:
    """frame -> fold id (random, balanced counts)."""
    g = all_label_frames()
    rng = np.random.RandomState(seed); rng.shuffle(g)
    return {f: i % N_FOLDS for i, f in enumerate(g)}


def train_sources(b1, b2, test_frames, rng):
    """Training label tables (frames outside the test fold): B1, B2, B2 size-matched to B1, B1+B2 (B2 priority)."""
    t1, t2 = b1[~b1.g.isin(test_frames)], b2[~b2.g.isin(test_frames)]
    t2m = t2.sample(n=min(len(t1), len(t2)), random_state=int(rng.randint(1e9)))
    t12 = pd.concat([t2, t1[~t1.fn.isin(set(t2.fn))]], ignore_index=True)
    return {"B1": t1, "B2": t2, "B2m": t2m, "B1+B2": t12}

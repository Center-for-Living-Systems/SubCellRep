#!/usr/bin/env python3
"""
save_actin_kmeans_patches.py

KMeans k=10 on actin CP features (DS1, ch3, PS32) in two settings:
  1. full_highdim  — KMeans on all 50 standardised CP features
  2. umap2d        — UMAP reduces 50-D → 2-D, then KMeans on 2-D embedding

For each cluster in each setting, saves the 16 patches closest to the
cluster centroid as individual uint16 TIFF files.

Output:
  eval_results/actin_kmeans_patches/
    full_highdim/
      cluster_00/ cluster_00_ex01_<name>.tif  ...  (16 files × 10 clusters)
      cluster_01/ ...
      ...
    umap2d/
      cluster_00/ ...
      ...

Usage:
  python scripts/save_actin_kmeans_patches.py [--out_dir PATH]
                                               [--no_umap]
                                               [--n_examples 16]
"""
from __future__ import annotations
import argparse, io, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/net/projects/CLS/lding/data/fa_data_analysis")
REPO      = Path(__file__).resolve().parents[1]

FEAT_CSV  = DATA_ROOT / "ae_results/features/cp_scale/ds1_ch3_s32.csv"
AE_CSV    = DATA_ROOT / "ae_results/protein_sweep/conae_3ch_pza_v_nl1/latents.csv"
H5_CTRL   = DATA_ROOT / "ae_results/patches/cio_rb/vinc/vinc_control_label.h5"
H5_YCOMP  = DATA_ROOT / "ae_results/patches/cio_rb/vinc/vinc_ycomp_label.h5"
H5_FULL   = DATA_ROOT / "ae_results/patches/cio_rb/vinc/data.h5"

DEFAULT_OUT = REPO / "eval_results" / "actin_kmeans_patches"

# ── Constants ─────────────────────────────────────────────────────────────────
N_CLUSTERS   = 10
N_EXAMPLES   = 16
PS           = 32
PAD          = 64
RANDOM_STATE = 42

FEAT_COLS = [
    "intensity_mean", "intensity_std", "intensity_median",
    "intensity_min",  "intensity_max", "intensity_integrated",
    "intensity_mad",  "intensity_p10", "intensity_p25",
    "intensity_p75",  "intensity_p90",
    "angular_second_moment_d1", "contrast_d1", "correlation_d1",
    "variance_d1", "inverse_difference_moment_d1", "sum_average_d1",
    "sum_variance_d1", "sum_entropy_d1", "entropy_d1",
    "difference_variance_d1", "difference_entropy_d1",
    "info_meas1_d1", "info_meas2_d1",
    "angular_second_moment_d2", "contrast_d2", "correlation_d2",
    "variance_d2", "inverse_difference_moment_d2", "sum_average_d2",
    "sum_variance_d2", "sum_entropy_d2", "entropy_d2",
    "difference_variance_d2", "difference_entropy_d2",
    "info_meas1_d2", "info_meas2_d2",
    "angular_second_moment_d4", "contrast_d4", "correlation_d4",
    "variance_d4", "inverse_difference_moment_d4", "sum_average_d4",
    "sum_variance_d4", "sum_entropy_d4", "entropy_d4",
    "difference_variance_d4", "difference_entropy_d4",
    "info_meas1_d4", "info_meas2_d4",
]


# ── Feature loading ───────────────────────────────────────────────────────────

def load_features() -> tuple[pd.DataFrame, np.ndarray]:
    print("Loading CP features ...", flush=True)
    df = pd.read_csv(FEAT_CSV)
    X  = df[FEAT_COLS].values.astype(np.float32)
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    print(f"  {len(df)} patches × {len(FEAT_COLS)} features", flush=True)
    return df, X_scaled


def load_ae_latents() -> tuple[pd.DataFrame, np.ndarray]:
    """Load ConAE 3ch (pax+zyx+act+vinc) z latents for DS1 patches."""
    print("Loading AE latents (conae_3ch_pza_v_nl1) ...", flush=True)
    df = pd.read_csv(AE_CSV)
    z_cols = [c for c in df.columns if c.startswith("z_")]
    X  = df[z_cols].values.astype(np.float32)
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X)
    print(f"  {len(df)} patches × {len(z_cols)}-D latent", flush=True)
    return df, X_scaled


# ── Clustering ────────────────────────────────────────────────────────────────

def run_kmeans(X: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    print(f"KMeans k={N_CLUSTERS} on {label} ({X.shape[1]}-D) ...", flush=True)
    km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    print(f"  Cluster sizes: {np.bincount(labels).tolist()}", flush=True)
    return labels, km.cluster_centers_


def run_dim_reduction(X_scaled: np.ndarray, method: str) -> np.ndarray:
    if method == "pca":
        from sklearn.decomposition import PCA
        print("Running PCA (50-D → 2-D) ...", flush=True)
        emb = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X_scaled)
        print("  PCA done.", flush=True)
        return emb
    elif method == "tsne":
        from sklearn.manifold import TSNE
        print("Running t-SNE (50-D → 2-D) ...", flush=True)
        emb = TSNE(n_components=2, perplexity=30, random_state=RANDOM_STATE,
                   n_jobs=4).fit_transform(X_scaled)
        print("  t-SNE done.", flush=True)
        return emb
    else:  # umap
        import os
        cache_dir = "/net/projects/CLS/lding/.numba_cache"
        os.makedirs(cache_dir, exist_ok=True)
        os.environ.setdefault("NUMBA_CACHE_DIR", cache_dir)
        import umap as umap_lib
        print("Running UMAP (50-D → 2-D) ...", flush=True)
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                 metric="euclidean", random_state=RANDOM_STATE,
                                 n_jobs=4, verbose=False)
        emb = reducer.fit_transform(X_scaled)
        print("  UMAP done.", flush=True)
        return emb


def find_examples(X: np.ndarray, km_labels: np.ndarray,
                  centres: np.ndarray, n: int) -> list[list[int]]:
    """Indices of n patches closest to each cluster centre."""
    examples = []
    for k in range(N_CLUSTERS):
        mask  = km_labels == k
        idxs  = np.where(mask)[0]
        dists = np.linalg.norm(X[idxs] - centres[k], axis=1)
        top   = idxs[np.argsort(dists)[:n]]
        examples.append(top.tolist())
    return examples


# ── Patch loading from H5 ─────────────────────────────────────────────────────

def build_patch_loader(df: pd.DataFrame):
    print("Loading actin images from H5 files ...", flush=True)

    def _read(path):
        with h5py.File(path) as f:
            imgs = f["images/act"][:]     # (N_frames, H, W) uint16
            meta = pd.read_csv(io.StringIO(f["meta/csv"][()].decode()))
        return imgs, meta

    ctrl_imgs,  ctrl_meta  = _read(H5_CTRL)
    ycomp_imgs, ycomp_meta = _read(H5_YCOMP)
    print(f"  ctrl: {ctrl_imgs.shape}  ycomp: {ycomp_imgs.shape}", flush=True)

    ctrl_lu  = {r.filename: (r.frame_idx, r.canvas_cx, r.canvas_cy)
                for r in ctrl_meta.itertuples()}
    ycomp_lu = {r.filename: (r.frame_idx, r.canvas_cx, r.canvas_cy)
                for r in ycomp_meta.itertuples()}

    ctrl_pad  = np.pad(ctrl_imgs,  ((0,0),(PAD,PAD),(PAD,PAD)), mode="edge")
    ycomp_pad = np.pad(ycomp_imgs, ((0,0),(PAD,PAD),(PAD,PAD)), mode="edge")
    half = PS // 2

    def _get(row_idx: int) -> tuple[np.ndarray | None, str]:
        fname = df["filename"].iloc[row_idx]
        cond  = (df["condition"].iloc[row_idx]
                 if "condition" in df.columns
                 else ("ycomp" if fname.startswith("ycomp") else "control"))
        if cond == "control" and fname in ctrl_lu:
            fi, cx, cy = ctrl_lu[fname];  imgs = ctrl_pad
        elif cond == "ycomp" and fname in ycomp_lu:
            fi, cx, cy = ycomp_lu[fname]; imgs = ycomp_pad
        else:
            return None, fname
        cy0 = cy + PAD - half
        cx0 = cx + PAD - half
        return imgs[fi, cy0:cy0+PS, cx0:cx0+PS].copy(), fname

    return _get


def build_patch_loader_full(df: pd.DataFrame):
    """Patch loader using the full data.h5 (all 27,637 patches, combined control+ycomp)."""
    print("Loading actin images from data.h5 ...", flush=True)
    with h5py.File(H5_FULL) as f:
        act_imgs = f["images/act"][:]      # (91, H, W) uint16
        meta = pd.read_csv(io.StringIO(f["meta/csv"][()].decode()))
    print(f"  data.h5 act: {act_imgs.shape}  meta: {len(meta)} rows", flush=True)

    lu = {r.filename: (r.frame_idx, r.canvas_cx, r.canvas_cy)
          for r in meta.itertuples()}

    act_pad = np.pad(act_imgs, ((0, 0), (PAD, PAD), (PAD, PAD)), mode="edge")
    half = PS // 2

    def _get(row_idx: int) -> tuple[np.ndarray | None, str]:
        fname = df["filename"].iloc[row_idx]
        if fname not in lu:
            return None, fname
        fi, cx, cy = lu[fname]
        cy0 = cy + PAD - half
        cx0 = cx + PAD - half
        return act_pad[fi, cy0:cy0+PS, cx0:cx0+PS].copy(), fname

    return _get


# ── Save one setting ──────────────────────────────────────────────────────────

def save_setting(df: pd.DataFrame, X: np.ndarray,
                 km_labels: np.ndarray, centres: np.ndarray,
                 get_patch, out_dir: Path, n_examples: int):
    examples = find_examples(X, km_labels, centres, n_examples)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = skipped = 0

    for k, idxs in enumerate(examples):
        cdir = out_dir / f"cluster_{k:02d}"
        cdir.mkdir(exist_ok=True)
        print(f"  Cluster {k:02d}  (n={int((km_labels==k).sum())}):", flush=True)
        for rank, row_idx in enumerate(idxs):
            patch, fname = get_patch(row_idx)
            if patch is None:
                print(f"    rank {rank+1}: SKIP — {fname} not in H5", flush=True)
                skipped += 1
                continue
            safe = Path(fname).stem.replace(" ", "_")
            stem = cdir / f"cluster_{k:02d}_ex{rank+1:02d}_{safe}"
            tifffile.imwrite(str(stem.with_suffix(".tif")), patch)
            # uint8 PNG for make_cluster_panel.py
            p_min, p_max = patch.min(), patch.max()
            if p_max > p_min:
                p8 = ((patch - p_min) * 255 / (p_max - p_min)).astype(np.uint8)
            else:
                p8 = np.zeros_like(patch, dtype=np.uint8)
            Image.fromarray(p8).save(str(stem.with_suffix(".png")))
            saved += 1
        print(f"    saved {min(len(idxs)-skipped, n_examples)} patches → {cdir}", flush=True)

    print(f"\n  Total: {saved} saved, {skipped} skipped", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",    default=str(DEFAULT_OUT))
    ap.add_argument("--no_umap",    action="store_true",
                    help="Skip 2-D reduction step (full_highdim only)")
    ap.add_argument("--no_highdim", action="store_true",
                    help="Skip full_highdim step (2-D only)")
    ap.add_argument("--ae",         action="store_true",
                    help="Also run AE latent KMeans → root-level cluster_XX/ (AE baseline)")
    ap.add_argument("--ae_only",    action="store_true",
                    help="Run AE latent KMeans only (skip CP steps)")
    ap.add_argument("--dim_red",    default="pca",
                    choices=["pca", "tsne", "umap"],
                    help="Dim reduction method for Case 2 (default: pca)")
    ap.add_argument("--n_examples", type=int, default=N_EXAMPLES)
    args = ap.parse_args()

    out_root = Path(args.out_dir)

    run_cp = not args.ae_only

    if run_cp:
        # Load CP features once
        df_cp, X_cp = load_features()
        get_patch_cp = build_patch_loader(df_cp)

        # ── Case 1: full high-dim ───────────────────────────────────────────
        if not args.no_highdim:
            print("\n" + "="*60, flush=True)
            print("Case 1: KMeans on full 50-D CP features", flush=True)
            print("="*60, flush=True)
            km_labels_hd, centres_hd = run_kmeans(X_cp, "full 50-D")
            save_setting(df_cp, X_cp, km_labels_hd, centres_hd,
                         get_patch_cp, out_root / "full_highdim", args.n_examples)

        if not args.no_umap:
            method = args.dim_red
            # ── Case 2: 2-D dim reduction then KMeans ──────────────────────
            print("\n" + "="*60, flush=True)
            print(f"Case 2: {method.upper()} 2-D then KMeans", flush=True)
            print("="*60, flush=True)
            emb = run_dim_reduction(X_cp, method)
            km_labels_2d, centres_2d = run_kmeans(emb, f"{method.upper()} 2-D")
            out_name = "umap2d" if method == "umap" else f"{method}2d"
            save_setting(df_cp, emb, km_labels_2d, centres_2d,
                         get_patch_cp, out_root / out_name, args.n_examples)

    if args.ae or args.ae_only:
        # ── Case 0: AE latent KMeans → root-level cluster_XX/ ──────────────
        print("\n" + "="*60, flush=True)
        print("Case 0: KMeans on AE latents (conae_3ch_pza_v_nl1)", flush=True)
        print("="*60, flush=True)
        df_ae, X_ae = load_ae_latents()
        get_patch_ae = build_patch_loader_full(df_ae)
        km_labels_ae, centres_ae = run_kmeans(X_ae, "AE 12-D latent")
        save_setting(df_ae, X_ae, km_labels_ae, centres_ae,
                     get_patch_ae, out_root, args.n_examples)

    # Summary
    print("\n" + "="*60, flush=True)
    print("Done. Output structure:", flush=True)
    for p in sorted(out_root.rglob("*.tif")):
        print(f"  {p.relative_to(out_root)}", flush=True)


if __name__ == "__main__":
    main()

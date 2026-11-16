# MT / actin / zyxin patch feature clustering — context (2026-11-16)

Unsupervised look at 32 px and 64 px patches: do hand-crafted features (CellProfiler-style, ilastik-style,
Haralick-only) give clusters that look coherent? Three patch sets, all ≥80 % inside the cell mask.

## Data

| Set | Source | Original dir | ≥80 % dir (`_mr80`) | Patches 32 / 64 px (kept) |
|---|---|---|---|---|
| MT | Ernest U2OS DMSO ND2→TIFF, ch0 segments, ch1 patches | `/home/lding/lding/data/Ernest/mt_patches` | `.../mt_patches_mr80` | 10,468 / 4,341 (of 13,241 / 6,647) |
| Actin | DS1 vinculin CZI (control + Ycomp), ch3 | `/home/lding/lding/fa_data_analysis/ae_results/ds1_actin_patches` | `.../ds1_actin_patches_mr80` | 21,223 / 4,734 (of 28,138 / 7,928) |
| Zyxin | same frames, ch2 | `.../ds1_zyxin_patches` | `.../ds1_zyxin_patches_mr80` | 21,223 / 4,734 |

Actin and zyxin use the same cell mask (segmented on ch3), hence identical counts and index rows.
The original extraction kept patches with `cell_cov ≥ 0.1`; the `_mr80` dirs are **filtered copies** of those
(no re-extraction), made by `subset_patches_by_cov.py` from the `cell_cov` column of `patches{S}_index.csv`.
Each `_mr80` dir holds patches, index, features (`cp`, `il`, `hc`), clusters, UMAPs; `display/` is a symlink
to the original overlays (boxes there still include the 10–80 % patches).

## Pipeline / scripts (scripts/)

| Script | Role |
|---|---|
| `extract_mt_patches.py` | Segment + extract grid patches (`--seg-ch`, `--mt-ch`, `--mask-ratio`, `--src` globs incl. CZI). Actin/zyxin originals = `--seg-ch 3` with `--mt-ch 3` / `2` (verified by matching the 375 / 102 patches of control-01). |
| `subset_patches_by_cov.py` | Filter an extraction dir to `cell_cov ≥ --min-cov`; carries CP + IL features and a recomputed `frame_summary.csv`. |
| `run_mt_features_clusters.py` | CP (50) / IL (80) / `hc` features → quantile→N(0,1) → KMeans (k=10 or `auto`) → 2-D UMAP. Use `--reuse-features` on subset dirs. |
| `make_haralick_features.py` | `features_hc{S}.csv` = 39 Haralick columns (drops the 11 `intensity_*`). |
| `make_pptx_mt_patches.py` | Deck: pipeline, counts, montage, per kind×size UMAP + 16 nearest-to-centre patches per cluster. `--kinds cp il hc`, `--min-cov`. |
| `make_pptx_umap_space_clusters.py` | `DIR KIND SIZE OUT [k|auto] [umap|feat]`: KMeans in UMAP or feature space, silhouette auto-k (4–20), slides with nearest-to-centre **and random** members. |
| `make_pptx_hdbscan_compare.py` | HDBSCAN on CP-PCA10, CP-UMAP and AE latents (actin 32 only so far). |

Env: `/home/lding/miniconda3/envs/subcellae-cuda/bin/python`. Decks are written to `results/` (gitignored).

## Decks produced (results/, not tracked)

- `mt_patch_clusters_mr80.pptx`, `ds1_{actin,zyxin}_patch_clusters_mr80.pptx` — CP + IL, k=10.
- `*_patch_clusters_haralick_mr80.pptx` — Haralick only, k=10.
- `ds1_actin_cp32_umapspace_clusters_mr80.pptx` (k=10) and `..._umapspace_auto_...` (k=6) — KMeans on UMAP coords.
- `ds1_actin_cp32_featspace_auto_clusters_mr80.pptx` — auto-k in 50-D feature space (picked k=4, the lower bound).
- `ds1_actin_cp32_hdbscan_compare_mr80.pptx`.
- `{mt,actin,zyxin}_hc{32,64}_featspace_auto_clusters_mr80.pptx` — Haralick, auto-k, nearest + random members.
  Auto k: MT 7/5, actin 15/4, zyxin 5/4 (32/64 px).

## Findings so far

1. **Weak cluster structure.** Actin CP 32 px, KMeans k=10 in feature space: silhouettes 0.10–0.27, first 3 PCs ≈73 % of variance.
   Auto-k silhouette decreases with k (picks the lower bound, 4). Patches look like a continuum.
2. **UMAP is a visualisation, not a clustering.** Clusters 0/8/9 touch in UMAP but their 50-D centroid distances (6.7–8.5) are no smaller than to others.
   KMeans in UMAP space (ARI 0.65 vs feature-space k=10) gives clusters whose *random* members still look unlike each other.
3. **HDBSCAN (actin CP 32):** 8 clusters (CP-PCA10, 2 % noise), 9 (UMAP), **0 on AE latents (100 % noise, one connected blob; also with smaller min-cluster-size/min-samples).**
4. **Invariance test (600 actin patches, features recomputed after transform; shift in units of the distance to the 10th nearest neighbour):**
   rotate/flip/transpose = 0 for CP and IL (exactly invariant); 4 px shift = 0.57× (CP) / 0.62× (IL), 25–40 % jump past the NN10 radius;
   brightness ×1.3 + offset = 2.7× (CP) / 5.0× (IL), ~100 % jump. So intensity scale/offset is the main non-invariance, translation is secondary.
5. Haralick-only should be intensity-scale/offset invariant (per-patch min–max to 16 grey levels) — **the invariance test has not been rerun on it, and visual quality of the Haralick clusters has not been judged yet.**

## Open / next

- Judge the random-member slides of the Haralick decks; decide if Haralick beats full CP.
- Rerun the shift/brightness invariance test on `hc`.
- Cheap check: per-patch z-score then CP features. Then contrastive embedding (augmentations: small shifts, intensity scale/offset, noise;
  rotation/flip optional) — existing setup: `scripts/run_contrastive_eval.py`, `contrastive_run/`, ConAE in `scripts/fig4_train_conae_local.py`.
- Alternative: show a continuum view (patches ordered along PC1/PC2 or UMAP axes) instead of discrete clusters.

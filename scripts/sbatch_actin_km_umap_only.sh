#!/usr/bin/env bash
#SBATCH --job-name=actin_km_umap
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm/actin_km_umap_%j.out

set -eo pipefail
exec 2>&1

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$PWD"
mkdir -p logs/slurm

echo "Node: $(hostname)"
echo "Start: $(date)"

# full_highdim already done; run PCA 2-D case (fast, no numba JIT)
$PYTHON scripts/save_actin_kmeans_patches.py --n_examples 16 --no_highdim --dim_red pca

echo "End: $(date)"
find eval_results/actin_kmeans_patches/pca2d/ -name "*.tif" | wc -l
echo "TIFFs written to pca2d/"

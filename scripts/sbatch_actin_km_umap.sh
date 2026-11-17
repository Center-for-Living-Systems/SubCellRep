#!/usr/bin/env bash
#SBATCH --job-name=actin_km_umap
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/actin_km_umap_%j.out

set -eo pipefail
exec 2>&1

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$PWD"
mkdir -p logs/slurm

echo "Node: $(hostname)"
echo "Start: $(date)"

# full pipeline including UMAP
$PYTHON scripts/save_actin_kmeans_patches.py --n_examples 16

echo "End: $(date)"
find eval_results/actin_kmeans_patches/ -name "*.tif" | wc -l
echo "total TIFFs"

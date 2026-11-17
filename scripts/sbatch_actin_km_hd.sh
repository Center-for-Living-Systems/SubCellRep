#!/usr/bin/env bash
#SBATCH --job-name=actin_km_hd
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:45:00
#SBATCH --output=logs/slurm/actin_km_hd_%j.out

set -eo pipefail
exec 2>&1

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$PWD"
mkdir -p logs/slurm

echo "Node: $(hostname)"
echo "Start: $(date)"

$PYTHON scripts/save_actin_kmeans_patches.py --n_examples 16 --no_umap

echo "End: $(date)"
find eval_results/actin_kmeans_patches/full_highdim/ -name "*.tif" | wc -l
echo "TIFFs written to full_highdim/"

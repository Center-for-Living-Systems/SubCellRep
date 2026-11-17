#!/usr/bin/env bash
#SBATCH --job-name=zyx_cp_km_patches
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/slurm/zyx_cp_km_patches_%j.out

set -eo pipefail
exec 2>&1

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
REPO="$PWD"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

$PYTHON scripts/save_zyx_cp_kmeans_patches.py --n_examples 16

echo ""
echo "End: $(date)"
echo ""
echo "Files written:"
find eval_results/zyx_cp_kmeans_patches/ -name "*.tif" | sort | head -40
echo "..."
find eval_results/zyx_cp_kmeans_patches/ -name "*.tif" | wc -l
echo "total .tif files"

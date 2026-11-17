#!/usr/bin/env bash
#SBATCH --job-name=dist_feat
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm/dist_feat_%j.out

# Extract distance-to-cell-boundary features (d00–d07, equiv_diam) for all
# DS1–DS4 from the cio_rb H5 files.  Output → ae_results/features/dist_features/
#
# Segmentation uses the same parameters as the cio_rb patchprep configs:
#   seg_threshold=0.1, seg_close_size=11, seg_min_size_final=30000

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON=/home/liyading/miniconda3/bin/python3
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm

echo "Node: $(hostname)"
echo "Start: $(date)"

time $PYTHON scripts/extract_dist_features.py --all

echo "End: $(date)"

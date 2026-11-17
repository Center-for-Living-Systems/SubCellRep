#!/usr/bin/env bash
#SBATCH --job-name=pptx_cp_combo
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm/pptx_cp_combo_%j.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm results

echo "Node: $(hostname)"
echo "Start: $(date)"

time $PYTHON scripts/make_pptx_cp_combo.py --out results/cp_combo_comparison.pptx

echo "End: $(date)"

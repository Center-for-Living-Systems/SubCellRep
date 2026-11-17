#!/usr/bin/env bash
#SBATCH --job-name=pptx_ablation
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:20:00
#SBATCH --output=logs/slurm/pptx_ablation_%j.out

set -eo pipefail
exec 2>&1
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$PWD:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"
mkdir -p logs/slurm results
echo "Node: $(hostname)"; echo "Start: $(date)"
time $PYTHON scripts/make_pptx_cp_ablation.py --out results/cp_ablation_comparison.pptx
echo "End: $(date)"

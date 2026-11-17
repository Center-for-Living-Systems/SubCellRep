#!/usr/bin/env bash
#SBATCH --job-name=fig2_panel
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm/fig2_panel_%j.out

set -eo pipefail
exec 2>&1

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$PWD"
mkdir -p logs/slurm

echo "Node: $(hostname)"
echo "Start: $(date)"

$PYTHON scripts/make_pptx_fig2_panel.py

echo "End: $(date)"
ls -lh results/fig2_panel.pptx 2>/dev/null

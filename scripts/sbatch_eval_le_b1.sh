#!/usr/bin/env bash
#SBATCH --job-name=eval_le_b1
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:30:00
#SBATCH --output=logs/slurm/eval_le_b1_%j.out

set -eo pipefail
exec 2>&1

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$PWD"
mkdir -p logs/slurm

echo "Node: $(hostname)"
echo "Start: $(date)"

$PYTHON scripts/eval_le_b1_ctrl_vs_ycomp.py

echo "End: $(date)"
ls -lh /net/projects/CLS/lding/data/fa_data_analysis/ae_results/features/eval_results/le_b1_ctrlvy_*_ds1.csv 2>/dev/null

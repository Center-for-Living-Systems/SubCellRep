#!/usr/bin/env bash
#SBATCH --job-name=plot_le
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:15:00
#SBATCH --output=logs/slurm/plot_le_%j.out

set -eo pipefail
exec 2>&1

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
mkdir -p logs/slurm

echo "Node: $(hostname)"
echo "Start: $(date)"

$PYTHON scripts/plot_le_b2_b12.py
$PYTHON scripts/plot_le_b2_fa4_perclass.py
$PYTHON scripts/plot_le_b1.py

echo "End: $(date)"
ls -lh results/le_b2_cp_il_supcon.png results/le_b12_cp_il_supcon.png \
        results/le_b2_fa4_recall.png results/le_b2_fa4_precision.png \
        results/le_b1_cp_il_supcon.png 2>/dev/null

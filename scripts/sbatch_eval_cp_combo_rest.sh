#!/usr/bin/env bash
#SBATCH --job-name=cp_combo_rest
#SBATCH --partition=general
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm/cp_combo_rest_%A_%a.out

# Evaluate remaining 3 feature combos (cp_s128, dist, cp_s32+s128)
# Task 0: LE budget study
# Task 1: Image efficiency

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm eval_results/patch_preds

FEATS="cp_s128 dist cp_s32+s128"

echo "Node: $(hostname)"
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Start: $(date)"

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    echo "=== LE budget study ==="
    time $PYTHON scripts/eval_le_cp_combo.py --clf lgbm logreg --feat $FEATS

elif [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
    echo "=== Image efficiency ==="
    time $PYTHON scripts/eval_image_efficiency_cp_combo.py --clf lgbm logreg --feat $FEATS
fi

echo "End: $(date)"

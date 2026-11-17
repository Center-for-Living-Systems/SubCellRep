#!/usr/bin/env bash
#SBATCH --job-name=cp_combo_eval
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm/cp_combo_eval_%j.out

# Evaluate CP-combo feature sets (cp_s32, cp_s128, dist, and combinations)
# on the B1/B2v2 LE budget study + image efficiency test.
#
# Task 0 ($SLURM_ARRAY_TASK_ID=0) : LE budget study (eval_le_cp_combo.py)
# Task 1 ($SLURM_ARRAY_TASK_ID=1) : Image efficiency (eval_image_efficiency_cp_combo.py)
#
# Usage:
#   # Both evals:
#   sbatch --array=0-1 scripts/sbatch_eval_cp_combo.sh
#   # LE only:
#   sbatch --array=0   scripts/sbatch_eval_cp_combo.sh
#   # Image efficiency only:
#   sbatch --array=1   scripts/sbatch_eval_cp_combo.sh

#SBATCH --array=0-1

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm eval_results/patch_preds

echo "Node: $(hostname)"
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Start: $(date)"

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    echo "=== LE budget study (lgbm + logreg) ==="
    time $PYTHON scripts/eval_le_cp_combo.py --clf lgbm logreg

elif [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
    echo "=== Image efficiency (lgbm + logreg) ==="
    time $PYTHON scripts/eval_image_efficiency_cp_combo.py --clf lgbm logreg
fi

echo "End: $(date)"

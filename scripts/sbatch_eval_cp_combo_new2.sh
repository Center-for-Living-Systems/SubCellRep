#!/usr/bin/env bash
#SBATCH --job-name=cp_new2
#SBATCH --partition=general
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/cp_new2_%A_%a.out

# New feature combos:
#   cp_s32+cp32ch3  : cp_s32 (ch1 pax) || cp_s32_ch3 (actin)
#   cp_s16+cp_s32   : cp_s16 || cp_s32  (two scales, same channel)
#
# Task 0: Label Efficiency
# Task 1: Image Efficiency

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm eval_results/patch_preds

echo "Node: $(hostname)"
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Start: $(date)"

FEATS="cp_s32+cp32ch3 cp_s16+cp_s32"

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    echo "=== LE budget study (new combos, append) ==="
    time $PYTHON scripts/eval_le_cp_combo.py \
        --clf lgbm logreg \
        --feat $FEATS \
        --append

elif [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
    echo "=== Image efficiency (new combos, append) ==="
    time $PYTHON scripts/eval_image_efficiency_cp_combo.py \
        --clf lgbm logreg \
        --feat $FEATS \
        --append
fi

echo "End: $(date)"

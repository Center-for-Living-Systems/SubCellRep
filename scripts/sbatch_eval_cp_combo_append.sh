#!/usr/bin/env bash
#SBATCH --job-name=cp_combo_app
#SBATCH --partition=general
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm/cp_combo_app_%A_%a.out

# Fill in missing feature combos and append to existing output CSVs.
# Task 0: LE with cp_s128, dist, cp_s32+s128  (append to LE file)
# Task 1: IE with cp_s32, cp_s32+dist, cp_s32+s128+dist  (append to IE file)

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
    echo "=== LE budget study (missing feats, append) ==="
    time $PYTHON scripts/eval_le_cp_combo.py \
        --clf lgbm logreg \
        --feat cp_s128 dist cp_s32+s128 \
        --append

elif [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
    echo "=== Image efficiency (missing feats, append) ==="
    time $PYTHON scripts/eval_image_efficiency_cp_combo.py \
        --clf lgbm logreg \
        --feat cp_s32 cp_s32+dist cp_s32+s128+dist \
        --append
fi

echo "End: $(date)"

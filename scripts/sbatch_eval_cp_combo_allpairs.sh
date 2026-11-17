#!/usr/bin/env bash
#SBATCH --job-name=cp_allpairs
#SBATCH --partition=general
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm/cp_allpairs_%A_%a.out

# All remaining cp_s32+X pairs not already running in job 2013768:
#   cp_s32+cp_s64       (ch1 64px scale)
#   cp_s32+cp_s16_ch2   (zyxin 16px)
#   cp_s32+cp_s32_ch2   (zyxin 32px)
#   cp_s32+cp_s64_ch2   (zyxin 64px)
#   cp_s32+cp_s128_ch2  (zyxin 128px)
#   cp_s32+cp_s16_ch3   (actin 16px)
#   cp_s32+cp_s64_ch3   (actin 64px)
#   cp_s32+cp_s128_ch3  (actin 128px)
#
# Task 0: Label Efficiency (append)
# Task 1: Image Efficiency (append)

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm eval_results/patch_preds

echo "Node: $(hostname)"
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Start: $(date)"

FEATS="cp_s32+cp_s64 \
       cp_s32+cp_s16_ch2 cp_s32+cp_s32_ch2 cp_s32+cp_s64_ch2 cp_s32+cp_s128_ch2 \
       cp_s32+cp_s16_ch3 cp_s32+cp_s64_ch3 cp_s32+cp_s128_ch3"

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    echo "=== LE budget study (all pairs, append) ==="
    time $PYTHON scripts/eval_le_cp_combo.py \
        --clf lgbm logreg \
        --feat $FEATS \
        --append

elif [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
    echo "=== Image efficiency (all pairs, append) ==="
    time $PYTHON scripts/eval_image_efficiency_cp_combo.py \
        --clf lgbm logreg \
        --feat $FEATS \
        --append
fi

echo "End: $(date)"

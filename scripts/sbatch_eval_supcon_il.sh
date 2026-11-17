#!/usr/bin/env bash
#SBATCH --job-name=supcon_il
#SBATCH --partition=general
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/supcon_il_%A_%a.out

# New combos with SupCon latents and ilastik features:
#   supcon_z               : 32-dim SupCon encoder latent (avg 3 seeds)
#   il_s32                 : ilastik 32px filter bank (80 feat)
#   cp_s32+supcon_z        : paxillin CP + SupCon latent
#   cp_s32+il_s32          : paxillin CP + ilastik
#   supcon_z+il_s32        : SupCon + ilastik
#   cp_s32+supcon_z+il_s32 : all three
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

FEATS="supcon_z il_s32 cp_s32+supcon_z cp_s32+il_s32 supcon_z+il_s32 cp_s32+supcon_z+il_s32"

if [[ "$SLURM_ARRAY_TASK_ID" == "0" ]]; then
    echo "=== LE budget study (supcon+ilastik, append) ==="
    time $PYTHON scripts/eval_le_cp_combo.py \
        --clf lgbm logreg \
        --feat $FEATS \
        --append

elif [[ "$SLURM_ARRAY_TASK_ID" == "1" ]]; then
    echo "=== Image efficiency (supcon+ilastik, append) ==="
    time $PYTHON scripts/eval_image_efficiency_cp_combo.py \
        --clf lgbm logreg \
        --feat $FEATS \
        --append
fi

echo "End: $(date)"

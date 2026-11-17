#!/usr/bin/env bash
# sbatch_b1b2_budget.sh
#
# 2-task array: task 0 = CellProfiler, task 1 = ilastik.
# Each task runs all three strategies with their respective budget lists:
#   b1_only / b2_only : N = 20, 40, ..., 260  (step 20, 13 points)
#   mixed             : N = 20, 40, ..., 500   (step 20→40 beyond 260), plus "all" (~519)
#
# Usage:
#   sbatch scripts/sbatch_b1b2_budget.sh
#   sbatch --export=CLF=lgbm,logreg scripts/sbatch_b1b2_budget.sh
#
#SBATCH --job-name=b1b2_budget
#SBATCH --partition=general
#SBATCH --array=0-1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:45:00
#SBATCH --output=logs/slurm/b1b2_budget_%A_%a.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

FEATS=("cp" "ilastik")
FEAT="${FEATS[$SLURM_ARRAY_TASK_ID]}"

: "${CLF:=lgbm}"

echo "Node: $(hostname)"
echo "Task: $SLURM_ARRAY_TASK_ID  feat=$FEAT  clf=$CLF"
echo "Start: $(date)"

$PYTHON "$REPO/scripts/eval_le_b1b2_budget.py" \
    --feat "$FEAT" \
    --clf  $CLF

echo "End: $(date)"

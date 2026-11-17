#!/usr/bin/env bash
# sbatch_b1b2v2_eval.sh
#
# 3-task array: task 0 = cross eval, task 1 = union eval, task 2 = budget study
# Each runs both CP and ilastik.
#
# Usage:
#   # Run all three evals:
#   sbatch --array=0-2 scripts/sbatch_b1b2v2_eval.sh
#
#   # Cross + union only:
#   sbatch --array=0-1 scripts/sbatch_b1b2v2_eval.sh
#
#   # Budget study only:
#   sbatch --array=2 scripts/sbatch_b1b2v2_eval.sh
#
#   # LogReg budget round (after LGBM):
#   sbatch --export=CLF=logreg --array=2 scripts/sbatch_b1b2v2_eval.sh
#
#SBATCH --job-name=b1b2v2_eval
#SBATCH --partition=general
#SBATCH --array=0-2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:30:00
#SBATCH --output=logs/slurm/b1b2v2_eval_%A_%a.out

set -eo pipefail
exec 2>&1

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
REPO="$PWD"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

: "${CLF:=lgbm}"

TASKS=("cross" "union" "budget")
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"

echo "Node: $(hostname)"
echo "Task: $SLURM_ARRAY_TASK_ID  eval=$TASK  clf=$CLF"
echo "Start: $(date)"
echo ""

case "$TASK" in
  cross)
    $PYTHON scripts/eval_le_b1b2v2_cross.py --feats cp ilastik --clf "$CLF"
    ;;
  union)
    $PYTHON scripts/eval_le_b1b2v2_union.py --feats cp ilastik --clf "$CLF"
    ;;
  budget)
    $PYTHON scripts/eval_le_b1b2v2_budget.py --feat cp ilastik --clf "$CLF"
    ;;
esac

echo ""
echo "End: $(date)"

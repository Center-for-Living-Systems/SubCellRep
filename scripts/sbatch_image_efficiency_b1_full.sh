#!/usr/bin/env bash
#SBATCH --job-name=img_eff_b1_full
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=3:00:00
#SBATCH --output=logs/slurm/img_eff_b1_full_%j.out

# Usage:
#   sbatch scripts/sbatch_image_efficiency_b1_full.sh
#   sbatch --export=CLF=logreg scripts/sbatch_image_efficiency_b1_full.sh

set -eo pipefail
exec 2>&1

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
REPO="$PWD"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm eval_results

: "${CLF:=lgbm}"

echo "Node: $(hostname)"
echo "clf: $CLF"
echo "Start: $(date)"
echo ""

$PYTHON scripts/eval_image_efficiency_b1_full.py --feats cp ilastik --clf "$CLF"

echo ""
echo "End: $(date)"

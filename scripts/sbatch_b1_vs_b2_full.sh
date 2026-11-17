#!/usr/bin/env bash
#SBATCH --job-name=b1_vs_b2_full
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm/b1_vs_b2_full_%A_%a.out

# Full B1 vs full B2, supcon + semisup AE (4 jobs).
#   python scripts/setup_b1_vs_b2_full.py
#   sbatch --array=0-3 scripts/sbatch_b1_vs_b2_full.sh

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm

JOB_LIST="config/b1_vs_b2_full/job_list.txt"
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOB_LIST")
NAME=$(basename "$CONFIG" .yaml)
RESULT_DIR="/net/projects/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run/b1_vs_b2_full/${NAME}"

echo "task $SLURM_ARRAY_TASK_ID -> $CONFIG"
echo "Node: $(hostname)  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"

$PYTHON scripts/run_ae_from_config.py "$CONFIG"

# Keep only the final model (+ latents/logs); drop best checkpoint and per-epoch PNGs
if [ -d "$RESULT_DIR" ] && [ -f "$RESULT_DIR/model_final.pt" ]; then
    rm -f "$RESULT_DIR/model_best.pt" "$RESULT_DIR"/*_ep[0-9]*.png
    echo "Cleaned intermediates in $RESULT_DIR"
    ls -la "$RESULT_DIR"
fi

echo "End: $(date)"

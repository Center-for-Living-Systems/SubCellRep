#!/usr/bin/env bash
#SBATCH --job-name=eval_ms_b2_ds1_ps64
#SBATCH --partition=general
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/eval_pptx_ms_b2_ds1_ps64_%j.out

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
mkdir -p logs/slurm results

JSON="$REPO/results/ms_b2_ds1_ps64_eval.json"

echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

echo "=== Eval: ps64 lat12p8 ==="
$PYTHON scripts/eval_ms_b2_ds1.py \
    --run-tag ms_b2_ds1_ps64_lat12p8 \
    --patch-size 64 --latent-dim 12 --proj-dim 8 \
    --save-json "$JSON"

echo ""
echo "=== Eval: ps32 lat12p8 ==="
$PYTHON scripts/eval_ms_b2_ds1.py \
    --run-tag ms_b2_ds1_ps32_lat12p8 \
    --patch-size 32 --latent-dim 12 --proj-dim 8 \
    --save-json "$JSON"

echo ""
echo "=== Eval: ps32 lat64p32 ==="
$PYTHON scripts/eval_ms_b2_ds1.py \
    --run-tag ms_b2_ds1_ps32_lat64p32 \
    --patch-size 32 --latent-dim 64 --proj-dim 32 \
    --save-json "$JSON"

echo ""
echo "=== Building PPT ==="
$PYTHON scripts/make_pptx_ms_b2_ds1_ps64.py

echo ""
echo "End: $(date)"

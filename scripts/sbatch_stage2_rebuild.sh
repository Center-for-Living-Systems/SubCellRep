#!/bin/bash
#SBATCH --job-name=s2_rebuild
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/stage2_rebuild_%a_%j.log
#SBATCH --array=0-6

# array 0 = stage2 pool (B1+B2 FA4 labels), 1 = whole set (no-ad + FA4 labels),
# 2 = stage2 pool + B1 FA4 labels only, 3 = stage2 pool + B2 FA4 labels only; batch 256
# 4-6 = same as 0,2,3 but pool restricted to p(adhesion)>=0.9 (8,260 patches)
REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

MODES=(stage2 whole stage2_b1 stage2_b2 stage2_conf90 stage2_b1_conf90 stage2_b2_conf90)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
echo "stage2 rebuild  mode=${MODE}"
$PYTHON scripts/run_stage2_rebuild_ae.py --mode ${MODE} --epochs 300 --batch-size 256 \
  && $PYTHON scripts/umap_stage2_rebuild.py --mode ${MODE}

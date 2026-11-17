#!/bin/bash
#SBATCH --job-name=s2_embed
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/embed_stage2_rebuild_%a_%j.log
#SBATCH --array=0-6

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

RUNS=(stage2 stage2_conf90 stage2_b1 stage2_b1_conf90 stage2_b2 stage2_b2_conf90 whole)
$PYTHON scripts/embed_stage2_rebuild.py --runs ${RUNS[$SLURM_ARRAY_TASK_ID]}

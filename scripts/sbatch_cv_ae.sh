#!/bin/bash
#SBATCH --job-name=cv_ae
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/cv_ae_%a_%j.log
#SBATCH --array=0-14

# 15 tasks = 3 label arms (b1, b2, b12) x 5 frame-grouped folds
REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"; mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
ARMS=(b1 b2 b12); T=$SLURM_ARRAY_TASK_ID
$PYTHON scripts/run_cv_ae.py --arm ${ARMS[$((T / 5))]} --fold $((T % 5))

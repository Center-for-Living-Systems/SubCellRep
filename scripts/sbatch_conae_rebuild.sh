#!/bin/bash
#SBATCH --job-name=conae_rebuild
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/conae_rebuild_%j.log

REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"; mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
$PYTHON scripts/run_conae_rebuild.py

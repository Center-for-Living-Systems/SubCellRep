#!/bin/bash
#SBATCH --job-name=s2_feat
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/features_stage2_rebuild_%a_%j.log
#SBATCH --array=0-15

# tasks 0-13: embeddings (feat x run); tasks 14-15: held-out logreg + confusion matrices (cp, ilastik)
REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"
mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4

RUNS=(stage2 stage2_conf90 stage2_b1 stage2_b1_conf90 stage2_b2 stage2_b2_conf90 whole)
FEATS=(cp ilastik)
T=$SLURM_ARRAY_TASK_ID
if [ "$T" -lt 14 ]; then
  $PYTHON scripts/embed_stage2_rebuild.py --feat ${FEATS[$((T / 7))]} --runs ${RUNS[$((T % 7))]}
else
  $PYTHON scripts/eval_stage2_rebuild_latents.py --feat ${FEATS[$((T - 14))]}
fi

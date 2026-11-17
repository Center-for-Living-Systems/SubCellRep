#!/bin/bash
#SBATCH --job-name=conae_eval
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/conae_eval_%a_%j.log
#SBATCH --array=0-7

# tasks 0-6: alternative embeddings of ConAE latents per run; task 7: probe + confusion matrices + kNN check
REPO=/net/projects/CLS/lding/gitcode/SubCellAE
PYTHON=/net/projects/CLS/lding/conda_env/core_env/bin/python
cd "$REPO"; mkdir -p logs
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
RUNS=(stage2 stage2_conf90 stage2_b1 stage2_b1_conf90 stage2_b2 stage2_b2_conf90 whole)
T=$SLURM_ARRAY_TASK_ID
if [ "$T" -lt 7 ]; then
  $PYTHON scripts/embed_stage2_rebuild.py --feat conae --runs ${RUNS[$T]}
else
  $PYTHON scripts/eval_stage2_rebuild_latents.py --feat conae && $PYTHON scripts/knn_separation_stage2_rebuild.py
fi

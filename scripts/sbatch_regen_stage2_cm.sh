#!/usr/bin/env bash
#SBATCH --job-name=regen_stage2_cm
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=logs/slurm/regen_stage2_cm_%j.out

set -eo pipefail
exec 2>&1

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$PWD"
export MPLBACKEND=Agg
mkdir -p logs/slurm

echo "Node: $(hostname)"
echo "Start: $(date)"

# Regenerate cm_{key}.png (latent, cp, ilastik)
$PYTHON scripts/eval_stage2_rebuild_latents.py --feat latent
$PYTHON scripts/eval_stage2_rebuild_latents.py --feat cp
$PYTHON scripts/eval_stage2_rebuild_latents.py --feat ilastik

# Regenerate cv_cm_{key}.png
$PYTHON scripts/figs_cv_b1b2.py

# Rebuild the PPT
$PYTHON scripts/make_pptx_stage2_rebuild.py

echo "End: $(date)"
ls -lh results/stage2_rebuild/cm_stage2.png results/cv_b1b2/cv_cm_CP.png 2>/dev/null

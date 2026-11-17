#!/usr/bin/env bash
#SBATCH --job-name=scale_ch23
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/slurm/scale_ch23_%j.out

# Extract multiscale CP and ilastik features for DS1 (vinc), channels 2 (zyxin)
# and 3 (actin), at all four spatial scales (s16/s32/s64/s128).
#
# Output:
#   ae_results/features/cp_scale/ds1_ch{2,3}_s{16,32,64,128}.csv
#   ae_results/features/ilastik_scale/ds1_ch{2,3}_s{16,32,64,128}.csv

set -eo pipefail
exec 2>&1

REPO="$PWD"
PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
export PYTHONPATH="$REPO:/net/projects/CLS/lding/conda_env/core_env/lib/python3.11/site-packages"

mkdir -p logs/slurm

echo "Node: $(hostname)"
echo "Start: $(date)"

for CH in 2 3; do
    echo ""
    echo "=== CP features, channel $CH ==="
    time $PYTHON scripts/extract_cp_scale_features.py --scale all --channel $CH

    echo ""
    echo "=== Ilastik features, channel $CH ==="
    time $PYTHON scripts/extract_ilastik_scale_features.py --scale all --channel $CH
done

echo ""
echo "End: $(date)"

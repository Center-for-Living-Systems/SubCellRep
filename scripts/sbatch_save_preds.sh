#!/usr/bin/env bash
#SBATCH --job-name=save_preds
#SBATCH --partition=general
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm/save_preds_%j.out

# Run all label efficiency and image efficiency evals with --save_preds.
# Per-patch predictions (fn, y_true, y_pred, prob_adhesion, ...) saved to:
#   eval_results/patch_preds/

set -eo pipefail
exec 2>&1

PYTHON="/net/projects/CLS/lding/conda_env/core_env/bin/python3"
REPO="$PWD"
export PYTHONPATH="$REPO"
mkdir -p logs/slurm eval_results/patch_preds

echo "Node: $(hostname)"
echo "Start: $(date)"
echo ""

# ── Label efficiency: B1/B2v2 cross ──────────────────────────────────────────
echo "=== LE cross (CP + ilastik, lgbm + logreg) ==="
$PYTHON scripts/eval_le_b1b2v2_cross.py \
    --feats cp ilastik --clf lgbm logreg --save_preds

# ── Label efficiency: B1/B2v2 union ──────────────────────────────────────────
echo ""
echo "=== LE union (CP + ilastik, lgbm + logreg) ==="
$PYTHON scripts/eval_le_b1b2v2_union.py \
    --feats cp ilastik --clf lgbm logreg --save_preds

# ── Label efficiency: B1/B2v2 budget (saves all-labels only) ─────────────────
echo ""
echo "=== LE budget (CP + ilastik, lgbm + logreg) ==="
$PYTHON scripts/eval_le_b1b2v2_budget.py \
    --feat cp ilastik --clf lgbm logreg --save_preds

# ── Image efficiency: B1 full (LOIO all-labels only) ─────────────────────────
echo ""
echo "=== Image efficiency B1 full (CP + ilastik, lgbm + logreg) ==="
$PYTHON scripts/eval_image_efficiency_b1_full.py \
    --feats cp ilastik --clf lgbm logreg --save_preds

# ── Image efficiency: B1/B2v2 (LOIO all-labels only) ─────────────────────────
echo ""
echo "=== Image efficiency B1B2v2 (CP + ilastik, lgbm + logreg) ==="
$PYTHON scripts/eval_image_efficiency_b1b2v2.py \
    --features cp ilastik --clf lgbm logreg --save_preds

echo ""
echo "End: $(date)"
echo ""
echo "Files written:"
ls -lh eval_results/patch_preds/

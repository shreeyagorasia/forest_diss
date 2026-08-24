#!/bin/bash
#
# Finding-5 full-prediction export (Table tab:age-height-error), for a given fold and variant.
# Fold 0 already done locally for both variants; this is for folds 1-4 (2026-08-24), so the
# age/height error table can pool across all 5 folds instead of fold-0-only.
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#
#   # One fold, one variant:
#   sbatch temp_results_pinn/jobs/run_finding5_export_cluster.sh 1 plain_pinn
#
#   # All 4 remaining folds, one variant, as an array:
#   sbatch --array=1-4 temp_results_pinn/jobs/run_finding5_export_cluster.sh "" plain_pinn
#   sbatch --array=1-4 temp_results_pinn/jobs/run_finding5_export_cluster.sh "" pinn_k
#
# Arguments:
#   fold_index   1-4 (fold 0 already exists). Leave blank ("") when submitting with --array.
#   variant      plain_pinn or pinn_k. Required.
#
# Results:
#   temp_results_pinn/outputs/example_curve/plain_pinn_fixed_full_predictions_fold<i>.csv
#   temp_results_pinn/outputs/example_curve/pinn_k_fixed_full_predictions_fold<i>.csv

#SBATCH --job-name=finding5_export
#SBATCH --output=logs/temp_results_pinn/%x_%A_%a.out
#SBATCH --error=logs/temp_results_pinn/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/temp_results_pinn

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

FOLD_INDEX=${1:-${SLURM_ARRAY_TASK_ID:-1}}
VARIANT=${2:?"variant is required: plain_pinn or pinn_k"}

echo "--- Finding-5 export job start ---"
echo "Node: $(hostname)"
echo "Fold index: ${FOLD_INDEX}"
echo "Variant: ${VARIANT}"

if [ "${VARIANT}" = "plain_pinn" ]; then
  python -u temp_results_pinn/pinn_env_terrain_fix/run_finding5_plain_pinn_export.py --fold-index "${FOLD_INDEX}"
else
  python -u temp_results_pinn/pinn_env_terrain_fix/run_finding5_pinn_k_export.py --fold-index "${FOLD_INDEX}"
fi

echo "--- Finding-5 export job end ---"

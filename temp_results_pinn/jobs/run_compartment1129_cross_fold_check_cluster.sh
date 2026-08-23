#!/bin/bash
#
# Cross-fold check: does compartment 1129 (blk=21) still get an inflated y_max prediction when
# the model HAS been trained on its own data (folds 1-4, where it's part of training), not just
# when held out (fold 0, where the inflation was originally found)? Distinguishes a genuine site
# effect from a held-out generalization artifact. See
# temp_results_pinn/pinn_env_terrain_fix/run_compartment1129_cross_fold_check.py.
# Trains plain PINN only (single variant), ~10-12 min per fold based on earlier local runs.
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#
#   # One fold:
#   sbatch temp_results_pinn/jobs/run_compartment1129_cross_fold_check_cluster.sh 0
#
#   # All 5 folds (job array):
#   sbatch --array=0-4 temp_results_pinn/jobs/run_compartment1129_cross_fold_check_cluster.sh ""
#
# Arguments:
#   fold_index  0-4. Leave blank ("") when submitting with --array.
#
# Results:
#   temp_results_pinn/outputs/CORRECTED_2026-08-23_mechanism_checks/compartment1129_cross_fold/fold_<i>/summary.json

#SBATCH --job-name=pinn_cpmt1129_check
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

FOLD_INDEX=${1:-${SLURM_ARRAY_TASK_ID:-0}}

echo "--- Compartment 1129 cross-fold check job start ---"
echo "Node: $(hostname)"
echo "Fold index: ${FOLD_INDEX}"

python -u temp_results_pinn/pinn_env_terrain_fix/run_compartment1129_cross_fold_check.py \
  --fold-index "${FOLD_INDEX}"

echo "--- Compartment 1129 cross-fold check job end ---"

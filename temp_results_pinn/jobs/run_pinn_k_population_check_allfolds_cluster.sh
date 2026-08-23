#!/bin/bash
#
# Runs the PINN-k population check (y_max AND k) on a given fold, saving blk/cpmt so all 5
# folds' test-set predictions can be pooled into one full-forest compartment map, matching what
# the plain-PINN map already does. See
# temp_results_pinn/pinn_env_terrain_fix/run_pinn_k_population_check_allfolds.py.
# ~10-12 min per fold, same architecture/data size as the plain-PINN job it mirrors.
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#
#   # Folds 1-4 (fold 0 already exists locally, see example_curve/test_set_predictions.csv):
#   sbatch --array=1-4 temp_results_pinn/jobs/run_pinn_k_population_check_allfolds_cluster.sh ""
#
# Arguments:
#   fold_index  0-4. Leave blank ("") when submitting with --array.
#
# Results:
#   temp_results_pinn/outputs/CORRECTED_2026-08-23_mechanism_checks/pinn_k_population_check/fold_<i>/predictions.csv

#SBATCH --job-name=pinn_k_popcheck
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

echo "--- PINN-k population check job start ---"
echo "Node: $(hostname)"
echo "Fold index: ${FOLD_INDEX}"

python -u temp_results_pinn/pinn_env_terrain_fix/run_pinn_k_population_check_allfolds.py \
  --fold-index "${FOLD_INDEX}"

echo "--- PINN-k population check job end ---"

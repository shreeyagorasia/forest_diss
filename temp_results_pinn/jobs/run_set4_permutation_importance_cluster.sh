#!/bin/bash
#
# Runs the Set4 permutation-importance check (why doesn't the broader feature set help over
# Set3?) for a given fold. Trains PINN-k on Set4, ~8-15 min based on the Set3 fold-0 local run.
# See temp_results_pinn/pinn_env_terrain_fix/run_set4_permutation_importance.py.
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#
#   # One fold:
#   sbatch temp_results_pinn/jobs/run_set4_permutation_importance_cluster.sh 0
#
#   # Several folds (job array):
#   sbatch --array=0-2 temp_results_pinn/jobs/run_set4_permutation_importance_cluster.sh ""
#
# Arguments:
#   fold_index  0-4. Leave blank ("") when submitting with --array.
#
# Results:
#   temp_results_pinn/outputs/CORRECTED_2026-08-23_mechanism_checks/set4_permutation_importance/fold_<i>/summary.json

#SBATCH --job-name=pinn_set4_perm_importance
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

echo "--- PINN Set4 permutation importance job start ---"
echo "Node: $(hostname)"
echo "Fold index: ${FOLD_INDEX}"

python -u temp_results_pinn/pinn_env_terrain_fix/run_set4_permutation_importance.py \
  --fold-index "${FOLD_INDEX}"

echo "--- PINN Set4 permutation importance job end ---"

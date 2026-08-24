#!/bin/bash
#
# Trunk-residual + terrain-extrapolation mechanism checks (Table tab:trunk-residual), for a
# given fold. Trains BOTH plain PINN and PINN-k in one job. Folds 0-2 already done locally;
# this is for folds 3-4 (2026-08-24), so the table covers all 5 folds instead of 3.
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch temp_results_pinn/jobs/run_pinn_mechanism_checks_cluster.sh 3
#   sbatch temp_results_pinn/jobs/run_pinn_mechanism_checks_cluster.sh 4
#
#   # Or both at once as an array:
#   sbatch --array=3-4 temp_results_pinn/jobs/run_pinn_mechanism_checks_cluster.sh ""
#
# Arguments:
#   fold_index   0-4. Leave blank ("") when submitting with --array.
#
# Results:
#   temp_results_pinn/outputs/CORRECTED_2026-08-23_mechanism_checks/trunk_and_terrain/fold_<i>/summary.json

#SBATCH --job-name=pinn_mech_checks
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

echo "--- PINN mechanism checks (trunk-residual) job start ---"
echo "Node: $(hostname)"
echo "Fold index: ${FOLD_INDEX}"

python -u temp_results_pinn/pinn_env_terrain_fix/run_pinn_mechanism_checks.py \
  --fold-index "${FOLD_INDEX}"

echo "--- PINN mechanism checks job end ---"

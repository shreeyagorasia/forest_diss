#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k.sh [cohort] [split_type] [run_name] [split_seed] [n_folds] [fold_index]
#
# Examples:
#   sbatch jobs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k.sh 4survey
#   sbatch jobs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k.sh 4survey spatial_block
#   sbatch jobs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k.sh 4survey spatial_block_kfold "" 42 5 0
#   sbatch jobs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k.sh 4survey spatial_block pinn_env_terrain_k_freezeymax
#
# Arguments:
#   cohort      4survey or 6survey. Omit to evaluate both cohorts.
#   split_type  temporal, temporal_narrow_gap, spatial_block, or spatial_block_kfold. Defaults
#               to temporal.
#   run_name    Must match the --run-name used when fitting, if one was used. Blank by default.
#   split_seed  Must match the --split-seed used when fitting. Defaults to 42.
#   n_folds     Must match --n-folds used when fitting, for split_type=spatial_block_kfold.
#               Defaults to 5.
#   fold_index  Must match --fold-index used when fitting, for split_type=spatial_block_kfold.
#               Defaults to 0.
#
# Logs:
#   stdout -> logs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k_<jobid>.out
#   stderr -> logs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k_<jobid>.err
#
# Notes:
#   This is a CPU job. It does not request a GPU.

#SBATCH -p Teaching
#SBATCH --job-name=evaluate_pinn_env_terrain_k
#SBATCH --output=logs/pinn_env_terrain_k/%x_%j.out
#SBATCH --error=logs/pinn_env_terrain_k/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G

cd ~/forest_diss

mkdir -p logs/pinn_env_terrain_k outputs

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-}
SPLIT_TYPE=${2:-temporal}
RUN_NAME=${3:-}
SPLIT_SEED=${4:-42}
N_FOLDS=${5:-5}
FOLD_INDEX=${6:-0}

echo "--- PINN env_terrain_k evaluate job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT:-both}"
echo "Split type: ${SPLIT_TYPE}"
echo "Run name: ${RUN_NAME:-(none, uses default pinn_env_terrain_k path)}"
echo "Split seed: ${SPLIT_SEED}"
echo "K-fold: ${FOLD_INDEX}/${N_FOLDS}"

EXTRA_ARGS=()
if [ -n "${COHORT}" ]; then
  EXTRA_ARGS+=(--cohort "${COHORT}")
fi
if [ -n "${RUN_NAME}" ]; then
  EXTRA_ARGS+=(--run-name "${RUN_NAME}")
fi

python -u -m models.pinn_env_terrain_k.evaluate_pinn_env_terrain_k \
  --split-type "${SPLIT_TYPE}" \
  --split-seed "${SPLIT_SEED}" \
  --n-folds "${N_FOLDS}" \
  --fold-index "${FOLD_INDEX}" \
  "${EXTRA_ARGS[@]}"

echo "--- PINN env_terrain_k evaluate job end ---"

#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh [cohort] [max_epochs] [patience] [split_type] [physics_weight] [trajectory_weight] [run_name] [seed] [batch_size] [feature_set] [dropout_rate] [split_seed] [n_folds] [fold_index] [learning_rate] [hidden_layer_sizes]
#
# Examples:
#   sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 4survey 5 3
#   sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 6survey 500 20
#   sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 4survey 500 20 spatial_block
#   sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 4survey 500 40 spatial_block 1.0 1.0 pinn_env_terrain_base
#   sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 4survey 500 40 spatial_block 0.05 0.05 pinn_env_terrain_pw0.05_tw0.05 42 128
#   sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 4survey 500 40 spatial_block 1.0 1.0 pinn_env_terrain_extended 42 128 terrain_wind_extended
#   sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh 4survey 500 40 spatial_block 1.0 1.0 pinn_env_terrain_drop0.2 42 128 terrain_wind_solid 0.2
#
# Arguments:
#   cohort             4survey or 6survey. Defaults to 4survey.
#   max_epochs         Maximum training epochs. Defaults to 5 for a quick test.
#   patience           Early-stopping patience. Defaults to 3 for a quick test.
#   split_type         temporal, temporal_narrow_gap, or spatial_block. Defaults to temporal.
#   physics_weight     Weight on the physics loss term. Defaults to 1.0 (the untested base case
#                      -- pinn_noenv's own low-weight sweep result is NOT assumed to carry over
#                      here, see run_pinn_env_terrain.py's module docstring).
#   trajectory_weight  Weight on the trajectory loss term. Defaults to 1.0.
#   run_name           Only changes where results are saved -- set this whenever
#                      physics_weight/trajectory_weight aren't both 1.0, or seed/feature_set/
#                      dropout_rate isn't the default, so the run doesn't overwrite the primary
#                      pinn_env_terrain result. Blank by default.
#   seed               Random seed (network init + batch shuffling). Defaults to 42.
#   batch_size         Main + pairs training batch size. Defaults to 128 (matching pinn_noenv's
#                      default).
#   feature_set        Named terrain/wind feature set for the y_max sub-network
#                      (terrain_wind_solid, terrain_wind_extended, or broad -- see
#                      ENV_TERRAIN_FEATURE_SETS in models/common/torch_data.py). Defaults to
#                      terrain_wind_solid.
#   dropout_rate       Dropout probability in both the main network's and the y_max
#                      sub-network's hidden layers. Defaults to 0.0 (no dropout, matching
#                      pinn_noenv's architecture).
#   learning_rate      Adam/SGD starting learning rate. Defaults to 0.0001, never swept for this
#                      model before 2026-08-06.
#   hidden_layer_sizes Comma-separated hidden layer sizes, e.g. "64,32". Blank/default: the
#                      original 3x128 network, unchanged.
#
# Logs:
#   stdout -> logs/pinn_env_terrain/pinn_env_terrain_<jobid>.out
#   stderr -> logs/pinn_env_terrain/pinn_env_terrain_<jobid>.err
#
# Results:
#   outputs/<split_type>/pinn_env_terrain/<cohort>/
#
# Prerequisite:
#   PINN reads frozen Chapman-Richards parameters from
#   outputs/<split_type>/chapman_richards/<cohort>/params.json -- the split-MATCHED fit, so the
#   baseline outputs for THIS split_type specifically must exist first (see
#   run_pinn_env_terrain.py's load_cr_params() for why). Also reads the terrain/wind feature
#   columns from data/processed/environmental/plot_environmental_features.parquet -- built by
#   aux_data_resolution_check.ipynb, not re-derived here. Make sure both are on the cluster
#   before submitting this job.

#SBATCH --job-name=pinn_env_terrain
#SBATCH --output=logs/pinn_env_terrain/%x_%j.out
#SBATCH --error=logs/pinn_env_terrain/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/pinn_env_terrain outputs

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-4survey}
MAX_EPOCHS=${2:-5}
PATIENCE=${3:-3}
SPLIT_TYPE=${4:-temporal}
PHYSICS_WEIGHT=${5:-1.0}
TRAJECTORY_WEIGHT=${6:-1.0}
RUN_NAME=${7:-}
SEED=${8:-42}
BATCH_SIZE=${9:-128}
FEATURE_SET=${10:-terrain_wind_solid}
DROPOUT_RATE=${11:-0.0}
SPLIT_SEED=${12:-42}
N_FOLDS=${13:-5}
FOLD_INDEX=${14:-0}
LEARNING_RATE=${15:-0.0001}
HIDDEN_LAYER_SIZES=${16:-}

echo "--- PINN env_terrain job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Patience: ${PATIENCE}"
echo "Split type: ${SPLIT_TYPE}"
echo "Physics weight: ${PHYSICS_WEIGHT}"
echo "Trajectory weight: ${TRAJECTORY_WEIGHT}"
echo "Run name: ${RUN_NAME:-(none, uses default pinn_env_terrain path)}"
echo "Seed: ${SEED}"
echo "Batch size: ${BATCH_SIZE}"
echo "Feature set: ${FEATURE_SET}"
echo "Dropout rate: ${DROPOUT_RATE}"
echo "Split seed: ${SPLIT_SEED}"
echo "K-fold: ${FOLD_INDEX}/${N_FOLDS}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Hidden layer sizes: ${HIDDEN_LAYER_SIZES:-(default 3x128)}"

RUN_NAME_ARGS=()
if [ -n "${RUN_NAME}" ]; then
  RUN_NAME_ARGS=(--run-name "${RUN_NAME}")
fi

HIDDEN_LAYER_SIZES_ARGS=()
if [ -n "${HIDDEN_LAYER_SIZES}" ]; then
  HIDDEN_LAYER_SIZES_ARGS=(--hidden-layer-sizes "${HIDDEN_LAYER_SIZES}")
fi

python -u -m models.pinn_env_terrain.run_pinn_env_terrain \
  --cohort "${COHORT}" \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --split-type "${SPLIT_TYPE}" \
  --physics-weight "${PHYSICS_WEIGHT}" \
  --trajectory-weight "${TRAJECTORY_WEIGHT}" \
  --seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --pairs-batch-size "${BATCH_SIZE}" \
  --feature-set "${FEATURE_SET}" \
  --dropout-rate "${DROPOUT_RATE}" \
  --split-seed "${SPLIT_SEED}" \
  --n-folds "${N_FOLDS}" \
  --fold-index "${FOLD_INDEX}" \
  --learning-rate "${LEARNING_RATE}" \
  "${RUN_NAME_ARGS[@]}" \
  "${HIDDEN_LAYER_SIZES_ARGS[@]}"

echo "--- PINN env_terrain job end ---"

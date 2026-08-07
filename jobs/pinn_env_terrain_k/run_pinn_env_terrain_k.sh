#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh [cohort] [max_epochs] [patience] [split_type] [physics_weight] [trajectory_weight] [run_name] [seed] [batch_size] [feature_set] [dropout_rate] [split_seed] [n_folds] [fold_index] [freeze_y_max] [learning_rate] [hidden_layer_sizes]
#
# Examples:
#   sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 4survey 5 3
#   sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 4survey 500 40 spatial_block
#   sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 4survey 500 40 spatial_block_kfold 1.0 1.0 pinn_env_terrain_k_kfold 42 128 terrain_wind_solid 0.0 42 5 0
#   sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh 4survey 500 40 spatial_block 1.0 1.0 pinn_env_terrain_k_freezeymax 42 128 terrain_wind_solid 0.0 42 5 0 true
#
# Arguments:
#   cohort             4survey or 6survey. Defaults to 4survey.
#   max_epochs         Maximum training epochs. Defaults to 5 for a quick test.
#   patience           Early-stopping patience. Defaults to 3 for a quick test.
#   split_type         temporal, temporal_narrow_gap, spatial_block, or spatial_block_kfold.
#                      Defaults to temporal.
#   physics_weight     Weight on the physics loss term. Defaults to 1.0 (the untested base case).
#   trajectory_weight  Weight on the trajectory loss term. Defaults to 1.0.
#   run_name           Only changes where results are saved -- set this whenever
#                      physics_weight/trajectory_weight aren't both 1.0, seed/feature_set/
#                      dropout_rate isn't the default, or freeze_y_max is used, so the run
#                      doesn't overwrite the primary pinn_env_terrain_k result. Blank by default.
#   seed               Random seed (network init + batch shuffling). Defaults to 42.
#   batch_size         Main + pairs training batch size. Defaults to 128.
#   feature_set        Named terrain/wind feature set, shared by both the y_max and k
#                      sub-networks (terrain_wind_solid, terrain_wind_extended, broad,
#                      terrain_wind_full, or broad_legitimate -- see ENV_TERRAIN_FEATURE_SETS in
#                      models/common/torch_data.py). Defaults to terrain_wind_solid.
#   dropout_rate       Dropout probability in the main network's and both sub-networks' hidden
#                      layers. Defaults to 0.0.
#   split_seed         Seed for spatial_block/spatial_block_kfold's own block-shuffle. Defaults
#                      to 42. load_cr_params() reads the matching CR anchor for a non-default
#                      value -- run jobs/baselines/run_baselines.sh for this split_type/split_seed
#                      (and matching fold_index, if spatial_block_kfold) first.
#   n_folds            Number of folds for split_type=spatial_block_kfold. Defaults to 5.
#                      Ignored for every other split type.
#   fold_index         Which fold to hold out as test, for split_type=spatial_block_kfold
#                      (0-indexed, must be < n_folds). Defaults to 0. Ignored for every other
#                      split type.
#   freeze_y_max       Pass "true" to run the council's freeze-one-vary-other ablation (pins
#                      y_max to the global CR constant, only k is a free per-plot parameter).
#                      Blank/anything else = full two-parameter model (default).
#   learning_rate      Adam/SGD starting learning rate. Defaults to 0.0001, never swept for this
#                      model before 2026-08-06.
#   hidden_layer_sizes Comma-separated hidden layer sizes, e.g. "64,32". Blank/default: the
#                      original 3x128 network, unchanged.
#
# Logs:
#   stdout -> logs/pinn_env_terrain_k/pinn_env_terrain_k_<jobid>.out
#   stderr -> logs/pinn_env_terrain_k/pinn_env_terrain_k_<jobid>.err
#
# Results:
#   outputs/<split_type>/pinn_env_terrain_k/<cohort>/ (or .../fold_<fold_index>/ under
#   spatial_block_kfold, or the run_name's own path if one was given)
#
# Prerequisite:
#   Reads frozen Chapman-Richards parameters from
#   outputs/<split_type>/chapman_richards[_fold<N>]/<cohort>/params.json -- the split-MATCHED
#   (and, under spatial_block_kfold, fold-MATCHED) fit, so run jobs/baselines/run_baselines.sh
#   for this split_type/split_seed/fold_index first. Also reads the terrain/wind feature columns
#   from data/processed/environmental/plot_environmental_features.parquet.

#SBATCH --job-name=pinn_env_terrain_k
#SBATCH --output=logs/pinn_env_terrain_k/%x_%j.out
#SBATCH --error=logs/pinn_env_terrain_k/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/pinn_env_terrain_k outputs

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
FREEZE_Y_MAX=${15:-}
LEARNING_RATE=${16:-0.0001}
HIDDEN_LAYER_SIZES=${17:-}

echo "--- PINN env_terrain_k job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Patience: ${PATIENCE}"
echo "Split type: ${SPLIT_TYPE}"
echo "Physics weight: ${PHYSICS_WEIGHT}"
echo "Trajectory weight: ${TRAJECTORY_WEIGHT}"
echo "Run name: ${RUN_NAME:-(none, uses default pinn_env_terrain_k path)}"
echo "Seed: ${SEED}"
echo "Batch size: ${BATCH_SIZE}"
echo "Feature set: ${FEATURE_SET}"
echo "Dropout rate: ${DROPOUT_RATE}"
echo "Split seed: ${SPLIT_SEED}"
echo "K-fold: ${FOLD_INDEX}/${N_FOLDS}"
echo "Freeze y_max: ${FREEZE_Y_MAX:-false}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Hidden layer sizes: ${HIDDEN_LAYER_SIZES:-(default 3x128)}"

RUN_NAME_ARGS=()
if [ -n "${RUN_NAME}" ]; then
  RUN_NAME_ARGS=(--run-name "${RUN_NAME}")
fi

FREEZE_Y_MAX_ARGS=()
if [ "${FREEZE_Y_MAX}" = "true" ]; then
  FREEZE_Y_MAX_ARGS=(--freeze-y-max)
fi

HIDDEN_LAYER_SIZES_ARGS=()
if [ -n "${HIDDEN_LAYER_SIZES}" ]; then
  HIDDEN_LAYER_SIZES_ARGS=(--hidden-layer-sizes "${HIDDEN_LAYER_SIZES}")
fi

python -u -m models.pinn_env_terrain_k.run_pinn_env_terrain_k \
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
  "${FREEZE_Y_MAX_ARGS[@]}" \
  "${HIDDEN_LAYER_SIZES_ARGS[@]}"

echo "--- PINN env_terrain_k job end ---"

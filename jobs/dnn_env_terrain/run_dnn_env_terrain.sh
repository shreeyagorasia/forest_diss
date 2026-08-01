#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh [cohort] [max_epochs] [patience] [split_type] [seed] [run_name] [batch_size] [feature_set] [dropout_rate]
#
# Examples:
#   sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 4survey 5 3
#   sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 6survey 500 20
#   sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 4survey 500 20 spatial_block
#   sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 6survey 500 40 temporal 43 dnn_env_terrain_seed43
#   sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 4survey 500 40 spatial_block 42 dnn_env_terrain_extended 512 terrain_wind_extended
#   sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh 4survey 500 40 spatial_block 42 dnn_env_terrain_drop0.2 512 terrain_wind_solid 0.2
#
# Arguments:
#   cohort         4survey or 6survey. Defaults to 4survey.
#   max_epochs     Maximum training epochs. Defaults to 5 for a quick test.
#   patience       Early-stopping patience. Defaults to 3 for a quick test.
#   split_type     temporal, temporal_narrow_gap, or spatial_block. Defaults to temporal.
#   seed           Random seed (network init + batch shuffling). Defaults to 42 (the model's
#                  own default) -- pass a different value to test run-to-run variance.
#   run_name       Only changes where results are saved -- set this whenever seed/feature_set/
#                  dropout_rate isn't the default, so the run doesn't overwrite the primary
#                  dnn_env_terrain checkpoint. Blank by default.
#   batch_size     Training batch size. Defaults to 512 (matching dnn_noenv's default).
#   feature_set    Named terrain/wind feature set (terrain_wind_solid, terrain_wind_extended, or
#                  broad -- see ENV_TERRAIN_FEATURE_SETS in models/common/torch_data.py). Defaults
#                  to terrain_wind_solid.
#   dropout_rate   Dropout probability in the main network's hidden layers. Defaults to 0.0 (no
#                  dropout, matching dnn_noenv's architecture).
#
# Logs:
#   stdout -> logs/dnn_env_terrain/dnn_env_terrain_<jobid>.out
#   stderr -> logs/dnn_env_terrain/dnn_env_terrain_<jobid>.err
#
# Results:
#   outputs/<split_type>/dnn_env_terrain/<cohort>/
#
# Prerequisite:
#   Reads the terrain/wind feature columns from
#   data/processed/environmental/plot_environmental_features.parquet -- built by
#   aux_data_resolution_check.ipynb, not re-derived here. Make sure that parquet file has been
#   transferred to the cluster before submitting this job.

#SBATCH --job-name=dnn_env_terrain
#SBATCH --output=logs/dnn_env_terrain/%x_%j.out
#SBATCH --error=logs/dnn_env_terrain/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/dnn_env_terrain outputs

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-4survey}
MAX_EPOCHS=${2:-5}
PATIENCE=${3:-3}
SPLIT_TYPE=${4:-temporal}
SEED=${5:-42}
RUN_NAME=${6:-}
BATCH_SIZE=${7:-512}
FEATURE_SET=${8:-terrain_wind_solid}
DROPOUT_RATE=${9:-0.0}

echo "--- DNN env_terrain job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Patience: ${PATIENCE}"
echo "Split type: ${SPLIT_TYPE}"
echo "Seed: ${SEED}"
echo "Run name: ${RUN_NAME:-(none, uses default dnn_env_terrain path)}"
echo "Batch size: ${BATCH_SIZE}"
echo "Feature set: ${FEATURE_SET}"
echo "Dropout rate: ${DROPOUT_RATE}"

RUN_NAME_ARGS=()
if [ -n "${RUN_NAME}" ]; then
  RUN_NAME_ARGS=(--run-name "${RUN_NAME}")
fi

python -u -m models.dnn_env_terrain.run_dnn_env_terrain \
  --cohort "${COHORT}" \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --split-type "${SPLIT_TYPE}" \
  --seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --feature-set "${FEATURE_SET}" \
  --dropout-rate "${DROPOUT_RATE}" \
  "${RUN_NAME_ARGS[@]}"

echo "--- DNN env_terrain job end ---"

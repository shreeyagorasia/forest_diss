#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh [cohort] [max_epochs] [patience] [split_type] [seed] [run_name] [batch_size] [feature_set] [dropout_rate] [split_seed] [n_folds] [fold_index] [learning_rate] [hidden_layer_sizes]
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
#   learning_rate  Adam/SGD starting learning rate. Defaults to 0.0001 (the model's own default --
#                  see documentation/experiment_log.md's 2026-08-02 entry, never swept for this
#                  model before 2026-08-06).
#   hidden_layer_sizes  Comma-separated hidden layer sizes, e.g. "64,32". Blank/default: the
#                  original 3x128 network, unchanged.
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

# Toolchain lives in a shared TA home directory under a dated folder name that gets replaced
# periodically -- hardcoding one date breaks silently (a confusing torch/CUDA import error deep
# inside the Python job, not an obvious "toolchain missing" message) the next time it rotates.
# Finds whatever toolchain-* currently exists instead, picks the most recently modified one, and
# fails loudly with a clear message immediately if none exist at all.
echo "Node: $(hostname)"  # printed BEFORE the toolchain check, on purpose -- if
# /home/htang2 isn'"'"'t mounted on this specific node, everything below dies immediately, and
# without this line the log would never say which node was the problem.
TOOLCHAIN_RC=$(ls -1t /home/htang2/toolchain-*/toolchain.rc 2>/dev/null | head -1)
if [ -z "${TOOLCHAIN_RC}" ]; then
  echo "ERROR: no toolchain.rc found under /home/htang2/toolchain-*/ on node $(hostname) -- /home/htang2 may not be mounted here. Ask a TA if this recurs on the same node." >&2
  exit 1
fi
. "${TOOLCHAIN_RC}"
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
SPLIT_SEED=${10:-42}
N_FOLDS=${11:-5}
FOLD_INDEX=${12:-0}
LEARNING_RATE=${13:-0.0001}
HIDDEN_LAYER_SIZES=${14:-}

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

python -u -m models.dnn_env_terrain.run_dnn_env_terrain \
  --cohort "${COHORT}" \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --split-type "${SPLIT_TYPE}" \
  --seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --feature-set "${FEATURE_SET}" \
  --dropout-rate "${DROPOUT_RATE}" \
  --split-seed "${SPLIT_SEED}" \
  --n-folds "${N_FOLDS}" \
  --fold-index "${FOLD_INDEX}" \
  --learning-rate "${LEARNING_RATE}" \
  "${RUN_NAME_ARGS[@]}" \
  "${HIDDEN_LAYER_SIZES_ARGS[@]}"

echo "--- DNN env_terrain job end ---"

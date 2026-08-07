#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/dnn_env_terrain/evaluate_dnn_env_terrain.sh [cohort] [split_type] [run_name]
#
# Examples:
#   sbatch jobs/dnn_env_terrain/evaluate_dnn_env_terrain.sh 4survey
#   sbatch jobs/dnn_env_terrain/evaluate_dnn_env_terrain.sh 6survey temporal
#   sbatch jobs/dnn_env_terrain/evaluate_dnn_env_terrain.sh 4survey spatial_block
#   sbatch jobs/dnn_env_terrain/evaluate_dnn_env_terrain.sh 4survey spatial_block dnn_env_terrain_extended
#   sbatch jobs/dnn_env_terrain/evaluate_dnn_env_terrain.sh
#
# Arguments:
#   cohort      4survey or 6survey. Omit to evaluate both cohorts.
#   split_type  temporal, temporal_narrow_gap, or spatial_block. Defaults to temporal.
#   run_name    Must match the --run-name used when fitting, if one was used. Blank by default.
#
# Purpose:
#   Evaluates an already-trained dnn_env_terrain checkpoint on the test split.
#   Run jobs/dnn_env_terrain/run_dnn_env_terrain.sh first, with the SAME split_type and run_name.
#
# Logs:
#   stdout -> logs/dnn_env_terrain/evaluate_dnn_env_terrain_<jobid>.out
#   stderr -> logs/dnn_env_terrain/evaluate_dnn_env_terrain_<jobid>.err
#
# Results:
#   outputs/<split_type>/dnn_env_terrain/<cohort>/predictions.csv
#   outputs/<split_type>/dnn_env_terrain/<cohort>/metrics.json
#
# Notes:
#   This is a CPU job. It does not request a GPU.

#SBATCH -p Teaching
#SBATCH --job-name=evaluate_dnn_env_terrain
#SBATCH --output=logs/dnn_env_terrain/%x_%j.out
#SBATCH --error=logs/dnn_env_terrain/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G

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

COHORT=${1:-}
SPLIT_TYPE=${2:-temporal}
RUN_NAME=${3:-}
SPLIT_SEED=${4:-42}
N_FOLDS=${5:-5}
FOLD_INDEX=${6:-0}

echo "--- DNN env_terrain evaluate job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT:-both}"
echo "Split type: ${SPLIT_TYPE}"
echo "Run name: ${RUN_NAME:-(none, uses default dnn_env_terrain path)}"
echo "K-fold: ${FOLD_INDEX}/${N_FOLDS}"

RUN_NAME_ARGS=()
if [ -n "${RUN_NAME}" ]; then
  RUN_NAME_ARGS=(--run-name "${RUN_NAME}")
fi

COHORT_ARGS=()
if [ -n "$COHORT" ]; then
  COHORT_ARGS=(--cohort "${COHORT}")
fi

python -u -m models.dnn_env_terrain.evaluate_dnn_env_terrain \
  --split-type "${SPLIT_TYPE}" \
  --split-seed "${SPLIT_SEED}" \
  --n-folds "${N_FOLDS}" \
  --fold-index "${FOLD_INDEX}" \
  "${COHORT_ARGS[@]}" \
  "${RUN_NAME_ARGS[@]}"

echo "--- DNN env_terrain evaluate job end ---"

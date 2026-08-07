#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/dnn_noenv/run_dnn_noenv.sh [cohort] [max_epochs] [patience] [split_type] [seed] [run_name] [batch_size] [split_seed] [n_folds] [fold_index]
#
# Examples:
#   sbatch jobs/dnn_noenv/run_dnn_noenv.sh 4survey 5 3
#   sbatch jobs/dnn_noenv/run_dnn_noenv.sh 6survey 500 20
#   sbatch jobs/dnn_noenv/run_dnn_noenv.sh 4survey 500 20 spatial_block
#   sbatch jobs/dnn_noenv/run_dnn_noenv.sh 6survey 500 40 temporal 43 dnn_noenv_seed43
#   sbatch jobs/dnn_noenv/run_dnn_noenv.sh 4survey 500 40 spatial_block 42 dnn_noenv_bs256 256
#
# Arguments:
#   cohort      4survey or 6survey. Defaults to 4survey.
#   max_epochs  Maximum training epochs. Defaults to 5 for a quick test.
#   patience    Early-stopping patience. Defaults to 3 for a quick test.
#   split_type  temporal, temporal_narrow_gap, or spatial_block. Defaults to temporal.
#   seed        Random seed (network init + batch shuffling). Defaults to 42 (the
#               model's own default) -- pass a different value to test run-to-run
#               variance.
#   run_name    Only changes where results are saved -- set this whenever seed
#               isn't the default, so the run doesn't overwrite the primary
#               dnn_noenv checkpoint. Blank by default.
#   batch_size  Training batch size. Defaults to 512 (the model's own default) --
#               pass a different value as part of a batch-size sweep (see
#               documentation/experiment_log.md's 2026-07-29 entry for why this
#               is now being swept rather than assumed).
#
# Logs:
#   stdout -> logs/dnn_noenv/dnn_noenv_<jobid>.out
#   stderr -> logs/dnn_noenv/dnn_noenv_<jobid>.err
#
# Results:
#   outputs/<split_type>/dnn_noenv/<cohort>/

#SBATCH --job-name=dnn_noenv
#SBATCH --output=logs/dnn_noenv/%x_%j.out
#SBATCH --error=logs/dnn_noenv/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/dnn_noenv outputs

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
SPLIT_SEED=${8:-42}
N_FOLDS=${9:-5}
FOLD_INDEX=${10:-0}

echo "--- DNN job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Patience: ${PATIENCE}"
echo "Split type: ${SPLIT_TYPE}"
echo "Seed: ${SEED}"
echo "Run name: ${RUN_NAME:-(none, uses default dnn_noenv path)}"
echo "Batch size: ${BATCH_SIZE}"
echo "Split seed: ${SPLIT_SEED}"
echo "K-fold: ${FOLD_INDEX}/${N_FOLDS}"

RUN_NAME_ARGS=()
if [ -n "${RUN_NAME}" ]; then
  RUN_NAME_ARGS=(--run-name "${RUN_NAME}")
fi

python -u -m models.dnn_noenv.run_dnn_noenv \
  --cohort "${COHORT}" \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --split-type "${SPLIT_TYPE}" \
  --seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --split-seed "${SPLIT_SEED}" \
  --n-folds "${N_FOLDS}" \
  --fold-index "${FOLD_INDEX}" \
  "${RUN_NAME_ARGS[@]}"

echo "--- DNN job end ---"

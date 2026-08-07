#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/dnn_noenv/evaluate_dnn_noenv.sh [cohort] [split_type] [run_name] [split_seed] [n_folds] [fold_index]
#
# Examples:
#   sbatch jobs/dnn_noenv/evaluate_dnn_noenv.sh 4survey
#   sbatch jobs/dnn_noenv/evaluate_dnn_noenv.sh 6survey temporal
#   sbatch jobs/dnn_noenv/evaluate_dnn_noenv.sh 4survey spatial_block
#   sbatch jobs/dnn_noenv/evaluate_dnn_noenv.sh
#
# Arguments:
#   cohort      4survey or 6survey. Omit to evaluate both cohorts.
#   split_type  temporal, temporal_narrow_gap, or spatial_block. Defaults to temporal.
#
# Purpose:
#   Evaluates an already-trained dnn_noenv checkpoint on the test split.
#   Run jobs/dnn_noenv/run_dnn_noenv.sh first, with the SAME split_type.
#
# Logs:
#   stdout -> logs/dnn_noenv/evaluate_dnn_noenv_<jobid>.out
#   stderr -> logs/dnn_noenv/evaluate_dnn_noenv_<jobid>.err
#
# Results:
#   outputs/<split_type>/dnn_noenv/<cohort>/predictions.csv
#   outputs/<split_type>/dnn_noenv/<cohort>/metrics.json
#
# Notes:
#   This is a CPU job. It does not request a GPU.

#SBATCH -p Teaching
#SBATCH --job-name=evaluate_dnn_noenv
#SBATCH --output=logs/dnn_noenv/%x_%j.out
#SBATCH --error=logs/dnn_noenv/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G

cd ~/forest_diss

mkdir -p logs/dnn_noenv outputs

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-}
SPLIT_TYPE=${2:-temporal}
RUN_NAME=${3:-}
SPLIT_SEED=${4:-42}
N_FOLDS=${5:-5}
FOLD_INDEX=${6:-0}

echo "--- DNN evaluate job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT:-both}"
echo "Split type: ${SPLIT_TYPE}"
echo "K-fold: ${FOLD_INDEX}/${N_FOLDS}"

EXTRA_ARGS=()
if [ -n "${COHORT}" ]; then
  EXTRA_ARGS+=(--cohort "${COHORT}")
fi
if [ -n "${RUN_NAME}" ]; then
  EXTRA_ARGS+=(--run-name "${RUN_NAME}")
fi

python -u -m models.dnn_noenv.evaluate_dnn_noenv \
  --split-type "${SPLIT_TYPE}" --split-seed "${SPLIT_SEED}" \
  --n-folds "${N_FOLDS}" --fold-index "${FOLD_INDEX}" \
  "${EXTRA_ARGS[@]}"

echo "--- DNN evaluate job end ---"

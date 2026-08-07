#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/pinn_noenv/evaluate_pinn_noenv.sh [cohort] [split_type]
#
# Examples:
#   sbatch jobs/pinn_noenv/evaluate_pinn_noenv.sh 4survey
#   sbatch jobs/pinn_noenv/evaluate_pinn_noenv.sh 6survey temporal
#   sbatch jobs/pinn_noenv/evaluate_pinn_noenv.sh 4survey spatial_block
#   sbatch jobs/pinn_noenv/evaluate_pinn_noenv.sh
#
# Arguments:
#   cohort      4survey or 6survey. Omit to evaluate both cohorts.
#   split_type  temporal, temporal_narrow_gap, or spatial_block. Defaults to temporal.
#
# Purpose:
#   Evaluates an already-trained pinn_noenv checkpoint on the test split.
#   Run jobs/pinn_noenv/run_pinn_noenv.sh first, with the SAME split_type.
#
# Logs:
#   stdout -> logs/pinn_noenv/evaluate_pinn_noenv_<jobid>.out
#   stderr -> logs/pinn_noenv/evaluate_pinn_noenv_<jobid>.err
#
# Results:
#   outputs/<split_type>/pinn_noenv/<cohort>/predictions.csv
#   outputs/<split_type>/pinn_noenv/<cohort>/metrics.json
#
# Notes:
#   This is a CPU job. It does not request a GPU.

#SBATCH -p Teaching
#SBATCH --job-name=evaluate_pinn_noenv
#SBATCH --output=logs/pinn_noenv/%x_%j.out
#SBATCH --error=logs/pinn_noenv/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G

cd ~/forest_diss

mkdir -p logs/pinn_noenv outputs

# Toolchain lives in a shared TA home directory under a dated folder name that gets replaced
# periodically -- hardcoding one date breaks silently (a confusing torch/CUDA import error deep
# inside the Python job, not an obvious "toolchain missing" message) the next time it rotates.
# Finds whatever toolchain-* currently exists instead, picks the most recently modified one, and
# fails loudly with a clear message immediately if none exist at all.
TOOLCHAIN_RC=$(ls -1t /home/htang2/toolchain-*/toolchain.rc 2>/dev/null | head -1)
if [ -z "${TOOLCHAIN_RC}" ]; then
  echo "ERROR: no toolchain.rc found under /home/htang2/toolchain-*/ -- ask a TA if this has moved." >&2
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

echo "--- PINN evaluate job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT:-both}"
echo "Split type: ${SPLIT_TYPE}"
echo "Run name: ${RUN_NAME:-(none, uses default pinn_noenv path)}"
echo "Split seed: ${SPLIT_SEED}"
echo "K-fold: ${FOLD_INDEX}/${N_FOLDS}"

RUN_NAME_ARGS=()
if [ -n "${RUN_NAME}" ]; then
  RUN_NAME_ARGS=(--run-name "${RUN_NAME}")
fi

COHORT_ARGS=()
if [ -n "$COHORT" ]; then
  COHORT_ARGS=(--cohort "${COHORT}")
fi

python -u -m models.pinn_noenv.evaluate_pinn_noenv \
  --split-type "${SPLIT_TYPE}" \
  --split-seed "${SPLIT_SEED}" \
  --n-folds "${N_FOLDS}" \
  --fold-index "${FOLD_INDEX}" \
  "${COHORT_ARGS[@]}" \
  "${RUN_NAME_ARGS[@]}"

echo "--- PINN evaluate job end ---"

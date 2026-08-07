#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/baselines/evaluate_baselines.sh [split_type] [split_seed] [n_folds] [fold_index]
#
# Examples:
#   sbatch jobs/baselines/evaluate_baselines.sh plot_level
#   sbatch jobs/baselines/evaluate_baselines.sh temporal
#   sbatch jobs/baselines/evaluate_baselines.sh spatial_block
#
# Argument:
#   split_type  plot_level, temporal, temporal_narrow_gap, or spatial_block. Defaults to plot_level.
#
# Purpose:
#   Evaluates already-fitted baseline models for both cohorts.
#   Run jobs/baselines/run_baselines.sh with the same split_type first.
#
# Logs:
#   stdout -> logs/baselines/evaluate_baselines_<jobid>.out
#   stderr -> logs/baselines/evaluate_baselines_<jobid>.err
#
# Results:
#   Writes predictions.csv and metrics.json beside each fitted baseline output.
#
# Notes:
#   This is a CPU job. It does not request a GPU.

#SBATCH -p Teaching
#SBATCH --job-name=evaluate_baselines
#SBATCH --output=logs/baselines/%x_%j.out
#SBATCH --error=logs/baselines/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/baselines outputs

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

SPLIT_TYPE=${1:-plot_level}
SPLIT_SEED=${2:-42}
N_FOLDS=${3:-5}
FOLD_INDEX=${4:-0}

echo "--- Baseline evaluate job start ---"
echo "Node: $(hostname)"
echo "Split type: ${SPLIT_TYPE}"
echo "K-fold: ${FOLD_INDEX}/${N_FOLDS}"

python -u -m models.baselines.evaluate_baselines --split-type "${SPLIT_TYPE}" \
  --split-seed "${SPLIT_SEED}" --n-folds "${N_FOLDS}" --fold-index "${FOLD_INDEX}"

echo "--- Baseline evaluate job end ---"

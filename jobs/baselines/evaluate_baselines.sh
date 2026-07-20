#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/baselines/evaluate_baselines.sh [split_type]
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

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

SPLIT_TYPE=${1:-plot_level}

echo "--- Baseline evaluate job start ---"
echo "Node: $(hostname)"
echo "Split type: ${SPLIT_TYPE}"

python -u -m models.baselines.evaluate_baselines --split-type "${SPLIT_TYPE}"

echo "--- Baseline evaluate job end ---"

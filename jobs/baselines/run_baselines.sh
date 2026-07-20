#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/baselines/run_baselines.sh [split_type]
#
# Examples:
#   sbatch jobs/baselines/run_baselines.sh plot_level
#   sbatch jobs/baselines/run_baselines.sh temporal
#   sbatch jobs/baselines/run_baselines.sh spatial_block
#
# Argument:
#   split_type  plot_level, temporal, temporal_narrow_gap, or spatial_block. Defaults to plot_level.
#
# Purpose:
#   Fits all baseline models for both cohorts:
#   chapman_richards, average_by_age, linear_baseline, rf_baseline.
#
# Logs:
#   stdout -> logs/baselines/run_baselines_<jobid>.out
#   stderr -> logs/baselines/run_baselines_<jobid>.err
#
# Results:
#   plot_level    -> outputs/<model>/<cohort>/
#   temporal      -> outputs/temporal/<model>/<cohort>/
#   spatial_block -> outputs/spatial_block/<model>/<cohort>/
#
# Notes:
#   This is a CPU job. It does not request a GPU.
#   PINN needs outputs/chapman_richards/<cohort>/params.json, so run
#   plot_level baselines before plot_level PINN jobs.

#SBATCH -p Teaching
#SBATCH --job-name=run_baselines
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

echo "--- Baseline fit job start ---"
echo "Node: $(hostname)"
echo "Split type: ${SPLIT_TYPE}"

python -u -m models.baselines.run_baselines --split-type "${SPLIT_TYPE}"

echo "--- Baseline fit job end ---"

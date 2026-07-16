#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/dnn_noenv/evaluate_dnn_noenv.sh [cohort]
#
# Examples:
#   sbatch jobs/dnn_noenv/evaluate_dnn_noenv.sh 4survey
#   sbatch jobs/dnn_noenv/evaluate_dnn_noenv.sh 6survey
#   sbatch jobs/dnn_noenv/evaluate_dnn_noenv.sh
#
# Argument:
#   cohort  4survey or 6survey. Omit to evaluate both cohorts.
#
# Purpose:
#   Evaluates an already-trained dnn_noenv checkpoint on the test split.
#   Run jobs/dnn_noenv/run_dnn_noenv.sh first.
#
# Logs:
#   stdout -> logs/dnn_noenv/evaluate_dnn_noenv_<jobid>.out
#   stderr -> logs/dnn_noenv/evaluate_dnn_noenv_<jobid>.err
#
# Results:
#   outputs/dnn_noenv/<cohort>/predictions.csv
#   outputs/dnn_noenv/<cohort>/metrics.json
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

echo "--- DNN evaluate job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT:-both}"

if [ -n "$COHORT" ]; then
  python -u -m models.dnn_noenv.evaluate_dnn_noenv --cohort "${COHORT}"
else
  python -u -m models.dnn_noenv.evaluate_dnn_noenv
fi

echo "--- DNN evaluate job end ---"

#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/pinn_noenv/evaluate_pinn_noenv.sh [cohort]
#
# Examples:
#   sbatch jobs/pinn_noenv/evaluate_pinn_noenv.sh 4survey
#   sbatch jobs/pinn_noenv/evaluate_pinn_noenv.sh 6survey
#   sbatch jobs/pinn_noenv/evaluate_pinn_noenv.sh
#
# Argument:
#   cohort  4survey or 6survey. Omit to evaluate both cohorts.
#
# Purpose:
#   Evaluates an already-trained pinn_noenv checkpoint on the test split.
#   Run jobs/pinn_noenv/run_pinn_noenv.sh first.
#
# Logs:
#   stdout -> logs/pinn_noenv/evaluate_pinn_noenv_<jobid>.out
#   stderr -> logs/pinn_noenv/evaluate_pinn_noenv_<jobid>.err
#
# Results:
#   outputs/pinn_noenv/<cohort>/predictions.csv
#   outputs/pinn_noenv/<cohort>/metrics.json
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

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-}

echo "--- PINN evaluate job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT:-both}"

if [ -n "$COHORT" ]; then
  python -u -m models.pinn_noenv.evaluate_pinn_noenv --cohort "${COHORT}"
else
  python -u -m models.pinn_noenv.evaluate_pinn_noenv
fi

echo "--- PINN evaluate job end ---"

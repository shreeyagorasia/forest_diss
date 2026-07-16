#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/pinn_noenv/run_pinn_noenv.sh [cohort] [max_epochs] [patience] [split_type]
#
# Examples:
#   sbatch jobs/pinn_noenv/run_pinn_noenv.sh 4survey 5 3
#   sbatch jobs/pinn_noenv/run_pinn_noenv.sh 6survey 500 20
#   sbatch jobs/pinn_noenv/run_pinn_noenv.sh 4survey 500 20 spatial_block
#
# Arguments:
#   cohort      4survey or 6survey. Defaults to 4survey.
#   max_epochs  Maximum training epochs. Defaults to 5 for a quick test.
#   patience    Early-stopping patience. Defaults to 3 for a quick test.
#   split_type  temporal or spatial_block. Defaults to temporal.
#
# Logs:
#   stdout -> logs/pinn_noenv/pinn_noenv_<jobid>.out
#   stderr -> logs/pinn_noenv/pinn_noenv_<jobid>.err
#
# Results:
#   outputs/<split_type>/pinn_noenv/<cohort>/
#
# Prerequisite:
#   PINN reads frozen Chapman-Richards parameters from
#   outputs/chapman_richards/<cohort>/params.json (the plot_level fit,
#   always -- regardless of which split_type the PINN itself trains under,
#   see run_pinn_noenv.py's module docstring for why).
#   Run/transfer the baseline outputs before submitting this job.

#SBATCH --job-name=pinn_noenv
#SBATCH --output=logs/pinn_noenv/%x_%j.out
#SBATCH --error=logs/pinn_noenv/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/pinn_noenv outputs

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-4survey}
MAX_EPOCHS=${2:-5}
PATIENCE=${3:-3}
SPLIT_TYPE=${4:-temporal}

echo "--- PINN job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Patience: ${PATIENCE}"
echo "Split type: ${SPLIT_TYPE}"

python -u -m models.pinn_noenv.run_pinn_noenv \
  --cohort "${COHORT}" \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --split-type "${SPLIT_TYPE}"

echo "--- PINN job end ---"

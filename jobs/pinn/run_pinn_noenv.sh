#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/pinn/run_pinn_noenv.sh [cohort] [max_epochs] [patience]
#
# Examples:
#   sbatch jobs/pinn/run_pinn_noenv.sh 4survey 5 3
#   sbatch jobs/pinn/run_pinn_noenv.sh 6survey 500 20
#
# Arguments:
#   cohort      4survey or 6survey. Defaults to 4survey.
#   max_epochs  Maximum training epochs. Defaults to 5 for a quick test.
#   patience    Early-stopping patience. Defaults to 3 for a quick test.
#
# Logs:
#   stdout -> logs/pinn/pinn_noenv_<jobid>.out
#   stderr -> logs/pinn/pinn_noenv_<jobid>.err
#
# Results:
#   outputs/pinn_noenv/<cohort>/
#
# Prerequisite:
#   PINN reads frozen Chapman-Richards parameters from
#   outputs/chapman_richards/<cohort>/params.json.
#   Run/transfer the baseline outputs before submitting this job.

#SBATCH --job-name=pinn_noenv
#SBATCH --output=logs/pinn/%x_%j.out
#SBATCH --error=logs/pinn/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=16G

cd ~/forest_diss

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-4survey}
MAX_EPOCHS=${2:-5}
PATIENCE=${3:-3}

echo "--- PINN job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Patience: ${PATIENCE}"

python -u -m models.pinn_noenv.run_pinn_noenv \
  --cohort "${COHORT}" \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}"

echo "--- PINN job end ---"

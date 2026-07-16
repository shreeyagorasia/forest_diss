#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/dnn_noenv/run_dnn_noenv.sh [cohort] [max_epochs] [patience]
#
# Examples:
#   sbatch jobs/dnn_noenv/run_dnn_noenv.sh 4survey 5 3
#   sbatch jobs/dnn_noenv/run_dnn_noenv.sh 6survey 500 20
#
# Arguments:
#   cohort      4survey or 6survey. Defaults to 4survey.
#   max_epochs  Maximum training epochs. Defaults to 5 for a quick test.
#   patience    Early-stopping patience. Defaults to 3 for a quick test.
#
# Logs:
#   stdout -> logs/dnn_noenv/dnn_noenv_<jobid>.out
#   stderr -> logs/dnn_noenv/dnn_noenv_<jobid>.err
#
# Results:
#   outputs/dnn_noenv/<cohort>/

#SBATCH --job-name=dnn_noenv
#SBATCH --output=logs/dnn_noenv/%x_%j.out
#SBATCH --error=logs/dnn_noenv/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/dnn_noenv outputs

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-4survey}
MAX_EPOCHS=${2:-5}
PATIENCE=${3:-3}

echo "--- DNN job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Patience: ${PATIENCE}"

python -u -m models.dnn_noenv.run_dnn_noenv \
  --cohort "${COHORT}" \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}"

echo "--- DNN job end ---"

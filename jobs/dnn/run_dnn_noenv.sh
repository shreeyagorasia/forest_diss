#!/bin/bash
#SBATCH --job-name=dnn_noenv
#SBATCH --output=logs/dnn/%x_%j.out
#SBATCH --error=logs/dnn/%x_%j.err
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

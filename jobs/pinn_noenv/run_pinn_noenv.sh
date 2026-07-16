#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/pinn_noenv/run_pinn_noenv.sh [cohort] [max_epochs] [patience] [split_type] [physics_weight] [trajectory_weight] [run_name]
#
# Examples:
#   sbatch jobs/pinn_noenv/run_pinn_noenv.sh 4survey 5 3
#   sbatch jobs/pinn_noenv/run_pinn_noenv.sh 6survey 500 20
#   sbatch jobs/pinn_noenv/run_pinn_noenv.sh 4survey 500 20 spatial_block
#   sbatch jobs/pinn_noenv/run_pinn_noenv.sh 4survey 500 40 spatial_block 5.0 5.0 pinn_noenv_pw5_tw5
#
# Arguments:
#   cohort             4survey or 6survey. Defaults to 4survey.
#   max_epochs         Maximum training epochs. Defaults to 5 for a quick test.
#   patience           Early-stopping patience. Defaults to 3 for a quick test.
#   split_type         temporal or spatial_block. Defaults to temporal.
#   physics_weight     Weight on the physics loss term. Defaults to 1.0 (the model's own default).
#   trajectory_weight  Weight on the trajectory loss term. Defaults to 1.0 (the model's own default).
#   run_name           Only changes where results are saved -- set this whenever
#                       physics_weight/trajectory_weight aren't both 1.0, so the
#                       run doesn't overwrite the primary pinn_noenv result. Blank by default.
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
PHYSICS_WEIGHT=${5:-1.0}
TRAJECTORY_WEIGHT=${6:-1.0}
RUN_NAME=${7:-}

echo "--- PINN job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Patience: ${PATIENCE}"
echo "Split type: ${SPLIT_TYPE}"
echo "Physics weight: ${PHYSICS_WEIGHT}"
echo "Trajectory weight: ${TRAJECTORY_WEIGHT}"
echo "Run name: ${RUN_NAME:-(none, uses default pinn_noenv path)}"

RUN_NAME_ARGS=()
if [ -n "${RUN_NAME}" ]; then
  RUN_NAME_ARGS=(--run-name "${RUN_NAME}")
fi

python -u -m models.pinn_noenv.run_pinn_noenv \
  --cohort "${COHORT}" \
  --max-epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --split-type "${SPLIT_TYPE}" \
  --physics-weight "${PHYSICS_WEIGHT}" \
  --trajectory-weight "${TRAJECTORY_WEIGHT}" \
  "${RUN_NAME_ARGS[@]}"

echo "--- PINN job end ---"

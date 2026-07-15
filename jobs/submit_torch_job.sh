#!/bin/bash

# Generic torch model submitter for DNN/PINN jobs.
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch --job-name=<name> jobs/submit_torch_job.sh <model_name> <cohort> [max_epochs] [patience]
#
# Examples:
#   sbatch --job-name=dnn_test_4survey  jobs/submit_torch_job.sh dnn_noenv  4survey 5 3
#   sbatch --job-name=pinn_full_6survey jobs/submit_torch_job.sh pinn_noenv 6survey 500 20
#
# Arguments:
#   model_name  dnn_noenv or pinn_noenv.
#   cohort      4survey or 6survey.
#   max_epochs  Optional. If omitted, Python uses its default of 500.
#   patience    Optional. If omitted, Python uses its default of 20.
#
# Logs:
#   stdout -> logs/torch/<job_name>_<jobid>.out
#   stderr -> logs/torch/<job_name>_<jobid>.err
#
# Results:
#   outputs/<model_name>/<cohort>/

#SBATCH --job-name=growth_model_torch
#SBATCH --output=logs/torch/%x_%j.out
#SBATCH --error=logs/torch/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=16G

# --- How to run ---------------------------------------------------
# Submit from the PROJECT ROOT (the folder this script's own relative paths,
# like jobs/... and logs/..., are written against):
#
#   sbatch --job-name=dnn_noenv_4survey  jobs/submit_torch_job.sh dnn_noenv  4survey
#   sbatch --job-name=pinn_noenv_6survey jobs/submit_torch_job.sh pinn_noenv 6survey 500 20
#
# Positional args: <model_name> <cohort> [max_epochs] [patience]
#   model_name : dnn_noenv or pinn_noenv (each has its own models/<model_name>/run_<model_name>.py)
#   cohort     : 4survey or 6survey
#   max_epochs : optional, defaults to the run script's own default (500)
#   patience   : optional, defaults to the run script's own default (20)
#
# This is for dnn_noenv/pinn_noenv only -- the four sklearn/scipy baselines
# (chapman_richards, average_by_age, linear_baseline, rf_baseline) fit in
# seconds on a laptop and don't need a GPU or a cluster job at all; run
# those locally with `python -m models.baselines.run_baselines`.

# --- environment setup (adjust paths to match your cluster account) ------
# 1. Load the ICF cluster's shared GPU toolchain (gives access to nvcc/CUDA
#    -- this is the ONE shared toolchain path everyone on the cluster
#    activates, per the cluster tutorial; it is not specific to this
#    project, so leave it as-is unless the cluster's toolchain path changes).
. /home/htang2/toolchain-20251006/toolchain.rc

# 2. Activate this project's venv. SLURM starts the job in the directory
#    you submitted `sbatch` from, so this assumes a .venv/ folder sits at
#    the project root there -- rename this line if your cluster venv has a
#    different name or location.
source ./.venv/bin/activate

# --- adapting this to how THIS project's code is actually laid out -------
# The original template this script is based on assumed a top-level
# common/ folder and a per-model train.py, invoked by `cd`-ing into
# models/<model_name>/ and running the script directly. This project is
# laid out differently: shared code lives under models/common/ (not a
# top-level common/), and every model is its own Python submodule
# (models.dnn_noenv, models.pinn_noenv) with internal imports like
# `from models.common.torch_data import ...` -- those only resolve
# correctly when Python is run as `python -m models.<x>.<y>` FROM THE
# PROJECT ROOT, with the project root on PYTHONPATH. So, unlike the
# original template: never `cd` into models/<model_name> before running
# anything, and PYTHONPATH must point at the project root, not a
# subfolder.
export PYTHONPATH="$(pwd)"

echo "--- SLURM JOB START ---"
echo "Node: $(hostname)"
echo "Current directory is: $(pwd)"

MODEL_NAME=$1
COHORT=$2
MAX_EPOCHS=$3
PATIENCE=$4

if [ -z "$MODEL_NAME" ] || [ -z "$COHORT" ]; then
    echo "Usage: sbatch jobs/submit_torch_job.sh <model_name> <cohort> [max_epochs] [patience]"
    echo "e.g.:  sbatch jobs/submit_torch_job.sh dnn_noenv 4survey"
    echo "e.g.:  sbatch jobs/submit_torch_job.sh pinn_noenv 6survey 500 20"
    exit 1
fi

if [ ! -d "models/${MODEL_NAME}" ]; then
    echo "Error: Directory models/${MODEL_NAME} does not exist."
    exit 1
fi

mkdir -p logs/torch outputs

# max_epochs/patience are only appended to the command if actually given --
# omitting them lets run_<model_name>.py fall back to its own defaults
# (500 / 20) rather than this script silently overriding them with the
# same numbers spelled out twice in two places.
EXTRA_ARGS=""
if [ -n "$MAX_EPOCHS" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --max-epochs $MAX_EPOCHS"
fi
if [ -n "$PATIENCE" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --patience $PATIENCE"
fi

echo "Model requested: ${MODEL_NAME}"
echo "Cohort requested: ${COHORT}"
echo "Extra args: ${EXTRA_ARGS}"
echo "Running: python -u -m models.${MODEL_NAME}.run_${MODEL_NAME} --cohort ${COHORT} ${EXTRA_ARGS}"

python -u -m "models.${MODEL_NAME}.run_${MODEL_NAME}" --cohort "${COHORT}" ${EXTRA_ARGS}

echo "--- SLURM JOB END ---"

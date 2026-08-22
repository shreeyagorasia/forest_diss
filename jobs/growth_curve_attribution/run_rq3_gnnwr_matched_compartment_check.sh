#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_matched_compartment_check.sh [max_epoch] [early_stop] [reference_set_size] [split_seed] [held_out_fold] [k_folds]
#
# Tests whether GNNWR's unexplained 6-survey collapse is caused by having FEW COMPARTMENTS (47),
# not just fewer points -- restricts 4-survey (231 compartments) to a random 47-compartment subset
# at FULL point density, matching 6-survey's own compartment count exactly. Same GPU/resource
# footprint as the other GNNWR ablation jobs. Always Set4/4survey -- see the Python script's own
# header for why.
#
# Submit once per fold for the full 5-fold spatial CV (run all 5 simultaneously):
#
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_matched_compartment_check.sh 200 20 0 42 0 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_matched_compartment_check.sh 200 20 0 42 1 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_matched_compartment_check.sh 200 20 0 42 2 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_matched_compartment_check.sh 200 20 0 42 3 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_matched_compartment_check.sh 200 20 0 42 4 5

#SBATCH --job-name=rq3_gnnwr_matched_cpmt
#SBATCH --output=logs/rq123_methodology/%x_%j.out
#SBATCH --error=logs/rq123_methodology/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G

cd ~/forest_diss

mkdir -p logs/rq123_methodology outputs/growth_curve_attribution/gnnwr

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

MAX_EPOCH=${1:-200}
EARLY_STOP=${2:-20}
REFERENCE_SET_SIZE=${3:-0}
SPLIT_SEED=${4:-42}
HELD_OUT_FOLD=${5:-}
K_FOLDS=${6:-5}

echo "--- RQ3 GNNWR matched-compartment-count job start ---"
echo "Node: $(hostname)"
echo "Held-out fold: ${HELD_OUT_FOLD:-(none -- single split)}"

FOLD_ARGS=()
if [ -n "${HELD_OUT_FOLD}" ]; then
  FOLD_ARGS=(--held-out-fold "${HELD_OUT_FOLD}" --k-folds "${K_FOLDS}")
fi

python -u -m models.growth_curve_attribution.run_rq3_gnnwr_matched_compartment_check \
  --max-epoch "${MAX_EPOCH}" \
  --early-stop "${EARLY_STOP}" \
  --reference-set-size "${REFERENCE_SET_SIZE}" \
  --split-seed "${SPLIT_SEED}" \
  --use-gpu \
  "${FOLD_ARGS[@]}"

echo "--- RQ3 GNNWR matched-compartment-count job end ---"

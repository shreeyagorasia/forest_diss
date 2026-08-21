#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_canopycover_ablation.sh [cohort] [max_epoch] [early_stop] [reference_set_size] [split_seed] [held_out_fold] [k_folds]
#
# CanopyCover-dropped ablation of Q2's own GNNWR model (Set4). Same resource footprint and GPU
# as jobs/growth_curve_attribution/run_rq3_gnnwr.sh (confirmed working on this cluster, 2026-08-20
# GRES fix) -- only the Python entry point differs (run_rq3_gnnwr_canopycover_ablation instead of
# run_rq3_gnnwr), which drops CanopyCover from Set4's column list before fitting and saves to a
# separate output path (scope label "nested_set4_no_canopycover") so it cannot collide with or
# overwrite the real production Set4 GNNWR results.
#
# Submit once per fold for the full 5-fold spatial CV (can run all 5 simultaneously as separate
# jobs, per this project's own established convention -- see run_rq3_gnnwr.sh's own examples):
#
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_canopycover_ablation.sh 4survey 200 20 0 42 0 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_canopycover_ablation.sh 4survey 200 20 0 42 1 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_canopycover_ablation.sh 4survey 200 20 0 42 2 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_canopycover_ablation.sh 4survey 200 20 0 42 3 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_canopycover_ablation.sh 4survey 200 20 0 42 4 5
#
# reference_set_size defaults to 0 (full/uncapped, ~31,000 plots) -- matches every other headline
# GNNWR result in this project, and capping does not save wall-clock time at this scale (checked
# directly against this project's own cluster timing).

#SBATCH --job-name=rq3_gnnwr_no_canopy
#SBATCH --output=logs/rq123_methodology/%x_%j.out
#SBATCH --error=logs/rq123_methodology/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=32G
# Same GRES as run_rq3_gnnwr.sh's own 2026-08-20 fix -- landonia11 (idle, 8x RTX A6000, 48GB
# each), confirmed correctly named and free, far more headroom than the ~12.9GB the full/uncapped
# reference-population run needs. Do not revert to an h200 MIG-slice request -- that name does
# not match this cluster's actual registered GRES string and will sit unschedulable.

cd ~/forest_diss

mkdir -p logs/rq123_methodology outputs/growth_curve_attribution/gnnwr

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-4survey}
MAX_EPOCH=${2:-200}
EARLY_STOP=${3:-20}
REFERENCE_SET_SIZE=${4:-0}
SPLIT_SEED=${5:-42}
HELD_OUT_FOLD=${6:-}
K_FOLDS=${7:-5}

echo "--- RQ3 GNNWR CanopyCover-ablation job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Max epoch: ${MAX_EPOCH}"
echo "Early stop patience: ${EARLY_STOP}"
echo "Reference set size: ${REFERENCE_SET_SIZE}"
echo "Split seed: ${SPLIT_SEED}"
echo "Held-out fold: ${HELD_OUT_FOLD:-(none -- single split)}"

FOLD_ARGS=()
if [ -n "${HELD_OUT_FOLD}" ]; then
  FOLD_ARGS=(--held-out-fold "${HELD_OUT_FOLD}" --k-folds "${K_FOLDS}")
fi

python -u -m models.growth_curve_attribution.run_rq3_gnnwr_canopycover_ablation \
  --cohort "${COHORT}" \
  --max-epoch "${MAX_EPOCH}" \
  --early-stop "${EARLY_STOP}" \
  --reference-set-size "${REFERENCE_SET_SIZE}" \
  --split-seed "${SPLIT_SEED}" \
  --use-gpu \
  "${FOLD_ARGS[@]}"

echo "--- RQ3 GNNWR CanopyCover-ablation job end ---"

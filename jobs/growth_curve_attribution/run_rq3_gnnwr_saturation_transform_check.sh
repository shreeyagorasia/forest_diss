#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_saturation_transform_check.sh [cohort] [max_epoch] [early_stop] [reference_set_size] [split_seed] [held_out_fold] [k_folds]
#
# IMPORTANT: run this once, locally or on the cluster's login node, before submitting any of the
# jobs below -- it builds the transformed-features parquet the job script reads:
#
#   PYTHONPATH=. .venv/bin/python -m models.growth_curve_attribution.build_saturation_transformed_features
#
# Tests whether GNNWR does better on Q2's target when slope_degrees/windward_topex are pre-
# saturated (capped/clipped) before fitting, instead of raw -- GNNWR is structurally local-linear
# and cannot represent the saturating curves the no-CanopyCover SHAP-dependence check found for
# both variables. Same resource footprint/GPU as every other GNNWR job in this project.
#
# Submit once per fold for the full 5-fold spatial CV (run all 5 simultaneously):
#
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_saturation_transform_check.sh 4survey 200 20 0 42 0 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_saturation_transform_check.sh 4survey 200 20 0 42 1 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_saturation_transform_check.sh 4survey 200 20 0 42 2 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_saturation_transform_check.sh 4survey 200 20 0 42 3 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr_saturation_transform_check.sh 4survey 200 20 0 42 4 5

#SBATCH --job-name=rq3_gnnwr_sat_transform
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

COHORT=${1:-4survey}
MAX_EPOCH=${2:-200}
EARLY_STOP=${3:-20}
REFERENCE_SET_SIZE=${4:-0}
SPLIT_SEED=${5:-42}
HELD_OUT_FOLD=${6:-}
K_FOLDS=${7:-5}

echo "--- RQ3 GNNWR saturation-transform check job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Held-out fold: ${HELD_OUT_FOLD:-(none -- single split)}"

if [ ! -f "data/processed/environmental/plot_environmental_features_saturation_transformed.parquet" ]; then
  echo "ERROR: transformed-features parquet not found. Run this first:"
  echo "  python -m models.growth_curve_attribution.build_saturation_transformed_features"
  exit 1
fi

FOLD_ARGS=()
if [ -n "${HELD_OUT_FOLD}" ]; then
  FOLD_ARGS=(--held-out-fold "${HELD_OUT_FOLD}" --k-folds "${K_FOLDS}")
fi

python -u -m models.growth_curve_attribution.run_rq3_gnnwr_saturation_transform_check \
  --cohort "${COHORT}" \
  --max-epoch "${MAX_EPOCH}" \
  --early-stop "${EARLY_STOP}" \
  --reference-set-size "${REFERENCE_SET_SIZE}" \
  --split-seed "${SPLIT_SEED}" \
  --use-gpu \
  "${FOLD_ARGS[@]}"

echo "--- RQ3 GNNWR saturation-transform check job end ---"

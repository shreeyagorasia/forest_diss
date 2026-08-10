#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr.sh [cohort] [set_name] [max_epoch] [early_stop] [reference_set_size] [split_seed] [held_out_fold] [k_folds]
#
# New-methodology sibling of run_gnnwr.sh -- identical resource footprint and training loop,
# just calls models.growth_curve_attribution.run_rq3_gnnwr (raw Set2-5 columns from
# documentation/env_feature_sets_manifest.csv) instead of gnnwr_check.py's --scope CLI directly.
#
# Examples:
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr.sh 4survey nested_set2_top5
#   # 5-fold spatial CV (submit once per fold):
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr.sh 4survey nested_set2_top5 200 20 0 42 0 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr.sh 4survey nested_set2_top5 200 20 0 42 1 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr.sh 4survey nested_set2_top5 200 20 0 42 2 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr.sh 4survey nested_set2_top5 200 20 0 42 3 5
#   sbatch jobs/growth_curve_attribution/run_rq3_gnnwr.sh 4survey nested_set2_top5 200 20 0 42 4 5
#
# Arguments: same meaning as run_gnnwr.sh's own (see that file's header comment for the full
# reasoning behind each default) -- set_name replaces scope, one of nested_set2_top5 /
# nested_set3_gated_terrain_wind / nested_set4_gated_all / nested_set5_all_ungated.

#SBATCH --job-name=rq3_gnnwr
#SBATCH --output=logs/growth_curve_attribution/%x_%j.out
#SBATCH --error=logs/growth_curve_attribution/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:h200_1g.18gb:1
#SBATCH --mem=32G

cd ~/forest_diss

mkdir -p logs/growth_curve_attribution outputs/growth_curve_attribution/gnnwr

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-4survey}
SET_NAME=${2:-nested_set2_top5}
MAX_EPOCH=${3:-200}
EARLY_STOP=${4:-20}
REFERENCE_SET_SIZE=${5:-0}
SPLIT_SEED=${6:-42}
HELD_OUT_FOLD=${7:-}
K_FOLDS=${8:-5}

echo "--- RQ3 GNNWR job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Set name: ${SET_NAME}"
echo "Max epoch: ${MAX_EPOCH}"
echo "Early stop patience: ${EARLY_STOP}"
echo "Reference set size: ${REFERENCE_SET_SIZE}"
echo "Split seed: ${SPLIT_SEED}"
echo "Held-out fold: ${HELD_OUT_FOLD:-(none -- single split)}"

FOLD_ARGS=()
if [ -n "${HELD_OUT_FOLD}" ]; then
  FOLD_ARGS=(--held-out-fold "${HELD_OUT_FOLD}" --k-folds "${K_FOLDS}")
fi

python -u -m models.growth_curve_attribution.run_rq3_gnnwr \
  --cohort "${COHORT}" \
  --set-name "${SET_NAME}" \
  --max-epoch "${MAX_EPOCH}" \
  --early-stop "${EARLY_STOP}" \
  --reference-set-size "${REFERENCE_SET_SIZE}" \
  --split-seed "${SPLIT_SEED}" \
  --use-gpu \
  "${FOLD_ARGS[@]}"

echo "--- RQ3 GNNWR job end ---"

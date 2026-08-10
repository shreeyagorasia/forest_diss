#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/spatial_attribution/run_rq2_attribution.sh [cohort] [set_name] [split_type] [fold_index] [n_folds] [split_seed]
#
# CPU-only job (no GPU needed -- NLME/Elastic Net/XGBoost all fit in seconds). Runs on the
# cluster anyway rather than locally, per this project's "most fits run on the cluster,
# evaluation stays local" convention -- consistency over marginal speed here.
#
# Examples:
#   sbatch jobs/spatial_attribution/run_rq2_attribution.sh 4survey nested_set2_top5 spatial_block_kfold 0
#   sbatch jobs/spatial_attribution/run_rq2_attribution.sh 4survey nested_set2_top5 spatial_block_kfold 1
#   sbatch jobs/spatial_attribution/run_rq2_attribution.sh 4survey nested_set2_top5 spatial_block_kfold 2
#   sbatch jobs/spatial_attribution/run_rq2_attribution.sh 4survey nested_set2_top5 spatial_block_kfold 3
#   sbatch jobs/spatial_attribution/run_rq2_attribution.sh 4survey nested_set2_top5 spatial_block_kfold 4
#
# Arguments:
#   cohort       4survey or 6survey. RQ2 is 4survey only by design (see
#                documentation/experiment_log.md's cohort-justification entry) -- defaults to
#                4survey but isn't hardcoded, in case that decision ever needs revisiting.
#   set_name     nested_set2_top5 / nested_set3_gated_terrain_wind_vif /
#                nested_set4_gated_all_vif / nested_set5_all_ungated_vif.
#   split_type   spatial_block (single split) or spatial_block_kfold (one fold of 5).
#   fold_index   Which fold to hold out, 0..n_folds-1. Only used for spatial_block_kfold.
#   n_folds      Number of folds. Defaults to 5.
#   split_seed   Defaults to 42.

#SBATCH -p Teaching
#SBATCH --job-name=rq2_attribution
#SBATCH --output=logs/rq123_methodology/%x_%j.out
#SBATCH --error=logs/rq123_methodology/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/rq123_methodology outputs

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-4survey}
SET_NAME=${2:-nested_set2_top5}
SPLIT_TYPE=${3:-spatial_block_kfold}
FOLD_INDEX=${4:-0}
N_FOLDS=${5:-5}
SPLIT_SEED=${6:-42}

echo "--- RQ2 attribution job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Set name: ${SET_NAME}"
echo "Split type: ${SPLIT_TYPE}"
echo "Fold index: ${FOLD_INDEX}"

python -u -m models.spatial_attribution.run_rq2_attribution \
  --cohort "${COHORT}" \
  --set-name "${SET_NAME}" \
  --split-type "${SPLIT_TYPE}" \
  --fold-index "${FOLD_INDEX}" \
  --n-folds "${N_FOLDS}" \
  --split-seed "${SPLIT_SEED}"

echo "--- RQ2 attribution job end ---"

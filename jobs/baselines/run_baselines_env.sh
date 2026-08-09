#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/baselines/run_baselines_env.sh [cohort] [split_type] [feature_set] [split_seed] [n_folds] [fold_index]
#
# Examples:
#   sbatch jobs/baselines/run_baselines_env.sh 4survey plot_level stage2_terrain_wind
#   sbatch jobs/baselines/run_baselines_env.sh 4survey spatial_block_kfold stage2_terrain_wind 42 5 0
#
# Arguments:
#   cohort       4survey or 6survey. Required.
#   split_type   plot_level or spatial_block_kfold only. Required.
#   feature_set  stage1_terrain, stage2_terrain_wind, or stage4_all_environmental. Required.
#   split_seed   Default 42.
#   n_folds      Default 5. Only used for spatial_block_kfold.
#   fold_index   Default 0. Only used for spatial_block_kfold -- run once per fold (0..n_folds-1).
#
# Purpose:
#   Fits linear_baseline_env, rf_baseline_env, xgb_baseline_env -- the same three baselines,
#   given the same terrain/wind/environmental features the neural models see (feature_set), to
#   check whether dnn_env_terrain/pinn_env_terrain_k's accuracy is actually beating a simple
#   model given equal information. Standalone script (models/baselines/run_baselines_env.py),
#   deliberately separate from run_baselines.py -- does not touch or affect the already-cited
#   plain baseline numbers.
#
# Logs:
#   stdout -> logs/baselines/run_baselines_env_<jobid>.out
#   stderr -> logs/baselines/run_baselines_env_<jobid>.err
#
# Results:
#   outputs/<split_type>/{linear,rf,xgb}_baseline_env_<feature_set>[_fold<i>]/<cohort>/metrics.json
#
# Notes:
#   This is a CPU job. It does not request a GPU -- sklearn/XGBoost only, same as the plain
#   baselines. XGBoost is forced to n_jobs=1 in code (a macOS ARM64 segfault found 2026-08-09,
#   irrelevant on the cluster's Linux/CUDA nodes, but left as-is for consistency).

#SBATCH -p Teaching
#SBATCH --job-name=run_baselines_env
#SBATCH --output=logs/baselines/%x_%j.out
#SBATCH --error=logs/baselines/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/baselines outputs

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:?cohort is required (4survey or 6survey)}
SPLIT_TYPE=${2:?split_type is required (plot_level or spatial_block_kfold)}
FEATURE_SET=${3:?feature_set is required (stage1_terrain, stage2_terrain_wind, or stage4_all_environmental)}
SPLIT_SEED=${4:-42}
N_FOLDS=${5:-5}
FOLD_INDEX=${6:-0}

echo "--- Baseline-with-environment fit job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Split type: ${SPLIT_TYPE}"
echo "Feature set: ${FEATURE_SET}"
echo "Split seed: ${SPLIT_SEED}"
echo "K-fold: ${FOLD_INDEX}/${N_FOLDS}"

python -u -m models.baselines.run_baselines_env \
  --cohort "${COHORT}" --split-type "${SPLIT_TYPE}" --feature-set "${FEATURE_SET}" \
  --split-seed "${SPLIT_SEED}" --n-folds "${N_FOLDS}" --fold-index "${FOLD_INDEX}"

echo "--- Baseline-with-environment fit job end ---"

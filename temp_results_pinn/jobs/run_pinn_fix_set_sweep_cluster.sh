#!/bin/bash
#
# Set2/Set4 feature-set sweep for the CORRECTED PINN/PINN-k -- same GPU/toolchain convention as
# the already-working temp_results_pinn/jobs/run_pinn_fix_cluster.sh (which produced the real
# Table 3 Set3 numbers), pointed at run_cluster_fold_set_sweep.py instead, which writes to a
# separate output directory (CORRECTED_2026-08-22_pinn_set_sweep/) so it can never collide with or
# overwrite the existing Set3 results.
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#
#   # One fold, one variant, one set:
#   sbatch temp_results_pinn/jobs/run_pinn_fix_set_sweep_cluster.sh 0 ymax nested_set2_top10
#
#   # All 5 folds at once for one variant+set (job array):
#   sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_set_sweep_cluster.sh "" ymax nested_set2_top10
#
#   # Full sweep -- 2 sets x 2 variants x 5 folds = 20 jobs:
#   sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_set_sweep_cluster.sh "" ymax nested_set2_top10
#   sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_set_sweep_cluster.sh "" k    nested_set2_top10
#   sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_set_sweep_cluster.sh "" ymax nested_set4_gated_all_vif
#   sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_set_sweep_cluster.sh "" k    nested_set4_gated_all_vif
#
# Arguments:
#   fold_index    0-4. Leave blank ("") when submitting with --array so each task picks up its
#                 own $SLURM_ARRAY_TASK_ID automatically.
#   variant       ymax (y_max-only fix) or k (y_max+k fix). Required.
#   feature_set   nested_set2_top10 or nested_set4_gated_all_vif. Required (Set3 already exists).
#
# Logs:
#   stdout -> logs/temp_results_pinn/%x_%A_%a.out
#   stderr -> logs/temp_results_pinn/%x_%A_%a.err
#
# Results:
#   temp_results_pinn/outputs/CORRECTED_2026-08-22_pinn_set_sweep/<feature_set>/fold_<i>/pinn_<variant>_fixed_summary.json

#SBATCH --job-name=pinn_fix_sweep
#SBATCH --output=logs/temp_results_pinn/%x_%A_%a.out
#SBATCH --error=logs/temp_results_pinn/%x_%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/temp_results_pinn

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

FOLD_INDEX=${1:-${SLURM_ARRAY_TASK_ID:-0}}
VARIANT=${2:?"variant is required: ymax or k"}
FEATURE_SET=${3:?"feature_set is required: nested_set2_top10 or nested_set4_gated_all_vif"}

echo "--- PINN fix set-sweep (temp_results_pinn) job start ---"
echo "Node: $(hostname)"
echo "Fold index: ${FOLD_INDEX}"
echo "Variant: ${VARIANT}"
echo "Feature set: ${FEATURE_SET}"

python -u temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_set_sweep.py \
  --fold-index "${FOLD_INDEX}" \
  --variant "${VARIANT}" \
  --feature-set "${FEATURE_SET}"

echo "--- PINN fix set-sweep (temp_results_pinn) job end ---"

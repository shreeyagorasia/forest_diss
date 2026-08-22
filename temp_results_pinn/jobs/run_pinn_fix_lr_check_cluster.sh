#!/bin/bash
#
# Tests whether the DNN's Aug-19 hyperparameter finding (learning_rate=0.001, weight_decay=1e-3)
# transfers to the CORRECTED PINN/PINN-k. Set3 only, 5-fold CV, lambda left at its current
# default (1.0). See temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_lr_check.py for why.
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#
#   # One fold, one variant:
#   sbatch temp_results_pinn/jobs/run_pinn_fix_lr_check_cluster.sh 0 k
#
#   # All 5 folds for one variant (job array):
#   sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_lr_check_cluster.sh "" k
#
#   # Both variants, all folds:
#   sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_lr_check_cluster.sh "" ymax
#   sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_lr_check_cluster.sh "" k
#
# Arguments:
#   fold_index  0-4. Leave blank ("") when submitting with --array.
#   variant     ymax or k. Required.
#
# Results:
#   temp_results_pinn/outputs/CORRECTED_2026-08-22_lr_check/<variant>/fold_<i>/pinn_<variant>_lr_check_summary.json

#SBATCH --job-name=pinn_lr_check
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

echo "--- PINN LR-check (temp_results_pinn) job start ---"
echo "Node: $(hostname)"
echo "Fold index: ${FOLD_INDEX}"
echo "Variant: ${VARIANT}"

python -u temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_lr_check.py \
  --fold-index "${FOLD_INDEX}" \
  --variant "${VARIANT}"

echo "--- PINN LR-check (temp_results_pinn) job end ---"

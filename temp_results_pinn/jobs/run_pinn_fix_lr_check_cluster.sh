#!/bin/bash
#
# Runs the CORRECTED PINN/PINN-k on Set3, 5-fold CV, with learning_rate/weight_decay/batch_size
# all overridable -- see temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_lr_check.py for
# the full history (originally tested the Aug-19-sweep-derived config, found flat-to-worse at
# batch_size=256; now also used for the batch_size=512 check).
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#
#   # One fold, one variant, project defaults (lr=0.0001, weight_decay=1e-5, batch_size=256):
#   sbatch temp_results_pinn/jobs/run_pinn_fix_lr_check_cluster.sh 0 k
#
#   # All 5 folds, project defaults:
#   sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_lr_check_cluster.sh "" k
#
#   # All 5 folds at batch_size=512 (writes to a different output dir automatically):
#   sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_lr_check_cluster.sh "" k 512 CORRECTED_2026-08-22_pinn_bs512_check
#
# Arguments:
#   fold_index       0-4. Leave blank ("") when submitting with --array.
#   variant           ymax or k. Required.
#   batch_size        Defaults to 256 (PINN's own default).
#   output_dir_name   Which outputs/<name>/ to write to. Defaults to
#                      CORRECTED_2026-08-22_lr_check. CHANGE THIS whenever batch_size differs
#                      from a previous run, so results never collide.
#
# Results:
#   temp_results_pinn/outputs/<output_dir_name>/<variant>/fold_<i>/pinn_<variant>_lr_check_summary.json

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
BATCH_SIZE=${3:-256}
OUTPUT_DIR_NAME=${4:-CORRECTED_2026-08-22_lr_check}

echo "--- PINN cluster job start ---"
echo "Node: $(hostname)"
echo "Fold index: ${FOLD_INDEX}"
echo "Variant: ${VARIANT}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output dir name: ${OUTPUT_DIR_NAME}"

python -u temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_lr_check.py \
  --fold-index "${FOLD_INDEX}" \
  --variant "${VARIANT}" \
  --batch-size "${BATCH_SIZE}" \
  --output-dir-name "${OUTPUT_DIR_NAME}"

echo "--- PINN cluster job end ---"

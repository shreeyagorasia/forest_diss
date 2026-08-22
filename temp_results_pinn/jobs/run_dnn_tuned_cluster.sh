#!/bin/bash
#
# Runs the DNN on Set3, 5-fold CV, with learning_rate/weight_decay/batch_size all overridable --
# see models/baselines/rq1_dnn_tuned_cluster_fold.py for the full history (originally tested the
# Aug-19-sweep-winning config, found flat-to-worse at batch_size=256; now also used for the
# batch_size=512 check). DNN trains fast, so this whole job is short.
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#
#   # One fold, project defaults (lr=0.0001, weight_decay=1e-5, batch_size=256):
#   sbatch temp_results_pinn/jobs/run_dnn_tuned_cluster.sh 0
#
#   # All 5 folds (job array), project defaults:
#   sbatch --array=0-4 temp_results_pinn/jobs/run_dnn_tuned_cluster.sh ""
#
#   # All 5 folds at batch_size=512 (writes to a different output dir automatically):
#   sbatch --array=0-4 temp_results_pinn/jobs/run_dnn_tuned_cluster.sh "" 512 CORRECTED_2026-08-22_dnn_bs512_check
#
# Arguments:
#   fold_index       0-4. Leave blank ("") when submitting with --array.
#   batch_size       Defaults to 256 (matches PINN/PINN-k's default).
#   output_dir_name  Which outputs/<name>/ to write to. Defaults to
#                     CORRECTED_2026-08-22_dnn_tuned_cluster. CHANGE THIS whenever batch_size
#                     differs from a previous run, so results never collide.
#
# Results:
#   outputs/<output_dir_name>/fold_<i>/dnn_tuned_summary.json

#SBATCH --job-name=dnn_tuned
#SBATCH --output=logs/temp_results_pinn/%x_%A_%a.out
#SBATCH --error=logs/temp_results_pinn/%x_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/temp_results_pinn

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

FOLD_INDEX=${1:-${SLURM_ARRAY_TASK_ID:-0}}
BATCH_SIZE=${2:-256}
OUTPUT_DIR_NAME=${3:-CORRECTED_2026-08-22_dnn_tuned_cluster}

echo "--- DNN cluster job start ---"
echo "Node: $(hostname)"
echo "Fold index: ${FOLD_INDEX}"
echo "Batch size: ${BATCH_SIZE}"
echo "Output dir name: ${OUTPUT_DIR_NAME}"

python -u models/baselines/rq1_dnn_tuned_cluster_fold.py \
  --fold-index "${FOLD_INDEX}" \
  --batch-size "${BATCH_SIZE}" \
  --output-dir-name "${OUTPUT_DIR_NAME}"

echo "--- DNN cluster job end ---"

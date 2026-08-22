#!/bin/bash
#
# Re-runs the DNN on Set3, 5-fold CV, at the Aug-19-sweep-winning config (learning_rate=0.001,
# weight_decay=1e-3) -- see models/baselines/rq1_dnn_tuned_cluster_fold.py for why. DNN trains
# fast, so this whole job is short.
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#
#   # One fold:
#   sbatch temp_results_pinn/jobs/run_dnn_tuned_cluster.sh 0
#
#   # All 5 folds (job array):
#   sbatch --array=0-4 temp_results_pinn/jobs/run_dnn_tuned_cluster.sh ""
#
# Arguments:
#   fold_index  0-4. Leave blank ("") when submitting with --array.
#
# Results:
#   outputs/CORRECTED_2026-08-22_dnn_tuned_cluster/fold_<i>/dnn_tuned_summary.json

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

echo "--- DNN tuned-config cluster job start ---"
echo "Node: $(hostname)"
echo "Fold index: ${FOLD_INDEX}"

python -u models/baselines/rq1_dnn_tuned_cluster_fold.py \
  --fold-index "${FOLD_INDEX}"

echo "--- DNN tuned-config cluster job end ---"

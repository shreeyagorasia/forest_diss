#!/bin/bash
#
# Cluster job for the PINN forward-pass fix experiment (see temp_results_pinn/PLAN.md).
# Fully isolated: only reads production data/CR-params (never writes to models/ or outputs/),
# and only writes under temp_results_pinn/outputs/full_rerun_cluster/. Cannot collide with
# production results, the local run_full_rerun.py job, or any other job in this repo.
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#
#   # One fold, one variant:
#   sbatch temp_results_pinn/jobs/run_pinn_fix_cluster.sh 0 ymax
#   sbatch temp_results_pinn/jobs/run_pinn_fix_cluster.sh 0 k
#
#   # All 5 folds x 2 variants at once (10 jobs, all running in parallel if the
#   # partition has capacity) -- submit a job array per variant:
#   sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_cluster.sh "" ymax
#   sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_cluster.sh "" k
#   # (leave fold_index blank so each array task uses $SLURM_ARRAY_TASK_ID instead)
#
# Arguments:
#   fold_index   0-4. Leave blank ("") when submitting with --array so each task picks up its
#                own $SLURM_ARRAY_TASK_ID automatically.
#   variant      ymax (y_max-only fix) or k (y_max+k fix). Required.
#
# No hyperparameter sweep args on purpose -- this experiment intentionally uses fixed,
# production-matched settings (see PLAN.md: "no hyperparameter sweeps... bare minimum").
#
# Logs:
#   stdout -> logs/temp_results_pinn/%x_%A_%a.out
#   stderr -> logs/temp_results_pinn/%x_%A_%a.err
#
# Results:
#   temp_results_pinn/outputs/full_rerun_cluster/fold_<i>/pinn_<variant>_fixed_summary.json
#
# Prerequisite (same as the production pinn_env_terrain job):
#   outputs/spatial_block_kfold/chapman_richards/4survey/ must already exist on the cluster
#   (frozen CR params, read-only here), and
#   data/processed/environmental/plot_environmental_features.parquet must be present.

#SBATCH --job-name=pinn_fix_temp
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

echo "--- PINN fix (temp_results_pinn) job start ---"
echo "Node: $(hostname)"
echo "Fold index: ${FOLD_INDEX}"
echo "Variant: ${VARIANT}"

python -u temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold.py \
  --fold-index "${FOLD_INDEX}" \
  --variant "${VARIANT}"

echo "--- PINN fix (temp_results_pinn) job end ---"

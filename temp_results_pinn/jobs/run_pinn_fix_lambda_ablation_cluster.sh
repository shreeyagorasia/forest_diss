#!/bin/bash
#
# Lambda (physics-loss weight) ablation for the CORRECTED PINN/PINN-k -- same GPU/toolchain
# convention as the other proven PINN cluster jobs. Fixed to Set3 (directly comparable to the
# trusted lambda=1.0 Table 3 number). Writes to a separate output directory
# (CORRECTED_2026-08-22_lambda_ablation/) -- cannot collide with anything else.
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#
#   # One fold, one variant, one lambda:
#   sbatch temp_results_pinn/jobs/run_pinn_fix_lambda_ablation_cluster.sh 0 k 0.5
#
#   # All 5 folds for one variant+lambda (job array):
#   sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_lambda_ablation_cluster.sh "" k 0.5
#
#   # Full ablation, both variants -- 7 lambda values x 5 folds x 2 variants = 70 jobs
#   # (lambda=1.0 already exists at full_rerun_cluster/ + full_rerun/ for both variants, no need
#   # to rerun it here):
#   for VARIANT in ymax k; do
#     for LAMBDA in 0 0.25 0.5 0.75 1.5 2.0 3.0; do
#       sbatch --array=0-4 temp_results_pinn/jobs/run_pinn_fix_lambda_ablation_cluster.sh "" "${VARIANT}" "${LAMBDA}"
#     done
#   done
#
# Arguments:
#   fold_index      0-4. Leave blank ("") when submitting with --array.
#   variant         ymax or k. Required.
#   physics_weight  lambda_phys to test (e.g. 0, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0). Required.
#
# Results:
#   temp_results_pinn/outputs/CORRECTED_2026-08-22_lambda_ablation/lambda<X>/<variant>/fold_<i>/pinn_<variant>_fixed_summary.json

#SBATCH --job-name=pinn_fix_lambda
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
PHYSICS_WEIGHT=${3:?"physics_weight is required, e.g. 0, 0.25, 0.5, 0.75, 1.5, 2.0, 3.0"}

echo "--- PINN lambda ablation (temp_results_pinn) job start ---"
echo "Node: $(hostname)"
echo "Fold index: ${FOLD_INDEX}"
echo "Variant: ${VARIANT}"
echo "Physics weight (lambda): ${PHYSICS_WEIGHT}"

python -u temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_lambda_ablation.py \
  --fold-index "${FOLD_INDEX}" \
  --variant "${VARIANT}" \
  --physics-weight "${PHYSICS_WEIGHT}"

echo "--- PINN lambda ablation (temp_results_pinn) job end ---"

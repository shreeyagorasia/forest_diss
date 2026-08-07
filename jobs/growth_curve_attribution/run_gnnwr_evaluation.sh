#!/bin/bash
#
# CPU-only evaluation job -- pools the 5-fold GNNWR results and checks residual spatial
# autocorrelation, all from the already-saved *_test_predictions.csv files. Deliberately
# separate from run_gnnwr.sh: this reads small CSVs only, never the multi-GB model checkpoints
# under outputs/growth_curve_attribution/gnnwr/models/ -- those should stay on the cluster and
# never be rsynced back (the full-population runs' SWNN first layer is ~500 million parameters,
# so each checkpoint is several GB in float32; none of the evaluation below needs the live model,
# only its saved predictions).
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/growth_curve_attribution/run_gnnwr_evaluation.sh [cohort] [scope] [reference_set_size]
#
# Examples:
#   sbatch jobs/growth_curve_attribution/run_gnnwr_evaluation.sh 4survey terrain_wind 0
#   sbatch jobs/growth_curve_attribution/run_gnnwr_evaluation.sh 4survey terrain_wind_plus_management 0
#
# Arguments:
#   cohort               4survey or 6survey. Defaults to 4survey.
#   scope                terrain_wind or terrain_wind_plus_management. Defaults to terrain_wind.
#   reference_set_size   Which reference-set-size run to pool -- 0 means the full-population
#                        5-fold CV run (the one this was built for). Defaults to 0.
#
# Results (small -- safe to rsync back):
#   outputs/growth_curve_attribution/gnnwr/gnnwr_<scope>_<cohort>_ref<size>_kfold_pooled_summary.csv
#   stdout also prints the residual Moran's I comparison table (EN/XGBoost/GNNWR/DNN).
#
# Prerequisite: requirements.txt now includes esda/libpysal (added for this) -- run
#   pip install -r requirements.txt
# on the cluster before the first use of this job, same as after any other requirements.txt change.

#SBATCH --job-name=gnnwr_eval
#SBATCH --output=logs/growth_curve_attribution/%x_%j.out
#SBATCH --error=logs/growth_curve_attribution/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=Teaching
#SBATCH --mem=8G

cd ~/forest_diss

mkdir -p logs/growth_curve_attribution

# Toolchain lives in a shared TA home directory under a dated folder name that gets replaced
# periodically -- hardcoding one date breaks silently (a confusing torch/CUDA import error deep
# inside the Python job, not an obvious "toolchain missing" message) the next time it rotates.
# Finds whatever toolchain-* currently exists instead, picks the most recently modified one, and
# fails loudly with a clear message immediately if none exist at all.
echo "Node: $(hostname)"  # printed BEFORE the toolchain check, on purpose -- if
# /home/htang2 isn'"'"'t mounted on this specific node, everything below dies immediately, and
# without this line the log would never say which node was the problem.
TOOLCHAIN_RC=$(ls -1t /home/htang2/toolchain-*/toolchain.rc 2>/dev/null | head -1)
if [ -z "${TOOLCHAIN_RC}" ]; then
  echo "ERROR: no toolchain.rc found under /home/htang2/toolchain-*/ on node $(hostname) -- /home/htang2 may not be mounted here. Ask a TA if this recurs on the same node." >&2
  exit 1
fi
. "${TOOLCHAIN_RC}"
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-4survey}
SCOPE=${2:-terrain_wind}
REFERENCE_SET_SIZE=${3:-0}

echo "--- GNNWR evaluation job start (CPU only, no GPU requested) ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Scope: ${SCOPE}"
echo "Reference set size: ${REFERENCE_SET_SIZE}"

echo ""
echo "=== Pooling 5-fold CV results ==="
python -u -m models.growth_curve_attribution.pool_gnnwr_kfold_results \
  --cohort "${COHORT}" \
  --scope "${SCOPE}" \
  --reference-set-size "${REFERENCE_SET_SIZE}"

echo ""
echo "=== Residual spatial autocorrelation (Moran's I) ==="
python -u -m models.growth_curve_attribution.residual_spatial_autocorrelation_check

echo "--- GNNWR evaluation job end ---"

#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh [cohort] [scope] [max_epoch] [early_stop] [reference_set_size] [split_seed]
#
# Examples:
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind 300 30
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind_plus_management 300 30
#
# Arguments:
#   cohort               4survey or 6survey. Defaults to 4survey.
#   scope                terrain_wind (17 features, the headline comparison against the
#                        established Elastic Net/XGBoost result of 0.125/0.117 pooled OOF R2) or
#                        terrain_wind_plus_management (22 features, the best-performing static
#                        scope found by broad_environmental_check.py, EN/XGB 0.289/0.302).
#                        Defaults to terrain_wind. See SCOPES in
#                        models/growth_curve_attribution/gnnwr_check.py.
#   max_epoch            Maximum training epochs. Defaults to 200.
#   early_stop           Stop if validation R2 hasn't improved for this many epochs. -1 disables
#                        early stopping. Defaults to 20.
#   reference_set_size   Caps GNNWR's reference/training set to this many plots
#                        (compartment-stratified) -- see gnnwr_check.py's module docstring.
#                        Defaults to 6000 (DEFAULT_REFERENCE_SET_SIZE), sized to fit comfortably
#                        on the GENERIC "gpu:1" gres below. Pass 0 to use the full ~31,000-plot
#                        population instead -- only attempt that with a high-VRAM GPU (e.g.
#                        --gres=gpu:nvidia_rtx_a6000:1), and only if one is actually available:
#                        a specific-GPU-type request sat PD ("ReqNodeNotAvail") for a full day on
#                        this cluster's queue, so the generic pool is the more reliable default.
#   split_seed           Seed for the compartment-based spatial_block_split. Defaults to 42 (this
#                        project's standard SPLIT_SEED), matching every other
#                        growth-curve-attribution check so results are directly comparable.
#
# Why the reference set is capped rather than requesting bigger hardware (see gnnwr_check.py's
# module docstring for the full investigation, including a second, separate memory bottleneck in
# gnnwr's own diagnostic code that is patched inside gnnwr_check.py itself): GNNWR's spatial-
# weighting sub-network takes each plot's full distance-to-every-reference-plot vector as input,
# so its first layer's width equals the reference-set size. At the full ~31,000-plot population
# that is roughly 500 million parameters, which OOM's the cluster's generic GPU allocation
# (10.57 GiB VRAM). A specific bigger GPU (RTX A6000, 48 GiB) fixes that particular number, but
# is not reliably available on this cluster's queue in practice. Capping the reference set to
# 6,000 plots (compartment-stratified, so it still covers the whole forest) keeps this comfortably
# within the generic pool's budget without waiting on a scarce resource -- a disclosed,
# deliberate trade-off (GNNWR sees fewer reference points than EN/XGBoost/the DNN baselines see
# training rows), not a hidden one.
#
# Logs:
#   stdout -> logs/growth_curve_attribution/gnnwr_<jobid>.out
#   stderr -> logs/growth_curve_attribution/gnnwr_<jobid>.err
#
# Results:
#   outputs/growth_curve_attribution/gnnwr/<run_name>_test_predictions.csv
#   outputs/growth_curve_attribution/gnnwr/models/<run_name>.pkl (best-on-validation checkpoint)
#
# Prerequisite:
#   Reads the terrain/wind/management columns from
#   data/processed/environmental/plot_environmental_features.parquet and the growth-curve target
#   tables gnnwr_check.py builds from -- make sure both are on the cluster before submitting.

#SBATCH --job-name=gnnwr
#SBATCH --output=logs/growth_curve_attribution/%x_%j.out
#SBATCH --error=logs/growth_curve_attribution/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/growth_curve_attribution outputs/growth_curve_attribution/gnnwr

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-4survey}
SCOPE=${2:-terrain_wind}
MAX_EPOCH=${3:-200}
EARLY_STOP=${4:-20}
REFERENCE_SET_SIZE=${5:-6000}
SPLIT_SEED=${6:-42}

echo "--- GNNWR job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Scope: ${SCOPE}"
echo "Max epoch: ${MAX_EPOCH}"
echo "Early stop patience: ${EARLY_STOP}"
echo "Reference set size: ${REFERENCE_SET_SIZE}"
echo "Split seed: ${SPLIT_SEED}"

python -u -m models.growth_curve_attribution.gnnwr_check \
  --cohort "${COHORT}" \
  --scope "${SCOPE}" \
  --max-epoch "${MAX_EPOCH}" \
  --early-stop "${EARLY_STOP}" \
  --reference-set-size "${REFERENCE_SET_SIZE}" \
  --split-seed "${SPLIT_SEED}" \
  --use-gpu

echo "--- GNNWR job end ---"

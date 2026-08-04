#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh [cohort] [scope] [max_epoch] [early_stop] [split_seed]
#
# Examples:
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind 300 30
#   sbatch jobs/growth_curve_attribution/run_gnnwr.sh 4survey terrain_wind_plus_management 300 30
#
# Arguments:
#   cohort       4survey or 6survey. Defaults to 4survey.
#   scope        terrain_wind (17 features, the headline comparison against the established
#                Elastic Net/XGBoost result of 0.125/0.117 pooled OOF R2) or
#                terrain_wind_plus_management (22 features, the best-performing static scope
#                found by broad_environmental_check.py, EN/XGB 0.289/0.302). Defaults to
#                terrain_wind. See SCOPES in models/growth_curve_attribution/gnnwr_check.py.
#   max_epoch    Maximum training epochs. Defaults to 200.
#   early_stop   Stop if validation R2 hasn't improved for this many epochs. -1 disables early
#                stopping. Defaults to 20.
#   split_seed   Seed for the compartment-based spatial_block_split. Defaults to 42 (this
#                project's standard SPLIT_SEED), matching every other growth-curve-attribution
#                check so results are directly comparable.
#
# Why this needs the cluster, not a laptop (see gnnwr_check.py's module docstring for the full
# investigation): GNNWR's spatial-weighting sub-network takes each plot's full distance-to-every-
# training-plot vector as input, so its first layer's width equals the training set size --
# roughly 500 million parameters for this project's ~31,000-plot 4survey training set, plus
# several GB of pairwise distance matrices. That does not fit an 8.6 GB RAM laptop. --mem is set
# well above this project's other DNN/PINN jobs (16G) for that reason.
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
#SBATCH --time=08:00:00
#SBATCH --partition=Teaching
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

cd ~/forest_diss

mkdir -p logs/growth_curve_attribution outputs/growth_curve_attribution/gnnwr

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-4survey}
SCOPE=${2:-terrain_wind}
MAX_EPOCH=${3:-200}
EARLY_STOP=${4:-20}
SPLIT_SEED=${5:-42}

echo "--- GNNWR job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Scope: ${SCOPE}"
echo "Max epoch: ${MAX_EPOCH}"
echo "Early stop patience: ${EARLY_STOP}"
echo "Split seed: ${SPLIT_SEED}"

python -u -m models.growth_curve_attribution.gnnwr_check \
  --cohort "${COHORT}" \
  --scope "${SCOPE}" \
  --max-epoch "${MAX_EPOCH}" \
  --early-stop "${EARLY_STOP}" \
  --split-seed "${SPLIT_SEED}" \
  --use-gpu

echo "--- GNNWR job end ---"

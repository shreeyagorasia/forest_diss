#!/bin/bash
#
# Run on the ICF cluster from the project root:
#
#   cd ~/forest_diss
#   sbatch jobs/growth_curve_attribution/run_rq3_en_xgb.sh [cohort] [set_name] [k_folds] [seed]
#
# CPU-only job. run_columns() already runs the full k-fold spatial CV internally in one call --
# unlike DNN/PINN/GNNWR, there's no "one job per fold" here, one job per (cohort, set_name)
# already produces the pooled result.
#
# Examples:
#   sbatch jobs/growth_curve_attribution/run_rq3_en_xgb.sh 4survey nested_set2_top5
#   sbatch jobs/growth_curve_attribution/run_rq3_en_xgb.sh 6survey nested_set3_gated_terrain_wind
#
# Arguments:
#   cohort     4survey or 6survey.
#   set_name   nested_set2_top5 / nested_set3_gated_terrain_wind / nested_set4_gated_all /
#              nested_set5_all_ungated. RQ3's Elastic Net here is deliberately NOT VIF-screened
#              (see documentation/experiment_log.md's 2026-08-10 entry) -- same raw sets as
#              XGBoost/GNNWR.
#   k_folds    Defaults to 5.
#   seed       Defaults to 42.

#SBATCH -p Teaching
#SBATCH --job-name=rq3_en_xgb
#SBATCH --output=logs/rq123_methodology/%x_%j.out
#SBATCH --error=logs/rq123_methodology/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G

cd ~/forest_diss

mkdir -p logs/rq123_methodology outputs

. /home/htang2/toolchain-20251006/toolchain.rc
. .venv/bin/activate

export PYTHONPATH="$(pwd)"

COHORT=${1:-4survey}
SET_NAME=${2:-nested_set2_top5}
K_FOLDS=${3:-5}
SEED=${4:-42}

echo "--- RQ3 EN/XGBoost job start ---"
echo "Node: $(hostname)"
echo "Cohort: ${COHORT}"
echo "Set name: ${SET_NAME}"

python -u -m models.growth_curve_attribution.run_rq3_en_xgb \
  --cohort "${COHORT}" \
  --set-name "${SET_NAME}" \
  --k-folds "${K_FOLDS}" \
  --seed "${SEED}"

echo "--- RQ3 EN/XGBoost job end ---"

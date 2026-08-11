#!/bin/bash
# RUN THIS ON: the cluster.
#
# RQ1 temporal-vs-spatial integrity check, current tiers. Set3 only (strongest, most consistent
# set across DNN/PINN/PINN_k -- avoids "which set" as a confound), all 3 models, both cohorts,
# spatial_block vs temporal as a matched SINGLE-split pair (not kfold) -- same comparison shape
# as the existing old-tier evidence (dnn_pinn_basecase_2026-07-30), so the new number is directly
# comparable to it, not just a different kind of check. 12 jobs total. Smoke-tested locally
# 2026-08-11 (DNN, temporal, Set3, 3 epochs) before this was written -- passed clean.
set -e

SET_NAME="nested_set3_gated_terrain_wind_vif"

mkdir -p logs/rq123_methodology/rq1_temporalcheck

for SPLIT_TYPE in spatial_block temporal; do
  for COHORT in 4survey 6survey; do
    sbatch --job-name="rq1_temporalcheck_dnn_${COHORT}_${SPLIT_TYPE}" --output="logs/rq123_methodology/rq1_temporalcheck/%x_%j.out" --error="logs/rq123_methodology/rq1_temporalcheck/%x_%j.err" jobs/dnn_env_terrain/run_dnn_env_terrain.sh "$COHORT" 500 40 "$SPLIT_TYPE" 42 "rq1_temporalcheck_dnn_env_terrain_${SET_NAME}_seed42" 256 "$SET_NAME" 0.0 42 5 0 0.0001

    sbatch --job-name="rq1_temporalcheck_pinn_${COHORT}_${SPLIT_TYPE}" --output="logs/rq123_methodology/rq1_temporalcheck/%x_%j.out" --error="logs/rq123_methodology/rq1_temporalcheck/%x_%j.err" jobs/pinn_env_terrain/run_pinn_env_terrain.sh "$COHORT" 500 40 "$SPLIT_TYPE" 1.0 1.0 "rq1_temporalcheck_pinn_env_terrain_${SET_NAME}_seed42" 42 256 "$SET_NAME" 0.0 42 5 0 0.0001

    sbatch --job-name="rq1_temporalcheck_pinnk_${COHORT}_${SPLIT_TYPE}" --output="logs/rq123_methodology/rq1_temporalcheck/%x_%j.out" --error="logs/rq123_methodology/rq1_temporalcheck/%x_%j.err" jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh "$COHORT" 500 40 "$SPLIT_TYPE" 1.0 1.0 "rq1_temporalcheck_pinn_env_terrain_k_${SET_NAME}_seed42" 42 256 "$SET_NAME" 0.0 42 5 0 "" 0.0001
  done
done

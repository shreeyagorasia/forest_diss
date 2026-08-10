#!/bin/bash
# RUN THIS ON: the cluster.
#
# Step 2a -- RQ3 Elastic Net + XGBoost. 3 sets x 2 cohorts = 6 jobs. CPU-only, fast --
# run_columns() does the full 5-fold spatial CV internally in one call (see the plan file for
# why this one can't be split into fit/evaluate the way RQ2 was).
set -e

for SET_NAME in nested_set2_top10 nested_set3_gated_terrain_wind_vif nested_set4_gated_all_vif; do
  for COHORT in 4survey 6survey; do
    sbatch jobs/growth_curve_attribution/run_rq3_en_xgb.sh "$COHORT" "$SET_NAME" 5 42
  done
done

#!/bin/bash
# RUN THIS ON: the cluster.
#
# Step 2b -- RQ3 GNNWR. 3 sets x 2 cohorts x 5 folds = 30 jobs. GPU, real training (200 epochs,
# early-stop 20 -- the established real-run defaults, not the smoke test's cheap 5-epoch check).
# reference_set_size=0 (full population) -- the H200 MIG slice this project's GNNWR jobs already
# use comfortably fits the full training population, no need to shrink it (see
# jobs/growth_curve_attribution/run_gnnwr.sh's own header comment for the memory-cost reasoning).
# Fit+evaluate combined on the cluster by design -- the model itself is too large to sync down.
set -e

for SET_NAME in nested_set2_top10 nested_set3_gated_terrain_wind_vif nested_set4_gated_all_vif; do
  for COHORT in 4survey 6survey; do
    for FOLD in 0 1 2 3 4; do
      sbatch jobs/growth_curve_attribution/run_rq3_gnnwr.sh "$COHORT" "$SET_NAME" 200 20 0 42 "$FOLD" 5
    done
  done
done

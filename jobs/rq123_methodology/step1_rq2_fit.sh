#!/bin/bash
# RUN THIS ON: the cluster (ssh in, cd ~/forest_diss, then: bash jobs/rq123_methodology/step1_rq2_fit.sh)
#
# Step 1 -- RQ2 FIT. 3 sets x 5 folds = 15 jobs. CPU-only, fast (each job took ~10s in the
# smoke test). 4survey only -- see documentation/experiment_log.md's cohort-justification entry.
set -e

for SET_NAME in nested_set2_top10 nested_set3_gated_terrain_wind_vif nested_set4_gated_all_vif; do
  for FOLD in 0 1 2 3 4; do
    sbatch jobs/spatial_attribution/run_rq2_attribution.sh 4survey "$SET_NAME" spatial_block_kfold "$FOLD" 5 42
  done
done

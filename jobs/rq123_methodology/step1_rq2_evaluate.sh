#!/bin/bash
# RUN THIS ON: your Mac, after step1_rq2_sync.sh finishes.
#
# Step 1 -- RQ2 EVALUATE. 3 sets x 5 folds = 15 local evaluate calls.
set -e

for SET_NAME in nested_set2_top10 nested_set3_gated_terrain_wind_vif nested_set4_gated_all_vif; do
  for FOLD in 0 1 2 3 4; do
    .venv/bin/python -m models.spatial_attribution.evaluate_rq2_attribution \
      --cohort 4survey --set-name "$SET_NAME" --split-type spatial_block_kfold \
      --fold-index "$FOLD" --n-folds 5 --split-seed 42
  done
done

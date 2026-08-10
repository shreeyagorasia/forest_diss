#!/bin/bash
# RUN THIS ON: your Mac, after squeue on the cluster shows step1_rq2_fit.sh's 15 jobs COMPLETED.
#
# Step 1 -- RQ2 SYNC. Targeted download, not the whole outputs/ tree -- each --include is one
# specific thing let through (the /*** suffix recurses into all 5 folds under it in one rule,
# no need to list each fold separately), the final --exclude='*' blocks everything else.
set -e

rsync -avz -e "ssh -J s2887183@student.ssh.inf.ed.ac.uk" \
  --include='/spatial_block_kfold/' \
  --include='/spatial_block_kfold/rq2_attribution_nested_set2_top10/' \
  --include='/spatial_block_kfold/rq2_attribution_nested_set2_top10/***' \
  --include='/spatial_block_kfold/rq2_attribution_nested_set3_gated_terrain_wind_vif/' \
  --include='/spatial_block_kfold/rq2_attribution_nested_set3_gated_terrain_wind_vif/***' \
  --include='/spatial_block_kfold/rq2_attribution_nested_set4_gated_all_vif/' \
  --include='/spatial_block_kfold/rq2_attribution_nested_set4_gated_all_vif/***' \
  --include='/run_logs/' --include='/run_logs/***' \
  --exclude='*' \
  s2887183@hastings.inf.ed.ac.uk:~/forest_diss/outputs/ \
  /Users/shreeyagorasia/UoE_Docs/Dissertation/forest_diss/outputs/

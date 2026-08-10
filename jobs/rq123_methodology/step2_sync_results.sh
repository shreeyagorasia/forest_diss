#!/bin/bash
# RUN THIS ON: your Mac, after squeue shows step2_rq3_en_xgb.sh's 6 jobs AND
# step2_rq3_gnnwr.sh's 30 jobs all COMPLETED.
#
# Step 2 -- RQ3 SYNC. EN/XGBoost's small metrics/predictions come down in full (/*** recurses
# through both cohorts under each set automatically). GNNWR is the one exception in this whole
# plan to "sync results, not just metrics" -- only its *_test_predictions.csv files come down
# (one wildcard pattern per set, covering both cohorts x 5 folds each); its models/ and internal
# runs/ logs never do, same as your own existing habit -- the trained model itself is too large
# to be worth moving, only the predictions needed to compute pooled metrics matter here.
set -e

rsync -avz -e "ssh -J s2887183@student.ssh.inf.ed.ac.uk" \
  --include='/spatial_block_kfold/' \
  --include='/spatial_block_kfold/rq3_en_xgb_nested_set2_top10/' \
  --include='/spatial_block_kfold/rq3_en_xgb_nested_set2_top10/***' \
  --include='/spatial_block_kfold/rq3_en_xgb_nested_set3_gated_terrain_wind_vif/' \
  --include='/spatial_block_kfold/rq3_en_xgb_nested_set3_gated_terrain_wind_vif/***' \
  --include='/spatial_block_kfold/rq3_en_xgb_nested_set4_gated_all_vif/' \
  --include='/spatial_block_kfold/rq3_en_xgb_nested_set4_gated_all_vif/***' \
  --include='/growth_curve_attribution/' \
  --include='/growth_curve_attribution/gnnwr/' \
  --include='/growth_curve_attribution/gnnwr/gnnwr_nested_set2_top10_*_test_predictions.csv' \
  --include='/growth_curve_attribution/gnnwr/gnnwr_nested_set3_gated_terrain_wind_vif_*_test_predictions.csv' \
  --include='/growth_curve_attribution/gnnwr/gnnwr_nested_set4_gated_all_vif_*_test_predictions.csv' \
  --include='/run_logs/' --include='/run_logs/***' \
  --exclude='/growth_curve_attribution/gnnwr/models/' \
  --exclude='/growth_curve_attribution/gnnwr/runs/' \
  --exclude='*' \
  s2887183@hastings.inf.ed.ac.uk:~/forest_diss/outputs/ \
  /Users/shreeyagorasia/UoE_Docs/Dissertation/forest_diss/outputs/

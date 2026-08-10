#!/bin/bash
# RUN THIS ON: your Mac, after squeue shows all 90 of step3_rq1_sweep_fit.sh's jobs COMPLETED.
#
# Step 3 -- RQ1 sweep SYNC. One include-pair per (model, set) combination -- /*** recurses
# through both cohorts and all 5 folds under each automatically, so this is 9 pairs (3 models x
# 3 sets), not 90.
set -e

FILTER_ARGS=()
for MODEL in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for SET_NAME in nested_set2_top10 nested_set3_gated_terrain_wind_vif nested_set4_gated_all_vif; do
    RUN_NAME="rq1_${MODEL}_${SET_NAME}_seed42"
    FILTER_ARGS+=(--include="/spatial_block_kfold/${RUN_NAME}/")
    FILTER_ARGS+=(--include="/spatial_block_kfold/${RUN_NAME}/***")
  done
done

rsync -avz -e "ssh -J s2887183@student.ssh.inf.ed.ac.uk" \
  --include='/spatial_block_kfold/' \
  "${FILTER_ARGS[@]}" \
  --include='/run_logs/' --include='/run_logs/***' \
  --exclude='*' \
  s2887183@hastings.inf.ed.ac.uk:~/forest_diss/outputs/ \
  /Users/shreeyagorasia/UoE_Docs/Dissertation/forest_diss/outputs/

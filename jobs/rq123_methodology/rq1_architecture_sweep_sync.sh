#!/bin/bash
# RUN THIS ON: your Mac, after squeue shows all 8 of rq1_architecture_sweep_fit.sh's jobs
# COMPLETED.
#
# Single spatial_block split (not spatial_block_kfold) -- these land under a different top-level
# output folder than the main RQ1 sweep, so this needs its own sync, not step3_rq1_sweep_sync.sh's
# filter.
set -e

SET_NAME="nested_set3_gated_terrain_wind_vif"

FILTER_ARGS=()
for ARCH_NAME in small medium large deeper; do
  RUN_NAME="rq1_dnn_env_terrain_${SET_NAME}_arch${ARCH_NAME}"
  FILTER_ARGS+=(--include="/spatial_block/${RUN_NAME}/")
  FILTER_ARGS+=(--include="/spatial_block/${RUN_NAME}/***")
done

rsync -avz -e "ssh -J s2887183@student.ssh.inf.ed.ac.uk" \
  --include='/spatial_block/' \
  "${FILTER_ARGS[@]}" \
  --include='/run_logs/' --include='/run_logs/***' \
  --exclude='*' \
  s2887183@hastings.inf.ed.ac.uk:~/forest_diss/outputs/ \
  /Users/shreeyagorasia/UoE_Docs/Dissertation/forest_diss/outputs/

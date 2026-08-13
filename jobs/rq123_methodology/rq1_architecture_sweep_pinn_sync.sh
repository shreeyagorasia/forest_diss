#!/bin/bash
# RUN THIS ON: your Mac, after squeue shows all 16 of rq1_architecture_sweep_pinn_fit.sh's jobs
# COMPLETED.
#
# Pulls fit checkpoints only (no predictions.csv yet -- evaluate runs locally afterward, same
# pattern as the DNN architecture sweep: evaluate is CPU-only, no cluster round-trip needed).
set -e

SET_NAME="nested_set3_gated_terrain_wind_vif"

FILTER_ARGS=()
for MODEL in rq1_pinn_env_terrain rq1_pinn_env_terrain_k; do
  for ARCH_NAME in small medium large deeper; do
    RUN_NAME="${MODEL}_${SET_NAME}_arch${ARCH_NAME}"
    FILTER_ARGS+=(--include="/spatial_block/${RUN_NAME}/")
    FILTER_ARGS+=(--include="/spatial_block/${RUN_NAME}/***")
  done
done

rsync -avz -e "ssh -J s2887183@student.ssh.inf.ed.ac.uk" \
  --include='/spatial_block/' \
  "${FILTER_ARGS[@]}" \
  --include='/run_logs/' --include='/run_logs/***' \
  --exclude='*' \
  s2887183@hastings.inf.ed.ac.uk:~/forest_diss/outputs/ \
  /Users/shreeyagorasia/UoE_Docs/Dissertation/forest_diss/outputs/

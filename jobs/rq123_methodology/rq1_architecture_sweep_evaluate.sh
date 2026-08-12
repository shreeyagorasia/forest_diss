#!/bin/bash
# RUN THIS ON: the cluster, after rq1_architecture_sweep_fit.sh's 8 jobs all show COMPLETED.
# CPU-only, cheap, matches this project's fit-on-GPU/evaluate-on-CPU split for every other DNN run.
set -e

SET_NAME="nested_set3_gated_terrain_wind_vif"

mkdir -p logs/dnn_env_terrain

for ARCH_NAME in small medium large deeper; do
  for COHORT in 4survey 6survey; do
    RUN_NAME="rq1_dnn_env_terrain_${SET_NAME}_arch${ARCH_NAME}"
    sbatch --job-name="rq1_arch_eval_${ARCH_NAME}_${COHORT}" --output="logs/dnn_env_terrain/%x_%j.out" --error="logs/dnn_env_terrain/%x_%j.err" jobs/dnn_env_terrain/evaluate_dnn_env_terrain.sh "$COHORT" spatial_block "$RUN_NAME"
  done
done

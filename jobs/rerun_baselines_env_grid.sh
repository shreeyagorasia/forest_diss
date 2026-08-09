#!/bin/bash
# Submits the full baseline-with-environment grid: 3 feature tiers x 2 cohorts, under both
# plot_level (1 run each) and spatial_block_kfold (5 folds each) -- matches E6_stage_sweep's own
# three tiers and 5-fold rigor level exactly, so these numbers are directly comparable to it.
# Each job fits all three baselines (linear, RF, XGBoost) in one go -- see
# models/baselines/run_baselines_env.py.
# Run on the cluster: bash jobs/rerun_baselines_env_grid.sh
# Total: 3 tiers x 2 cohorts x (1 plot_level + 5 spatial_block_kfold folds) = 36 jobs.
set -e

FEATURE_SETS="stage1_terrain stage2_terrain_wind stage4_all_environmental"

for cohort in 4survey 6survey; do
  for feature_set in $FEATURE_SETS; do
    sbatch jobs/baselines/run_baselines_env.sh "$cohort" plot_level "$feature_set"

    for fold in 0 1 2 3 4; do
      sbatch jobs/baselines/run_baselines_env.sh "$cohort" spatial_block_kfold "$feature_set" 42 5 "$fold"
    done
  done
done

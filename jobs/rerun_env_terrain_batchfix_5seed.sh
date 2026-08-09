#!/bin/bash
# Step 1: fit jobs. Run on the cluster: bash jobs/rerun_env_terrain_batchfix_5seed.sh
# Purpose: pure seed-robustness check, NOT a batch-size fix -- checked run_logs directly
# (2026-08-09) and confirmed the actual established fits behind the cited dnn_env_terrain vs
# pinn_env_terrain*/pinn_env_terrain_k numbers already use --batch-size 256 for all three via
# explicit CLI override; dnn_env_terrain.py's file-level default of 512 was never actually used
# in any cited comparison, so there was no confound to fix. This job just gets 5-seed evidence
# at each model's default hyperparameters (dropout=0.0, lr=0.0001, default feature tier =
# terrain_wind_solid) -- the same config ("E3") that already produced the y_max/k sign-flip
# finding, so this reseed is a faithful replication of that result, not a new, incomparable setup.
# Cluster only -- local Mac run of this model family was abandoned earlier (24 min/fold).
# After all 30 of these finish, run jobs/evaluate_env_terrain_batchfix_5seed.sh
set -e

for cohort in 4survey 6survey; do
  for seed in 42 43 44 45 46; do
    run_name="env_terrain_batchfix_seed${seed}"

    sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh \
      "$cohort" 500 40 spatial_block "$seed" "dnn_${run_name}" 256

    sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh \
      "$cohort" 500 40 spatial_block 1.0 1.0 "pinn_${run_name}" "$seed" 256

    sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh \
      "$cohort" 500 40 spatial_block 1.0 1.0 "pinnk_${run_name}" "$seed" 256
  done
done

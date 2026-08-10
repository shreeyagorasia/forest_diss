#!/bin/bash
# RUN THIS ON: the cluster (ssh in, cd ~/forest_diss, then run this file).
# Tier 4 -- FIT. env_terrain @ set2 (default tier), 5-seed, spatial_block_kfold, all 5 folds
# (300 jobs total, 150 fit here). Same gap as tier 3, env_terrain side -- bigger because it's
# 3 models instead of 2.
# Prerequisite: run tier1_fit.sh on your Mac FIRST (pushes the 5 per-fold CR anchors).
source "$(dirname "$0")/_lib.sh"

for model in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for cohort in $COHORTS; do
    for seed in $SEEDS; do
      for fold in 0 1 2 3 4; do
        submit_env_terrain_fit "$model" "$cohort" "$seed" spatial_block_kfold set2 "$fold"
      done
    done
  done
done

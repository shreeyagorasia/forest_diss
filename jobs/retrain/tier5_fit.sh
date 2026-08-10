#!/bin/bash
# RUN THIS ON: the cluster (ssh in, cd ~/forest_diss, then run this file).
# Tier 5 -- FIT. env_terrain @ set2 (default tier), 5-seed, temporal (60 jobs total, 30 fit
# here). Secondary RQ (temporal generalisation) -- worth doing, lower priority than the
# spatial_block tiers above since spatial_block is the dissertation's named primary test.
# Prerequisite: run tier1_fit.sh on your Mac FIRST (pushes temporal's CR anchor).
source "$(dirname "$0")/_lib.sh"

for model in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for cohort in $COHORTS; do
    for seed in $SEEDS; do
      submit_env_terrain_fit "$model" "$cohort" "$seed" temporal set2
    done
  done
done

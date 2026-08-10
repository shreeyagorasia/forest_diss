#!/bin/bash
# RUN THIS ON: the cluster (ssh in, cd ~/forest_diss, then run this file).
# Tier 2 -- FIT. env_terrain @ set2 (default tier), 5-seed, spatial_block ONLY
# (60 jobs total, 30 fit here). HIGH VALUE: the env-conditioned DNN/PINN comparison -- the
# dissertation's primary generalisation test -- currently has ZERO seed-robustness, unlike the
# no-env comparison (already 5-seeded for this exact split).
# Prerequisite: run tier1_fit.sh on your Mac FIRST (fits baselines locally, pushes the CR anchor
# these PINN jobs read up to the cluster).
source "$(dirname "$0")/_lib.sh"

for model in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for cohort in $COHORTS; do
    for seed in $SEEDS; do
      submit_env_terrain_fit "$model" "$cohort" "$seed" spatial_block set2
    done
  done
done

#!/bin/bash
# RUN THIS ON: the cluster (ssh in, cd ~/forest_diss, then run this file).
# Tier 3 -- FIT. no-env models, 5-seed, spatial_block_kfold, all 5 folds
# (200 jobs total, 100 fit here). The POOLED/CI'd headline no-env result is currently 1 seed
# per fold -- this is the seed-robustness check for the number that gets cited with a CI.
# Prerequisite: run tier1_fit.sh on your Mac FIRST (fits baselines locally per fold, pushes the
# 5 per-fold CR anchors these PINN jobs read up to the cluster).
source "$(dirname "$0")/_lib.sh"

for model in dnn_noenv pinn_noenv; do
  for cohort in $COHORTS; do
    for seed in $SEEDS; do
      for fold in 0 1 2 3 4; do
        submit_noenv_fit "$model" "$cohort" "$seed" spatial_block_kfold "$fold"
      done
    done
  done
done

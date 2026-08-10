#!/bin/bash
# RUN THIS ON: the cluster (ssh in, cd ~/forest_diss, then run this file).
# Tier 7 -- FIT. Full feature-tier reseed: set3a + set3b + set4, x 5 seeds, x every split
# (including all 5 kfold folds), on top of tiers 2/4/5/6's set2-only coverage
# (720 fit jobs here, 1,440 total with evaluate). NOT RECOMMENDED as a default -- E6 already has
# single-seed coverage answering "does more environment help"; this mostly reseeds an
# already-answered question rather than closing an open one. Only run this if you're
# deliberately committing multiple days of cluster time.
# Prerequisite: run tier1_fit.sh on your Mac FIRST (pushes every split's CR anchor, including
# all 5 kfold folds).
# Overlaps tier0 by exactly one cell (set4/plot_level/seed42) -- harmless, idempotent, just a
# few minutes of redundant compute, not worth special-casing out.
source "$(dirname "$0")/_lib.sh"

for model in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for cohort in $COHORTS; do
    for seed in $SEEDS; do
      for tag in set3a set3b set4; do
        for split in plot_level spatial_block temporal; do
          submit_env_terrain_fit "$model" "$cohort" "$seed" "$split" "$tag"
        done
        for fold in 0 1 2 3 4; do
          submit_env_terrain_fit "$model" "$cohort" "$seed" spatial_block_kfold "$tag" "$fold"
        done
      done
    done
  done
done

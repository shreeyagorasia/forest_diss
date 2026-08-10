#!/bin/bash
# RUN THIS ON: the cluster (ssh in, cd ~/forest_diss, then run this file).
# Tier 6 -- FIT. no-env + env_terrain @ set2, 5-seed, plot_level (50 jobs total, 25 fit here).
# LOWEST priority tier -- the ledger's own text calls plot_level "the easy case, not a
# generalisation test". Do this last, if at all.
# Prerequisite: run tier1_fit.sh on your Mac FIRST (pushes plot_level's CR anchor).
source "$(dirname "$0")/_lib.sh"

for model in dnn_noenv pinn_noenv; do
  for cohort in $COHORTS; do
    for seed in $SEEDS; do
      submit_noenv_fit "$model" "$cohort" "$seed" plot_level
    done
  done
done

for model in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for cohort in $COHORTS; do
    for seed in $SEEDS; do
      submit_env_terrain_fit "$model" "$cohort" "$seed" plot_level set2
    done
  done
done

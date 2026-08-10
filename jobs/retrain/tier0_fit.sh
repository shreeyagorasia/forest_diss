#!/bin/bash
# RUN THIS ON: the cluster (ssh in, cd ~/forest_diss, then run this file).
# Tier 0 -- FIT. Already-known-broken, cheapest fix (12 jobs total: 6 cluster fits here).
# Refits the 3 stage4_all_environmental plot_level cells the ledger's own text flags as
# predating the tas_mean/HadUK-Grid fix (commit b8884a4, 2026-08-09). Single seed (42) -- a
# targeted correction, not a robustness sweep.
#
# Prerequisite: run tier1_fit.sh on your Mac FIRST -- it fits baselines locally and pushes the
# resulting CR anchor (chapman_richards/params.json) up to the cluster, which every PINN job
# below reads from wherever IT runs (the cluster, for fitting).
source "$(dirname "$0")/_lib.sh"

for model in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for cohort in $COHORTS; do
    submit_env_terrain_fit "$model" "$cohort" 42 plot_level set4
  done
done

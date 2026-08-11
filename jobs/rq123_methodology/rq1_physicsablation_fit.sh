#!/bin/bash
# RUN THIS ON: the cluster.
#
# RQ1 physics-weight ablation -- isolates whether the physics LOSS itself helps, given the
# y_max/k sub-network architecture is already present (DNN has neither, so it isn't included
# here -- it's the existing zero-architecture reference point, not something to ablate).
# PINN + PINN_k only, Set3 only (same set as the existing physics_weight=1.0 numbers, so this is
# a true delta, not confounded by feature richness), both cohorts, full 5-fold
# spatial_block_kfold (matching the main sweep's rigor -- this needs to be as trustworthy as the
# numbers it's being compared against, not a cheaper single-split check).
# physics_weight=1.0/trajectory_weight=1.0 already exist from the main sweep
# (rq1_pinn_env_terrain_nested_set3_gated_terrain_wind_vif_seed42 etc) -- only the
# physics_weight=0.0/trajectory_weight=0.0 control below is new. 20 jobs total. Smoke-tested
# locally 2026-08-11 (PINN, w=0/0, Set3, fold 0, 3 epochs) before this was written -- passed clean.
set -e

SET_NAME="nested_set3_gated_terrain_wind_vif"

mkdir -p logs/rq123_methodology/rq1_physicsablation

for COHORT in 4survey 6survey; do
  for FOLD in 0 1 2 3 4; do
    sbatch --job-name="rq1_physicsablation_pinn_${COHORT}_fold${FOLD}" --output="logs/rq123_methodology/rq1_physicsablation/%x_%j.out" --error="logs/rq123_methodology/rq1_physicsablation/%x_%j.err" jobs/pinn_env_terrain/run_pinn_env_terrain.sh "$COHORT" 500 40 spatial_block_kfold 0.0 0.0 "rq1_pinn_env_terrain_${SET_NAME}_w0_seed42" 42 256 "$SET_NAME" 0.0 42 5 "$FOLD" 0.0001

    sbatch --job-name="rq1_physicsablation_pinnk_${COHORT}_fold${FOLD}" --output="logs/rq123_methodology/rq1_physicsablation/%x_%j.out" --error="logs/rq123_methodology/rq1_physicsablation/%x_%j.err" jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh "$COHORT" 500 40 spatial_block_kfold 0.0 0.0 "rq1_pinn_env_terrain_k_${SET_NAME}_w0_seed42" 42 256 "$SET_NAME" 0.0 42 5 "$FOLD" "" 0.0001
  done
done

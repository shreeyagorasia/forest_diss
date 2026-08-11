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
# physics_weight=1.0/trajectory_weight=1.0 already exist from the main sweep, IN KFOLD FORM
# (rq1_pinn_env_terrain_nested_set3_gated_terrain_wind_vif_seed42 etc) -- only the
# physics_weight=0.0/trajectory_weight=0.0 control below is new. 20 jobs. Smoke-tested locally
# 2026-08-11 (PINN, w=0/0, Set3, fold 0, 3 epochs) before this was written -- passed clean.
#
# SECOND TIER (added 2026-08-11): a cheap single-split exploratory pass across
# physics_weight/trajectory_weight in {0, 1, 2} -- off / the standard default used everywhere
# else in this project / doubled. Answers "does more physics emphasis help" (something a bare
# 0-vs-1 comparison can't rule out) without paying for a dense grid -- existing evidence
# (experiment_log.md's 2026-07-30 pinn_noenv 10-point grid) already found every value BETWEEN 0
# and 1 landed within noise of both endpoints, so this deliberately skips the in-between values
# (0.1, 0.5) rather than re-discovering the same null result. Single split (spatial_block, not
# kfold) since this is a cheap shape check, not a citable headline number -- only the w=0 vs w=1
# comparison above needs 5-fold rigor. weight=1 here is a NEW single-split run too (the main
# sweep only has a kfold version at w=1, not a single-split one) -- all 3 weights are new at this
# tier. 3 weights x 2 models x 2 cohorts = 12 jobs.
#
# Total this file: 32 jobs.
set -e

SET_NAME="nested_set3_gated_terrain_wind_vif"

mkdir -p logs/rq123_methodology/rq1_physicsablation

for COHORT in 4survey 6survey; do
  for FOLD in 0 1 2 3 4; do
    sbatch --job-name="rq1_physicsablation_pinn_${COHORT}_fold${FOLD}" --output="logs/rq123_methodology/rq1_physicsablation/%x_%j.out" --error="logs/rq123_methodology/rq1_physicsablation/%x_%j.err" jobs/pinn_env_terrain/run_pinn_env_terrain.sh "$COHORT" 500 40 spatial_block_kfold 0.0 0.0 "rq1_pinn_env_terrain_${SET_NAME}_w0_seed42" 42 256 "$SET_NAME" 0.0 42 5 "$FOLD" 0.0001

    sbatch --job-name="rq1_physicsablation_pinnk_${COHORT}_fold${FOLD}" --output="logs/rq123_methodology/rq1_physicsablation/%x_%j.out" --error="logs/rq123_methodology/rq1_physicsablation/%x_%j.err" jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh "$COHORT" 500 40 spatial_block_kfold 0.0 0.0 "rq1_pinn_env_terrain_k_${SET_NAME}_w0_seed42" 42 256 "$SET_NAME" 0.0 42 5 "$FOLD" "" 0.0001
  done
done

for WEIGHT in 0.0 1.0 2.0; do
  WEIGHT_LABEL=$(echo "$WEIGHT" | cut -d. -f1)
  for COHORT in 4survey 6survey; do
    sbatch --job-name="rq1_physicsablation_singlesplit_pinn_${COHORT}_w${WEIGHT_LABEL}" --output="logs/rq123_methodology/rq1_physicsablation/%x_%j.out" --error="logs/rq123_methodology/rq1_physicsablation/%x_%j.err" jobs/pinn_env_terrain/run_pinn_env_terrain.sh "$COHORT" 500 40 spatial_block "$WEIGHT" "$WEIGHT" "rq1_physicsablation_singlesplit_pinn_env_terrain_${SET_NAME}_w${WEIGHT_LABEL}_seed42" 42 256 "$SET_NAME" 0.0 42 5 0 0.0001

    sbatch --job-name="rq1_physicsablation_singlesplit_pinnk_${COHORT}_w${WEIGHT_LABEL}" --output="logs/rq123_methodology/rq1_physicsablation/%x_%j.out" --error="logs/rq123_methodology/rq1_physicsablation/%x_%j.err" jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh "$COHORT" 500 40 spatial_block "$WEIGHT" "$WEIGHT" "rq1_physicsablation_singlesplit_pinn_env_terrain_k_${SET_NAME}_w${WEIGHT_LABEL}_seed42" 42 256 "$SET_NAME" 0.0 42 5 0 "" 0.0001
  done
done

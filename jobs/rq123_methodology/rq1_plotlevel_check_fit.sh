#!/bin/bash
# RUN THIS ON: the cluster.
#
# RQ1 plot_level vs. spatial_block integrity check, current tier. Old-tier evidence
# (terrain_wind_solid, 2026-08-08, see documentation/research_questions_overview.md) already
# showed DNN inflates badly on the easy plot_level split (0.799 vs 0.663 spatial_block_kfold,
# a 0.136 gap) while PINN barely moves (0.589 vs 0.579, a 0.010 gap) -- this reruns the same
# check on the current VIF-screened Set3, the RQ1 winner's set.
#
# Only plot_level is new here -- the spatial_block side of the comparison already exists from
# rq1_temporalcheck_fit.sh (same models, same set, same cohorts, single split), no need to refit
# it. Same run_name convention as that script -- plot_level and spatial_block never collide since
# model_output_dir() saves plot_level with no split-type prefix folder at all (a different top-
# level path), unlike spatial_block/temporal which each get their own subtree.
#
# 3 models x 2 cohorts = 6 jobs. Single split (plot_level has no fold concept), seed 42.
set -e

SET_NAME="nested_set3_gated_terrain_wind_vif"

mkdir -p logs/rq123_methodology/rq1_plotlevel_check

for COHORT in 4survey 6survey; do
  sbatch --job-name="rq1_plotlevelcheck_dnn_${COHORT}" --output="logs/rq123_methodology/rq1_plotlevel_check/%x_%j.out" --error="logs/rq123_methodology/rq1_plotlevel_check/%x_%j.err" jobs/dnn_env_terrain/run_dnn_env_terrain.sh "$COHORT" 500 40 plot_level 42 "rq1_temporalcheck_dnn_env_terrain_${SET_NAME}_seed42" 256 "$SET_NAME" 0.0 42 5 0 0.0001

  sbatch --job-name="rq1_plotlevelcheck_pinn_${COHORT}" --output="logs/rq123_methodology/rq1_plotlevel_check/%x_%j.out" --error="logs/rq123_methodology/rq1_plotlevel_check/%x_%j.err" jobs/pinn_env_terrain/run_pinn_env_terrain.sh "$COHORT" 500 40 plot_level 1.0 1.0 "rq1_temporalcheck_pinn_env_terrain_${SET_NAME}_seed42" 42 256 "$SET_NAME" 0.0 42 5 0 0.0001

  sbatch --job-name="rq1_plotlevelcheck_pinnk_${COHORT}" --output="logs/rq123_methodology/rq1_plotlevel_check/%x_%j.out" --error="logs/rq123_methodology/rq1_plotlevel_check/%x_%j.err" jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh "$COHORT" 500 40 plot_level 1.0 1.0 "rq1_temporalcheck_pinn_env_terrain_k_${SET_NAME}_seed42" 42 256 "$SET_NAME" 0.0 42 5 0 "" 0.0001
done

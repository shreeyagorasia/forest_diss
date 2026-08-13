#!/bin/bash
# RUN THIS ON: the cluster.
#
# Extends the DNN-only architecture sweep (rq1_architecture_sweep_fit.sh) to PINN + PINN_k --
# a genuinely different question, not a repeat: PINN's physics (+ trajectory, for PINN_k) loss
# term competes with the data loss for the same weights, so a capacity conclusion drawn from DNN
# alone doesn't tell you whether PINN's combined loss landscape has a different requirement.
# Directly checks the concern "does PINN need a different architecture to work properly" instead
# of leaving it as an assumption -- see documentation/research_questions_overview.md's physics
# section for the current "constraint rigidity" narrative this could qualify.
#
# Same 4 presets, same coarse single-spatial_block-split screen, same Set3, seed 42,
# physics_weight=1.0/trajectory_weight=1.0 (matching the main RQ1 sweep's winning config, NOT the
# w=0 physics-ablation arm -- this checks capacity, not physics weight).
#
# 4 architectures x 2 models (pinn_env_terrain, pinn_env_terrain_k) x 2 cohorts = 16 jobs.
set -e

SET_NAME="nested_set3_gated_terrain_wind_vif"

mkdir -p logs/rq123_methodology/rq1_architecture_sweep

declare -A ARCHITECTURES=(
  [small]="64,32"
  [medium]="128,64"
  [large]="256,128,64"
  [deeper]="256,128,64,32"
)

for ARCH_NAME in "${!ARCHITECTURES[@]}"; do
  HIDDEN_LAYER_SIZES="${ARCHITECTURES[$ARCH_NAME]}"
  for COHORT in 4survey 6survey; do
    sbatch --job-name="rq1_arch_pinn_${ARCH_NAME}_${COHORT}" --output="logs/rq123_methodology/rq1_architecture_sweep/%x_%j.out" --error="logs/rq123_methodology/rq1_architecture_sweep/%x_%j.err" jobs/pinn_env_terrain/run_pinn_env_terrain.sh "$COHORT" 500 40 spatial_block 1.0 1.0 "rq1_pinn_env_terrain_${SET_NAME}_arch${ARCH_NAME}" 42 256 "$SET_NAME" 0.0 42 5 0 0.0001 "$HIDDEN_LAYER_SIZES"

    sbatch --job-name="rq1_arch_pinnk_${ARCH_NAME}_${COHORT}" --output="logs/rq123_methodology/rq1_architecture_sweep/%x_%j.out" --error="logs/rq123_methodology/rq1_architecture_sweep/%x_%j.err" jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh "$COHORT" 500 40 spatial_block 1.0 1.0 "rq1_pinn_env_terrain_k_${SET_NAME}_arch${ARCH_NAME}" 42 256 "$SET_NAME" 0.0 42 5 0 "" 0.0001 "$HIDDEN_LAYER_SIZES"
  done
done

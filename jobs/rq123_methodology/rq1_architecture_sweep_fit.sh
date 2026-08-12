#!/bin/bash
# RUN THIS ON: the cluster.
#
# RQ1 architecture-size sweep, extended to the actual current winner (DNN + Set3,
# nested_set3_gated_terrain_wind_vif) -- the 2026-08-02 architecture sweep
# (documentation/experiment_log.md) found capacity wasn't the bottleneck, but ONLY tested
# dnn_noenv/pinn_noenv (no environmental conditioning, old feature-set convention, incomplete:
# 7/8 planned runs, no 6survey) -- never directly checked on the current _env_terrain models this
# dissertation actually cites. This closes that gap with real evidence instead of leaning on the
# old, only-partially-matching result.
#
# Reuses the exact 4 named presets from the 2026-08-02 sweep (same --hidden-layer-sizes flag,
# already wired into run_dnn_env_terrain.sh): small=64,32 / medium=128,64 / large=256,128,64 /
# deeper=256,128,64,32. The 3x128 baseline itself is not re-run here -- it's already the primary
# RQ1 sweep's own dnn_env_terrain result (TEMP_rq1_sweep_results_2026-08-11.tex), so comparing
# against it needs no new job.
#
# Single spatial_block split per architecture x cohort (not the full 5-fold kfold) -- matches the
# 2026-08-02 sweep's own "coarse screen" convention (documentation/experiment_log.md, 2026-08-06
# entry: "single spatial_block split as a coarse screen matching the 2026-08-02 sweep's own
# convention"). If any architecture materially beats the baseline here, THEN it's worth a full
# 5-fold rerun -- not before, to keep this cheap.
#
# 4 architectures x 2 cohorts = 8 jobs. DNN trains fast (seconds per epoch on this project's own
# past runs), so 500 max-epochs is still cheap even x8.
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
    sbatch --job-name="rq1_arch_${ARCH_NAME}_${COHORT}" --output="logs/rq123_methodology/rq1_architecture_sweep/%x_%j.out" --error="logs/rq123_methodology/rq1_architecture_sweep/%x_%j.err" jobs/dnn_env_terrain/run_dnn_env_terrain.sh "$COHORT" 500 40 spatial_block 42 "rq1_dnn_env_terrain_${SET_NAME}_arch${ARCH_NAME}" 256 "$SET_NAME" 0.0 42 5 0 0.0001 "$HIDDEN_LAYER_SIZES"
  done
done

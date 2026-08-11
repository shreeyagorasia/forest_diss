#!/bin/bash
# RUN THIS ON: the cluster.
#
# RQ1 winner reseed -- DNN + Set3 (nested_set3_gated_terrain_wind_vif) picked as the RQ1 winner
# 2026-08-11 (see documentation/research_questions_overview.md and
# TEMP_results/TEMP_rq1_sweep_results_2026-08-11.tex for the full reasoning: non-overlapping 95%
# CIs vs. every PINN variant on 4survey, statistically tied -- not beaten -- on 6survey, confirmed
# by RMSE/MAE/per-fold stability too). Seed 42 already exists from the main sweep; this adds 4
# more seeds (43-46) to robustness-check that the win isn't a lucky single seed.
#
# The physics-weight reseed + zero-physics-control reseed originally planned alongside this
# (Step 4's DNN-wins case) is deliberately NOT included here -- the physics-weight ablation
# already run at Set3 (rq1_physicsablation_fit.sh, 32 jobs) gives a clean, monotonic,
# two-tier-confirmed answer to "does physics help at the winning set" without needing a further
# 5-seed reseed of that specific question.
#
# IMPORTANT: SEED (arg 5, the model's own weight-init/training randomness) is what varies here.
# SPLIT_SEED (arg 10, which fold assignment is used) stays FIXED at 42, identical to the original
# sweep -- each reseed must be tested against the exact same train/test split, so only the
# model's own stochastic training differs, not the data split too (that's already covered by the
# 5-fold CV itself).
#
# 4 seeds x 2 cohorts x 5 folds = 40 jobs. DNN trains fast (the main sweep's smoke test finished
# in seconds per epoch), so this is cheap even at 500 max-epochs.
set -e

SET_NAME="nested_set3_gated_terrain_wind_vif"

mkdir -p logs/rq123_methodology/rq1_dnn_reseed

for SEED in 43 44 45 46; do
  for COHORT in 4survey 6survey; do
    for FOLD in 0 1 2 3 4; do
      sbatch --job-name="rq1_dnn_reseed_${COHORT}_seed${SEED}_fold${FOLD}" --output="logs/rq123_methodology/rq1_dnn_reseed/%x_%j.out" --error="logs/rq123_methodology/rq1_dnn_reseed/%x_%j.err" jobs/dnn_env_terrain/run_dnn_env_terrain.sh "$COHORT" 500 40 spatial_block_kfold "$SEED" "rq1_dnn_env_terrain_${SET_NAME}_seed${SEED}" 256 "$SET_NAME" 0.0 42 5 "$FOLD" 0.0001
    done
  done
done

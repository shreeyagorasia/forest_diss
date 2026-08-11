#!/bin/bash
# RUN THIS ON: the cluster.
#
# Step 3 -- RQ1 env-conditioned tier sweep FIT. 3 models x 3 sets x 2 cohorts x 5 folds = 90
# jobs. GPU, real settings (500 max-epochs, patience 40, batch_size 256 -- this project's
# established real-run defaults, not the smoke test's cheap 5-epoch check). Single seed (42) --
# this is a comparison sweep to pick a winner, not the final robustness-checked number; Step 4
# reseeds only the winner. Same scale as the old E6 sweep (also 90 jobs) -- known, already-
# survived cluster load.
#
# Naming: rq1_<model>_<set_name>_seed42 -- distinct from every existing name in outputs/.
#
# Logs (2026-08-10): the underlying per-model scripts (jobs/dnn_env_terrain/run_dnn_env_terrain.sh
# etc.) hardcode #SBATCH --output=logs/<model_family>/%x_%j.out -- a single flat folder shared by
# EVERY experiment that has ever used that model (E6, E10, physics-weight sweeps, now this sweep
# too), which is why that folder gets polluted with runs that no longer apply. Rather than edit
# those shared scripts (which would redirect every OTHER experiment's logs too, not just this
# one), sbatch's own --job-name/--output/--error flags on the command line override the script's
# #SBATCH directives for just this submission -- so only Step 3's 90 jobs land in a new,
# RQ123-specific folder, one subfolder per model, and the shared scripts are untouched.
set -e

SETS="nested_set2_top10 nested_set3_gated_terrain_wind_vif nested_set4_gated_all_vif"
COHORTS="4survey 6survey"
SEED=42

mkdir -p logs/rq123_methodology/rq1_dnn_env_terrain logs/rq123_methodology/rq1_pinn_env_terrain logs/rq123_methodology/rq1_pinn_env_terrain_k

for SET_NAME in $SETS; do
  for COHORT in $COHORTS; do
    for FOLD in 0 1 2 3 4; do
      RUN_NAME_DNN="rq1_dnn_env_terrain_${SET_NAME}_seed${SEED}"
      sbatch --job-name="${RUN_NAME_DNN}_${COHORT}_fold${FOLD}" \
        --output="logs/rq123_methodology/rq1_dnn_env_terrain/%x_%j.out" \
        --error="logs/rq123_methodology/rq1_dnn_env_terrain/%x_%j.err" \
        jobs/dnn_env_terrain/run_dnn_env_terrain.sh \
        "$COHORT" 500 40 spatial_block_kfold "$SEED" "$RUN_NAME_DNN" 256 "$SET_NAME" 0.0 42 5 "$FOLD" 0.0001

      RUN_NAME_PINN="rq1_pinn_env_terrain_${SET_NAME}_seed${SEED}"
      sbatch --job-name="${RUN_NAME_PINN}_${COHORT}_fold${FOLD}" \
        --output="logs/rq123_methodology/rq1_pinn_env_terrain/%x_%j.out" \
        --error="logs/rq123_methodology/rq1_pinn_env_terrain/%x_%j.err" \
        jobs/pinn_env_terrain/run_pinn_env_terrain.sh \
        "$COHORT" 500 40 spatial_block_kfold 1.0 1.0 "$RUN_NAME_PINN" "$SEED" 256 "$SET_NAME" 0.0 42 5 "$FOLD" 0.0001

      RUN_NAME_PINNK="rq1_pinn_env_terrain_k_${SET_NAME}_seed${SEED}"
      sbatch --job-name="${RUN_NAME_PINNK}_${COHORT}_fold${FOLD}" \
        --output="logs/rq123_methodology/rq1_pinn_env_terrain_k/%x_%j.out" \
        --error="logs/rq123_methodology/rq1_pinn_env_terrain_k/%x_%j.err" \
        jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh \
        "$COHORT" 500 40 spatial_block_kfold 1.0 1.0 "$RUN_NAME_PINNK" "$SEED" 256 "$SET_NAME" 0.0 42 5 "$FOLD" "" 0.0001
    done
  done
done

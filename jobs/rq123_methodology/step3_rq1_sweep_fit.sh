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
set -e

SETS="nested_set2_top10 nested_set3_gated_terrain_wind_vif nested_set4_gated_all_vif"
COHORTS="4survey 6survey"
SEED=42

for SET_NAME in $SETS; do
  for COHORT in $COHORTS; do
    for FOLD in 0 1 2 3 4; do
      RUN_NAME_DNN="rq1_dnn_env_terrain_${SET_NAME}_seed${SEED}"
      sbatch jobs/dnn_env_terrain/run_dnn_env_terrain.sh \
        "$COHORT" 500 40 spatial_block_kfold "$SEED" "$RUN_NAME_DNN" 256 "$SET_NAME" 0.0 42 5 "$FOLD" 0.0001

      RUN_NAME_PINN="rq1_pinn_env_terrain_${SET_NAME}_seed${SEED}"
      sbatch jobs/pinn_env_terrain/run_pinn_env_terrain.sh \
        "$COHORT" 500 40 spatial_block_kfold 1.0 1.0 "$RUN_NAME_PINN" "$SEED" 256 "$SET_NAME" 0.0 42 5 "$FOLD" 0.0001

      RUN_NAME_PINNK="rq1_pinn_env_terrain_k_${SET_NAME}_seed${SEED}"
      sbatch jobs/pinn_env_terrain_k/run_pinn_env_terrain_k.sh \
        "$COHORT" 500 40 spatial_block_kfold 1.0 1.0 "$RUN_NAME_PINNK" "$SEED" 256 "$SET_NAME" 0.0 42 5 "$FOLD" "" 0.0001
    done
  done
done

#!/bin/bash
# Step 2: the 30 evaluate jobs, run AFTER all 30 fit jobs from
# rerun_env_terrain_batchfix_5seed.sh have finished.
# Run once: bash jobs/evaluate_env_terrain_batchfix_5seed.sh
set -e

for cohort in 4survey 6survey; do
  for seed in 42 43 44 45 46; do
    run_name="env_terrain_batchfix_seed${seed}"

    sbatch jobs/dnn_env_terrain/evaluate_dnn_env_terrain.sh \
      "$cohort" spatial_block "dnn_${run_name}"

    sbatch jobs/pinn_env_terrain/evaluate_pinn_env_terrain.sh \
      "$cohort" spatial_block "pinn_${run_name}"

    sbatch jobs/pinn_env_terrain_k/evaluate_pinn_env_terrain_k.sh \
      "$cohort" spatial_block "pinnk_${run_name}"
  done
done

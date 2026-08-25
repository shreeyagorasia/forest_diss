#!/bin/bash
# Step 2: the 20 evaluate jobs, run AFTER all 20 fit jobs from rerun_pinn_w1_anchorfix.sh
# have finished. Run once: bash jobs/evaluate_pinn_w1_anchorfix.sh
set -e

for cohort in 4survey 6survey; do
  for split in spatial_block temporal; do
    for seed in 42 43 44 45 46; do
      sbatch jobs/pinn_noenv/evaluate_pinn_noenv.sh "$cohort" "$split" "final_pinn_w1_anchorfix_seed${seed}"
    done
  done
done

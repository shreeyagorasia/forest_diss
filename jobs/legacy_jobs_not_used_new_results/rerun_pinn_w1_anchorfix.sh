#!/bin/bash
# Step 1: fit jobs. Run on the cluster: bash jobs/rerun_pinn_w1_anchorfix.sh
# Purpose: pinn_noenv(w=1), corrected split-matched anchor, 5 seeds x 2 splits x 2 cohorts.
# Fills the last remaining gap in the no-env DNN/PINN(w=1) comparison -- DNN, the tuned arm,
# and the zero-physics arm all already have 5-seed evidence (Stage 4); w=1 never did.
# After all 20 of these finish, run jobs/evaluate_pinn_w1_anchorfix.sh.
set -e

for cohort in 4survey 6survey; do
  for split in spatial_block temporal; do
    for seed in 42 43 44 45 46; do
      sbatch jobs/pinn_noenv/run_pinn_noenv.sh "$cohort" 500 40 "$split" 1.0 1.0 "final_pinn_w1_anchorfix_seed${seed}" "$seed" 256
    done
  done
done

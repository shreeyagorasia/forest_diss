#!/bin/bash
#
# Pools E6_stage_sweep's 5-fold results for all 18 (tier, model, cohort) combinations into
# whole-population scores, via models/common/kfold_summary.py. A real file instead of a
# copy-pasted heredoc, so there's nothing for a terminal to mangle mid-paste.
#
# Run on the cluster, from the project root, AFTER both "E6_stage_sweep fit" and
# "E6_stage_sweep evaluate" have fully finished (check squeue -u $USER is empty of those jobs
# first -- this script does not wait for them, it assumes they're already done):
#
#   bash jobs/pool_e6_stage_sweep.sh
#
# This is cheap, CPU-only, plain Python -- no sbatch/GPU needed, safe to run directly.

set -e

TIERS="stage1_terrain stage2_terrain_wind stage4_all_environmental"
MODELS="dnn_env_terrain pinn_env_terrain pinn_env_terrain_k"
COHORTS="4survey 6survey"

for tier in $TIERS; do
  for model in $MODELS; do
    for cohort in $COHORTS; do
      python -m models.common.kfold_summary --model-name "${tier}_${model}" --cohort "$cohort"
    done
  done
done

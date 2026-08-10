#!/bin/bash
# RUN THIS ON: your Mac, after step3_rq1_sweep_sync.sh finishes.
#
# Step 3 -- RQ1 sweep EVALUATE. Evaluates all 9 (model, set) combinations across all 5 folds
# (--cohort omitted each call -- evaluate_*.py already does both cohorts in one call), then
# pools each into a real 5-fold summary via models/common/kfold_summary.py (the same tool this
# project's own headline DNN/PINN numbers are already pooled with) and prints one readable table.
#
# READ THE PRINTED SUMMARY to pick the winner by hand (highest pooled R2, averaged across both
# cohorts) -- deliberately not auto-selected, so you can sanity-check it before committing
# cluster time to Step 4's reseed.
set -e

SETS="nested_set2_top10 nested_set3_gated_terrain_wind_vif nested_set4_gated_all_vif"

echo "===== Evaluating all 9 (model, set) combinations, 5 folds each ====="
for MODEL in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for SET_NAME in $SETS; do
    RUN_NAME="rq1_${MODEL}_${SET_NAME}_seed42"
    for FOLD in 0 1 2 3 4; do
      .venv/bin/python -m "models.${MODEL}.evaluate_${MODEL}" \
        --split-type spatial_block_kfold --run-name "$RUN_NAME" \
        --split-seed 42 --n-folds 5 --fold-index "$FOLD"
    done
  done
done

echo ""
echo "===== Pooled 5-fold summary, every (model, set) combination -- READ THIS TO PICK THE WINNER ====="
for MODEL in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for SET_NAME in $SETS; do
    RUN_NAME="rq1_${MODEL}_${SET_NAME}_seed42"
    .venv/bin/python -m models.common.kfold_summary --model-name "$RUN_NAME" --n-folds 5
  done
done

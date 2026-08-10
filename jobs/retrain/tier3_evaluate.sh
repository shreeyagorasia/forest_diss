#!/bin/bash
# RUN THIS ON: your Mac, after tier3_sync.sh finishes.
source "$(dirname "$0")/_lib.sh"

for model in dnn_noenv pinn_noenv; do
  for seed in $SEEDS; do
    for fold in 0 1 2 3 4; do
      local_evaluate_noenv "$model" spatial_block_kfold "$seed" "$fold"
    done
  done
done

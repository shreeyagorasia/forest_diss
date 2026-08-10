#!/bin/bash
# RUN THIS ON: your Mac, after tier4_sync.sh finishes.
source "$(dirname "$0")/_lib.sh"

for model in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for seed in $SEEDS; do
    for fold in 0 1 2 3 4; do
      local_evaluate_env_terrain "$model" spatial_block_kfold set2 "$seed" "$fold"
    done
  done
done

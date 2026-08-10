#!/bin/bash
# RUN THIS ON: your Mac, after tier2_sync.sh finishes.
source "$(dirname "$0")/_lib.sh"

for model in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for seed in $SEEDS; do
    local_evaluate_env_terrain "$model" spatial_block set2 "$seed"
  done
done

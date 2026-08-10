#!/bin/bash
# RUN THIS ON: your Mac, after tier7_sync.sh finishes.
source "$(dirname "$0")/_lib.sh"

for model in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for seed in $SEEDS; do
    for tag in set3a set3b set4; do
      for split in plot_level spatial_block temporal; do
        local_evaluate_env_terrain "$model" "$split" "$tag" "$seed"
      done
      for fold in 0 1 2 3 4; do
        local_evaluate_env_terrain "$model" spatial_block_kfold "$tag" "$seed" "$fold"
      done
    done
  done
done

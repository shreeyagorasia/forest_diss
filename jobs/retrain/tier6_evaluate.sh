#!/bin/bash
# RUN THIS ON: your Mac, after tier6_sync.sh finishes.
source "$(dirname "$0")/_lib.sh"

for model in dnn_noenv pinn_noenv; do
  for seed in $SEEDS; do
    local_evaluate_noenv "$model" plot_level "$seed"
  done
done

for model in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  for seed in $SEEDS; do
    local_evaluate_env_terrain "$model" plot_level set2 "$seed"
  done
done

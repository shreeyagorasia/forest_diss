#!/bin/bash
# RUN THIS ON: your Mac, after tier0_sync.sh finishes. --cohort is omitted below on purpose --
# every local_evaluate_* call runs BOTH cohorts in one go (see models/*/evaluate_*.py's own
# --cohort flag: "Omit to run both cohorts.").
source "$(dirname "$0")/_lib.sh"

for model in dnn_env_terrain pinn_env_terrain pinn_env_terrain_k; do
  local_evaluate_env_terrain "$model" plot_level set4 42
done

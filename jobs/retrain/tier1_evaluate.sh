#!/bin/bash
# RUN THIS ON: your Mac. No cluster interaction -- baselines never leave your machine.
# Tier 1 -- EVALUATE. Run after tier1_fit.sh finishes (baselines fit in seconds, so there's
# barely any wait).
source "$(dirname "$0")/_lib.sh"

for split in plot_level spatial_block temporal; do
  local_evaluate_baselines_for_split "$split"
done
local_evaluate_baselines_for_split "spatial_block_kfold"

#!/bin/bash
# RUN THIS ON: your Mac (needs .venv/).
# Tier 1 -- FIT. Baselines, every split incl. all 5 kfold folds (16 jobs total, 8 fit here).
# The ONE tier that's entirely local -- CR/average-by-age/Linear/RF fit in seconds on CPU, no
# GPU or cluster job needed. After this finishes, run tier1_push.sh (a separate, fully explicit
# rsync upload) to send the resulting CR anchor up to the cluster -- every PINN fit job in every
# other tier reads it from wherever IT runs.
source "$(dirname "$0")/_lib.sh"

for split in plot_level spatial_block temporal; do
  local_fit_baselines_for_split "$split"
done
local_fit_baselines_for_split "spatial_block_kfold"

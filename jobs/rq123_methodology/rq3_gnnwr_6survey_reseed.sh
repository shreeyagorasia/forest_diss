#!/bin/bash
# RUN THIS ON: the cluster.
#
# GNNWR 6survey reseed -- 6survey's per-fold R2 SD is large (0.174-0.224 across the 3 sets,
# TEMP_results/TEMP_rq3_gnnwr_results_2026-08-11.tex) with negative pooled R2 for Set2/Set4.
# This checks whether "GNNWR loses on 6survey" is a stable finding or an artefact of one
# unstable training run -- 4survey is NOT included here, its result is already consistent enough
# not to need this. 2 new seeds (43, 44) x 3 sets x 5 folds = 30 jobs.
#
# GNNWR's own run_name already includes a fold label but not a seed label (see
# jobs/growth_curve_attribution/run_rq3_gnnwr.sh / gnnwr_check.py's run_gnnwr()) -- SPLIT_SEED
# (arg 6) is what actually varies the fold assignment for GNNWR, unlike RQ1's DNN/PINN where
# SEED and SPLIT_SEED are separate knobs. Varying SPLIT_SEED here means each reseed also gets a
# genuinely different fold assignment, not just different model-internal randomness -- the
# closest equivalent GNNWR has to a "seed" reseed, given its own script doesn't expose a separate
# training-randomness seed.
set -e

mkdir -p logs/rq123_methodology/rq3_gnnwr_6survey_reseed

for SET_NAME in nested_set2_top10 nested_set3_gated_terrain_wind_vif nested_set4_gated_all_vif; do
  for SPLIT_SEED in 43 44; do
    for FOLD in 0 1 2 3 4; do
      sbatch --job-name="rq3_gnnwr_6sreseed_${SET_NAME}_seed${SPLIT_SEED}_fold${FOLD}" --output="logs/rq123_methodology/rq3_gnnwr_6survey_reseed/%x_%j.out" --error="logs/rq123_methodology/rq3_gnnwr_6survey_reseed/%x_%j.err" jobs/growth_curve_attribution/run_rq3_gnnwr.sh 6survey "$SET_NAME" 200 20 0 "$SPLIT_SEED" "$FOLD" 5
    done
  done
done

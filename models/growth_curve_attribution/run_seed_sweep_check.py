# Run as: python -m models.growth_curve_attribution.run_seed_sweep_check
#
# Checks whether 6survey's near-zero/negative plot-level R2 (found in the 2026-08-03
# scale-comparison check, seed=42 only) is a real, stable result or a seed artefact. This
# project already found one real precedent for exactly this trap (2026-08-02 split-seed
# robustness entries: "terrain hurts the DNN" held only under seed 42, reversed under other
# seeds). Plot-level only. Subcompartment is already settled as dead-on-arrival (see the
# 2026-08-03 scale-comparison entry), no need to re-check it here.

import pandas as pd

from models.common.splits import SPLIT_SEED
from models.growth_curve_attribution.scale_comparison_check import run_for_cohort

COHORTS = ["4survey", "6survey"]
SEEDS_TO_CHECK = [SPLIT_SEED, 43, 44, 45]


def main():
    all_results = []
    for cohort in COHORTS:
        for seed in SEEDS_TO_CHECK:
            all_results.append(run_for_cohort(cohort, split_seed=seed, scales=("plot",)))

    results = pd.concat(all_results, ignore_index=True)
    pd.set_option("display.width", 160)
    print()
    print(results[["cohort", "split_seed", "method", "n_train", "n_val", "n_test", "r2", "rmse", "bias"]].to_string(index=False))


if __name__ == "__main__":
    main()

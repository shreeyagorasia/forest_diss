# Run as: python -m models.growth_curve_attribution.run_scale_comparison_check
#
# Answers: for the growth-curve attribution work, is per-plot the right unit of analysis, or
# should plots be aggregated to the subcompartment level first? See
# scale_comparison_check.py's own header for the full reasoning.

import pandas as pd

from models.growth_curve_attribution.scale_comparison_check import run_for_cohort

COHORTS = ["4survey", "6survey"]


def main():
    all_results = []
    for cohort in COHORTS:
        all_results.append(run_for_cohort(cohort))

    results = pd.concat(all_results, ignore_index=True)
    pd.set_option("display.width", 160)
    print()
    print(results[["cohort", "scale", "method", "n_train", "n_val", "r2", "rmse", "mae", "bias"]].to_string(index=False))


if __name__ == "__main__":
    main()

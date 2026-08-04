# Run as: python -m models.growth_curve_attribution.run_disturbance_cleaning_comparison
#
# Compares the plot-level attribution result BEFORE and AFTER excluding clearfell-like and
# measurement-inconsistent plots from the y_max curve fit (see disturbance_checks.py's
# summarize_plot_disturbance_status() and scale_comparison_check.py's build_plot_level_table()).
# Answers: does the data-quality cleaning actually change the reported R2, or was the uncleaned
# number already close to the truth? Plot-level only -- subcompartment is already dead-on-arrival
# regardless of cleaning.

import pandas as pd

from models.growth_curve_attribution.scale_comparison_check import run_for_cohort

COHORTS = ["4survey", "6survey"]


def main():
    all_results = []
    for cohort in COHORTS:
        for apply_cleaning in [False, True]:
            label = "cleaned" if apply_cleaning else "uncleaned"
            print(f"\n===== {cohort}, {label} =====")
            result = run_for_cohort(cohort, scales=("plot",), apply_disturbance_cleaning=apply_cleaning)
            result["cleaning"] = label
            all_results.append(result)

    results = pd.concat(all_results, ignore_index=True)
    pd.set_option("display.width", 160)
    print()
    print(results[["cohort", "cleaning", "method", "n_train", "n_val", "r2", "rmse", "bias"]].to_string(index=False))


if __name__ == "__main__":
    main()

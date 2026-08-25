# Run as: python -m models.growth_curve_attribution.run_disturbance_checks
#
# Prints the two disturbance checks agreed before touching any external data source. See
# models/growth_curve_attribution/disturbance_checks.py's own header for what each one tests.

from models.growth_curve_attribution.disturbance_checks import (
    fit_all_years_and_compute_residuals,
    per_plot_residual_range,
    residual_by_survey_year,
    thinning_during_gap_bias_check,
)

COHORTS = ["4survey", "6survey"]


def run_check_1(cohort):
    df_with_residuals = fit_all_years_and_compute_residuals(cohort)

    print("\n-- Check 1a: residual by survey year (population-level) --")
    print(residual_by_survey_year(df_with_residuals))

    print("\n-- Check 1b: per-plot residual range (max minus min across own years) --")
    residual_range = per_plot_residual_range(df_with_residuals)
    print(residual_range["residual_range"].describe())


def run_check_2(cohort):
    print("\n-- Check 2: residual at held-out years, grouped by thinned-during-gap --")
    results = thinning_during_gap_bias_check(cohort)
    for evaluation_year, table in results.items():
        print(f"\n  Held-out year {evaluation_year}:")
        print(table)


def main():
    for cohort in COHORTS:
        print(f"\n===== {cohort} =====")
        run_check_1(cohort)
        run_check_2(cohort)


if __name__ == "__main__":
    main()

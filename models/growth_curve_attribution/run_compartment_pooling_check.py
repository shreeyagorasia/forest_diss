# Run as: python -m models.growth_curve_attribution.run_compartment_pooling_check
#
# Prints the compartment-variance (ICC) check agreed before building any compartment-pooled
# growth-curve fit -- see models/growth_curve_attribution/compartment_pooling_check.py's own
# header for why this needs checking first.

from models.growth_curve_attribution.compartment_pooling_check import (
    compartment_variance_check,
    compartment_variance_check_by_age_band,
)
from models.growth_curve_attribution.data import load_filtered_growth_curve_table
from models.growth_curve_attribution.phase0_checks import yldc_deviation_summary

COHORTS = ["4survey", "6survey"]


def run_for_cohort(cohort):
    print(f"\n===== {cohort} =====")
    df = load_filtered_growth_curve_table(cohort)

    # yldc_deviation_summary() already builds the yldc_deviation column and drops rows with no
    # valid p1-p5/yldc combination -- reused here, not recomputed, so this check reads the exact
    # same deviation values the Phase 0 spread check already reported on.
    valid, _, _ = yldc_deviation_summary(df)

    print("\n-- Overall ICC: how much of yldc_deviation's variance sits BETWEEN compartments --")
    overall = compartment_variance_check(valid)
    for key, value in overall.items():
        print(f"  {key}: {value}")

    print("\n-- ICC by 5-year age band --")
    by_age_band = compartment_variance_check_by_age_band(valid)
    for age_band, result in sorted(by_age_band.items()):
        print(f"  age_band={age_band}: n_rows={result['n_rows']:,}, n_groups={result['n_groups']}, icc={result['icc']:.4f}")


def main():
    for cohort in COHORTS:
        run_for_cohort(cohort)


if __name__ == "__main__":
    main()

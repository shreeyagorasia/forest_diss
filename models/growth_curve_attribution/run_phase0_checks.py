# Run as: python -m models.growth_curve_attribution.run_phase0_checks
#
# Prints the Phase 0 checks agreed in documentation/experiment_log.md's Stage 2 planning notes --
# cheap, no-model-fitting diagnostics that gate which candidate architecture (per-plot fixed-
# shape fit, Bayesian hierarchical, GNNWR/GRF-style hypernetwork) is worth building, before any
# of them get real engineering investment.

from models.growth_curve_attribution.data import load_filtered_growth_curve_table
from models.growth_curve_attribution.phase0_checks import (
    distinct_timestamp_counts,
    neighbour_coverage_check,
    thinning_confound_check,
    yldc_deviation_summary,
)

COHORTS = ["4survey", "6survey"]


def run_for_cohort(cohort):
    print(f"\n===== {cohort} =====")
    df = load_filtered_growth_curve_table(cohort)
    print(f"Filtered rows: {len(df):,}, plots: {df['identification'].nunique():,}")

    print("\n-- 1. Distinct survey-year count per plot --")
    counts_per_plot, summary = distinct_timestamp_counts(df)
    print(summary)

    print("\n-- 2. yldc deviation distribution --")
    valid, overall_stats, by_age_band = yldc_deviation_summary(df)
    for key, value in overall_stats.items():
        print(f"  {key}: {value}")
    print("  By 5-year age band:")
    print(by_age_band)

    print("\n-- 3. Thinning confound check --")
    print(thinning_confound_check(valid))

    print("\n-- 4. Neighbour coverage under spatial_block_split (75m radius) --")
    coverage = neighbour_coverage_check(df)
    for key, value in coverage.items():
        print(f"  {key}: {value}")


def main():
    for cohort in COHORTS:
        run_for_cohort(cohort)


if __name__ == "__main__":
    main()

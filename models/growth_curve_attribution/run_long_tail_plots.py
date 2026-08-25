# Run as: python -m models.growth_curve_attribution.run_long_tail_plots
#
# Pulls out the actual top 1-2% of plots by residual range (problem 2 in
# documentation/model_instructions/growth_curve_stage2_handover.md) and prints each one's own
# real Age/height/thinning trajectory, so a real exclude/flag/investigate decision can be made
# from the data itself rather than just the summary statistic.

from models.growth_curve_attribution.disturbance_checks import fit_all_years_and_compute_residuals
from models.growth_curve_attribution.long_tail_plots import (
    flag_single_year_outlier,
    get_flagged_plot_trajectories,
    identify_long_tail_plots,
)

COHORTS = ["4survey", "6survey"]


def run_for_cohort(cohort, top_fraction=0.02, n_to_print_in_full=15):
    print(f"\n===== {cohort} =====")
    df_with_residuals = fit_all_years_and_compute_residuals(cohort)

    flagged = identify_long_tail_plots(df_with_residuals, top_fraction=top_fraction)
    print(f"\nFlagged {len(flagged)} plots total (top {top_fraction:.0%} by residual range).")
    print(f"residual_range summary across all {len(flagged)} flagged plots:")
    print(flagged["residual_range"].describe())

    trajectories = get_flagged_plot_trajectories(df_with_residuals, flagged["identification"])
    outlier_flags = flag_single_year_outlier(trajectories)

    n_single_year = outlier_flags["single_year_outlier"].sum()
    print(
        f"\n{n_single_year} of {len(outlier_flags)} flagged plots ({n_single_year / len(outlier_flags):.1%}) "
        "look like a single bad survey year (one year's |residual| >= 3x the median of that plot's other years)."
    )

    # Does the long tail cluster in a handful of compartments, or spread evenly? Relevant to
    # whether this is a few bad compartments (boundary/clearfell issues concentrated spatially)
    # or scattered plot-by-plot noise.
    flagged_with_cpmt = trajectories.drop_duplicates("identification")[["identification", "cpmt"]]
    plots_per_cpmt_among_flagged = flagged_with_cpmt.groupby("cpmt").size().sort_values(ascending=False)
    print(f"\nFlagged plots span {len(plots_per_cpmt_among_flagged)} distinct compartments.")
    print("Top 10 compartments by count of flagged plots:")
    print(plots_per_cpmt_among_flagged.head(10))

    # Full row-level trajectories only for the single worst n_to_print_in_full plots. Enough to
    # actually eyeball real Age/height/thinning patterns without dumping thousands of rows.
    worst_plot_ids = flagged.head(n_to_print_in_full)["identification"]
    worst_trajectories = trajectories[trajectories["identification"].isin(worst_plot_ids)]
    print(f"\nFull trajectories for the {n_to_print_in_full} single worst plots:")
    print(worst_trajectories.to_string(index=False))

    print(f"\nSingle-year-outlier check for those same {n_to_print_in_full} plots:")
    worst_outlier_flags = outlier_flags[outlier_flags["identification"].isin(worst_plot_ids)]
    print(worst_outlier_flags.to_string(index=False))


def main():
    for cohort in COHORTS:
        run_for_cohort(cohort)


if __name__ == "__main__":
    main()

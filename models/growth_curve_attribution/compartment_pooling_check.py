# Purpose: before building any compartment-pooled ("borrow strength across a compartment's own
# plots") version of the growth-curve fit, check whether compartments even explain a meaningful
# share of the yldc-deviation's variance in the first place. If they don't, pooling plots within
# a compartment cannot fix the long-tail noise problem (problem 2 in
# documentation/model_instructions/growth_curve_stage2_handover.md) -- there would be nothing
# real to borrow strength from, and the cheaper fix is just excluding the flagged long-tail plots.
#
# This computes the intraclass correlation coefficient (ICC) of yldc_deviation grouped by cpmt
# (forestry compartment) -- the standard "how much of the total variance sits BETWEEN groups,
# versus WITHIN a group" measure, same idea as a one-way ANOVA. ICC close to 0 means compartment
# membership barely matters (plots inside the same compartment are just as different from each
# other as plots in different compartments) -- ICC close to 1 means plots in the same compartment
# are much more alike than plots in different compartments, which is exactly the condition needed
# for compartment-pooling to help.
#
# Important: this is checked on yldc_deviation directly (Stage 2's own target), NOT copied from
# the ~5% compartment-variance number already found for mean_cr_residual (Stage 1's pooled-CR
# target, a different quantity, computed a different way) -- that number does not automatically
# transfer to this target, per this project's own standing rule to re-check every number against
# its own data rather than reasoning by category/narrative similarity.

import numpy as np
import pandas as pd


def compute_icc_one_way(values, group_ids):
    # values: one number per row (here, yldc_deviation). group_ids: which compartment each row
    # belongs to. Returns a dict with the variance decomposition and the final ICC.
    #
    # This is the standard one-way random-effects ICC formula for UNEQUAL group sizes (compartments
    # range from 1 to 1,336 plots each -- see documentation/experiment_log.md's Stage 2 section --
    # so the equal-group-size shortcut formula would not apply here).
    working = pd.DataFrame({"value": values, "group": group_ids}).dropna()

    grand_mean = working["value"].mean()
    n_total = len(working)

    group_stats = working.groupby("group")["value"].agg(["count", "mean"])
    n_groups = len(group_stats)

    # Sum of squares BETWEEN groups: how far each group's own mean is from the grand mean,
    # weighted by how many rows that group has.
    ss_between = (group_stats["count"] * (group_stats["mean"] - grand_mean) ** 2).sum()

    # Sum of squares WITHIN groups: how far each row is from its OWN group's mean.
    working = working.merge(group_stats["mean"].rename("group_mean"), on="group")
    ss_within = ((working["value"] - working["group_mean"]) ** 2).sum()

    df_between = n_groups - 1
    df_within = n_total - n_groups

    mean_square_between = ss_between / df_between
    mean_square_within = ss_within / df_within

    # n0: the "average group size" correction needed when group sizes are unequal (Fisher's
    # method). If every group had exactly the same size, n0 would just equal that size.
    sum_of_squared_group_sizes = (group_stats["count"] ** 2).sum()
    n0 = (n_total - sum_of_squared_group_sizes / n_total) / df_between

    # The between-group variance can come out slightly negative from this formula when the real
    # value is close to zero (a known property of this estimator, not a bug) -- clipped to 0,
    # since a variance can never actually be negative.
    variance_between = max((mean_square_between - mean_square_within) / n0, 0.0)
    variance_within = mean_square_within

    icc = variance_between / (variance_between + variance_within)

    return {
        "n_rows": n_total,
        "n_groups": n_groups,
        "variance_between_compartments": variance_between,
        "variance_within_compartments": variance_within,
        "icc": icc,
    }


def collapse_to_one_row_per_plot(df_with_deviation):
    # df_with_deviation has one row per PLOT-YEAR (a plot contributes 4 or 6 rows, one per survey
    # year -- a genuinely balanced panel, per the Phase 0 check). Those same-plot rows are
    # repeated measurements of the same physical trees at the same location, not independent
    # observations -- pooling them directly into a compartment-level ICC would conflate real
    # between-PLOT variance (what compartment-pooling could actually exploit) with trivial
    # within-plot-across-years variance (small precisely because it's the same trees). Averaging
    # each plot's own yldc_deviation across its own years first -- same move already used to build
    # mean_cr_residual in Stage 1 -- collapses to one independent unit per plot before the
    # compartment grouping happens.
    return df_with_deviation.groupby(["identification", "cpmt"], as_index=False)["yldc_deviation"].mean()


def compartment_variance_check(df_with_deviation):
    # df_with_deviation must already have a "yldc_deviation" column (built by
    # phase0_checks.yldc_deviation_summary()) and a "cpmt" column. Collapsed to one row per plot
    # first -- see collapse_to_one_row_per_plot()'s own comment for why that matters here.
    one_row_per_plot = collapse_to_one_row_per_plot(df_with_deviation)
    return compute_icc_one_way(one_row_per_plot["yldc_deviation"], one_row_per_plot["cpmt"])


def compartment_variance_check_by_age_band(df_with_deviation):
    # Phase 0 already found the deviation's overall spread GROWS with Age (documentation/
    # experiment_log.md, 2026-08-03 Phase 0 entry) -- checked here too, in case the ICC itself is
    # not constant across age bands (e.g. compartment membership could matter more for young
    # stands than old ones, or vice versa). age_band_5yr is a per-ROW value (Age changes within a
    # plot across its own years), so this check necessarily stays at the row level within each
    # band -- it cannot also collapse to one-row-per-plot the way the overall check above does,
    # since a plot's different years can fall in different age bands. Read this result with that
    # in mind: some of the within-band "within-compartment" spread here is still repeated-measures
    # noise, not purely between-plot variance.
    results = {}
    for age_band, rows in df_with_deviation.groupby("age_band_5yr"):
        if rows["cpmt"].nunique() < 2:
            continue  # ICC is not defined with fewer than 2 groups
        results[age_band] = compute_icc_one_way(rows["yldc_deviation"], rows["cpmt"])
    return results

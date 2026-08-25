# Purpose: problem 2 in documentation/model_instructions/growth_curve_stage2_handover.md --
# the disturbance checks already found a handful of plots with an implausibly large per-plot
# residual swing (max 47.2m for 4survey, 35.3m for 6survey, against a median of only 2-3m), almost
# certainly a data problem (clearfell/replant, boundary mismatch) rather than real growth. But
# WHICH specific plots these are was never actually pulled out and looked at. This module does
# that: identifies the top 1-2% of plots by residual range, and pulls each one's own full
# Age/height/thinning trajectory so a real decision (exclude / flag / investigate case-by-case)
# can be made from actual data, not just the summary statistic.

import numpy as np
import pandas as pd

from models.growth_curve_attribution.disturbance_checks import per_plot_residual_range

TRAJECTORY_COLUMNS = [
    "identification", "cpmt", "LiDAR_year", "Age", "Top_Height95",
    "predicted_height", "residual", "Thin", "last_thinn", "next_thin_", "thinning_status",
]


def identify_long_tail_plots(df_with_residuals, top_fraction=0.02):
    # top_fraction=0.02 means "the top 2% of plots by residual range". Matches the handover
    # doc's own "top 1-2%" framing for problem 2.
    residual_range = per_plot_residual_range(df_with_residuals)
    n_plots = len(residual_range)
    n_to_flag = max(1, int(np.ceil(n_plots * top_fraction)))

    flagged = residual_range.sort_values("residual_range", ascending=False).head(n_to_flag)
    return flagged.reset_index(drop=True)


def get_flagged_plot_trajectories(df_with_residuals, flagged_plot_ids):
    # Returns one row per plot per survey year, for only the flagged plots, sorted so each plot's
    # own years read in chronological order. Meant to be printed/inspected directly, not
    # aggregated further.
    rows = df_with_residuals[df_with_residuals["identification"].isin(flagged_plot_ids)]
    rows = rows[TRAJECTORY_COLUMNS].copy()
    return rows.sort_values(["identification", "LiDAR_year"]).reset_index(drop=True)


def flag_single_year_outlier(trajectories, ratio_threshold=3.0):
    # For each flagged plot: is ONE survey year's residual much bigger than the OTHER years' own
    # typical residual size (median of the others' absolute residuals)? If so, that plot's large
    # residual_range likely comes from a single bad/unusual survey, not a smooth multi-year drift
    #. A real, checkable distinction that changes what "exclude vs flag vs investigate" should
    # mean for that specific plot.
    results = []
    for plot_id, plot_rows in trajectories.groupby("identification"):
        abs_residuals = plot_rows["residual"].abs()
        worst_year_index = abs_residuals.idxmax()
        worst_residual = abs_residuals.loc[worst_year_index]
        worst_year = plot_rows.loc[worst_year_index, "LiDAR_year"]

        other_residuals = abs_residuals.drop(worst_year_index)
        median_of_others = other_residuals.median() if len(other_residuals) > 0 else np.nan

        is_single_year_outlier = (
            not np.isnan(median_of_others)
            and median_of_others > 0
            and worst_residual >= ratio_threshold * median_of_others
        )

        results.append({
            "identification": plot_id,
            "worst_year": worst_year,
            "worst_abs_residual": worst_residual,
            "median_abs_residual_other_years": median_of_others,
            "single_year_outlier": is_single_year_outlier,
        })

    return pd.DataFrame(results)

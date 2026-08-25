# Purpose: checks whether "one Chapman-Richards curve per plot, with a single free y_max" is a
# trustworthy premise at all, BEFORE building any spatial attribution model on top of it (see
# documentation/experiment_log.md's Stage 2 section).
#
# The idea: fit each plot's y_max using ONLY its EARLY survey years, then see how well that
# fitted curve predicts the SAME plot's own LATER, held-out years. If it predicts well, a single
# time-invariant y_max per plot is a sound target to attribute to terrain/wind. If it predicts
# badly. Especially if error grows a lot with how far ahead we're extrapolating. That's
# evidence a real plot's growth isn't well described by one fixed curve across its whole survey
# span (e.g. a disturbance event reset it mid-way), and the per-plot-curve premise itself needs
# revisiting before any spatial attribution work is trustworthy.
#
# This reuses the exact TEMPORAL_YEARS year assignment already used elsewhere in this project
# (models/common/splits.py) for "early years" vs "held-out later years". Not a fresh
# definition, so this check's years line up with Stage 1's own temporal_split convention.

import numpy as np
import pandas as pd

from models.common.metrics import compute_metrics
from models.common.splits import TEMPORAL_YEARS
from models.growth_curve_attribution.data import load_filtered_growth_curve_table

# Matches the yldc curve's own units, per the Stage 2 design decision to compare in Top_Height95
# space (the FC/GYC convention), not the raw elev_percentile_95th Stage 1 trains on.
HEIGHT_COLUMN = "Top_Height95"


def compute_shape_term(df):
    # (1 - exp(-p4 * Age))^p5. The known, Age-dependent part of a plot's own Chapman-Richards
    # curve, given ITS OWN fixed (p4, p5). y_max is the only unknown, and it multiplies this term
    # LINEARLY (height = y_max * shape_term). So fitting y_max is a weighted linear regression
    # through the origin, not an iterative curve_fit.
    return (1 - np.exp(-df["p4"] * df["Age"])) ** df["p5"]


def fit_y_max_per_plot(fitting_rows):
    # fitting_rows: one row per plot per fitting-year, with Age/HEIGHT_COLUMN/p4/p5 already
    # present. Closed-form least-squares solution for y_max in height = y_max * shape_term:
    #   y_max_hat = sum(height * shape_term) / sum(shape_term^2)
    # Works with as few as 1 fitting row per plot (an exact solve), and with more rows becomes a
    # genuine least-squares fit.
    working = fitting_rows.copy()
    working["shape_term"] = compute_shape_term(working)
    working["height_x_shape"] = working[HEIGHT_COLUMN] * working["shape_term"]
    working["shape_squared"] = working["shape_term"] ** 2

    grouped = working.groupby("identification")
    y_max_fit = grouped["height_x_shape"].sum() / grouped["shape_squared"].sum()
    n_fitting_points = grouped.size()

    result = pd.DataFrame({"y_max_fit": y_max_fit, "n_fitting_points": n_fitting_points})
    return result.reset_index()


def evaluate_temporal_stability(cohort):
    df = load_filtered_growth_curve_table(cohort)

    # Only plots with a valid p1-p5/yldc combination can get a yldc-curve comparison point --
    # same NaN population data_processing/export_growth_curve_tables.py already flagged.
    df = df.dropna(subset=["y_max_yldc"]).copy()

    fitting_years = TEMPORAL_YEARS[cohort]["train_years"]
    evaluation_years = TEMPORAL_YEARS[cohort]["val_years"] + TEMPORAL_YEARS[cohort]["test_years"]

    fitting_rows = df[df["LiDAR_year"].isin(fitting_years)]
    y_max_fit_table = fit_y_max_per_plot(fitting_rows)
    print(f"  Fitted y_max for {len(y_max_fit_table):,} plots using years {fitting_years}")

    results = {}
    for evaluation_year in evaluation_years:
        rows_at_this_year = df[df["LiDAR_year"] == evaluation_year]
        evaluation_rows = rows_at_this_year.merge(y_max_fit_table, on="identification", how="inner")

        n_dropped = len(rows_at_this_year) - len(evaluation_rows)
        if n_dropped > 0:
            print(
                f"  WARNING: {n_dropped:,} plots at {evaluation_year} had no fitted y_max "
                "(no fitting-year rows). Excluded from this year's comparison"
            )

        evaluation_rows = evaluation_rows.copy()
        evaluation_rows["early_fit_predicted"] = (
            evaluation_rows["y_max_fit"] * compute_shape_term(evaluation_rows)
        )

        early_fit_metrics = compute_metrics(
            evaluation_rows[HEIGHT_COLUMN], evaluation_rows["early_fit_predicted"], age=evaluation_rows["Age"],
        )
        yldc_curve_metrics = compute_metrics(
            evaluation_rows[HEIGHT_COLUMN], evaluation_rows["top_height95_yldc_predicted"], age=evaluation_rows["Age"],
        )

        results[evaluation_year] = {
            "n_plots": len(evaluation_rows),
            "early_fit": early_fit_metrics,
            "yldc_curve": yldc_curve_metrics,
        }

    return results

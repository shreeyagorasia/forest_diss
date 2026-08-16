# Purpose: classify each compartment into one of five growth-trajectory "archetypes" (close to
# yldc / consistently above / consistently below / trajectory-break outlier / possible reset or
# measurement issue), based on how that compartment's plots deviate from the official yield-class
# benchmark curve and whether any disturbance flags are present. Re-implements the classification
# idea from an earlier, now-stale notebook (notebooks/growth_curve_attribution/
# av2_local_growth_curve_grouped_importance.ipynb, last run 2026-08-08, before the environmental
# Set2-5 restructuring) as a standalone script against the CURRENT data pipeline, so we know
# whether it still finds real examples in every category rather than relying on old cached output.
#
# Same rule shapes as the notebook (percentile-band thresholds recomputed fresh on today's data,
# not copied from the old run), plus two things the notebook didn't have: (1) each representative
# compartment's own mean yield class and Chapman-Richards shape parameters (k, p), and (2) what
# fraction of the curve's eventual asymptote is reached at that compartment's mean survey age --
# i.e. whether the compartment's data sits on the steep early-growth part of the sigmoid or near
# the flat top, since a "consistently above/below yldc" reading means something different at each.
#
# Run as: python -m models.growth_curve_attribution.rq3_compartment_archetype_check --cohort 4survey

import argparse

import numpy as np
import pandas as pd

from models.growth_curve_attribution.data import load_filtered_growth_curve_table
from models.growth_curve_attribution.disturbance_checks import (
    fit_all_years_and_compute_residuals,
    per_plot_residual_range,
    summarize_plot_disturbance_status,
)
from models.growth_curve_attribution.scale_comparison_check import build_plot_level_table, merge_environmental_features
from models.growth_curve_attribution.temporal_stability_check import compute_shape_term

TARGET = "local_y_max_difference"

# A small, targeted set -- not the full environmental candidate pool -- chosen because each one
# is already established elsewhere in this project's RQ3 work as a real signal worth checking
# against this new compartment-level classification: elevation/topex/windward_topex/slope/
# gwa_wind_speed_50m (terrain/wind, already RQ3's own attribution headline variables), CanopyCover
# (the dominant baseline variable everywhere), and the three boundary-distance columns (the
# individual-outlier-plot boundary-proximity finding this check aims to test at compartment scale).
ENVIRONMENT_COLUMNS = [
    "elevation", "topex", "windward_topex", "slope_degrees", "gwa_wind_speed_50m", "CanopyCover",
    "dist_to_cpmt_boundary", "dist_to_block_boundary", "dist_to_forest_perimeter",
]


def build_plot_table(cohort):
    plot_level = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
    # any_ambiguous_disturbance/max_*_drop_frac already exist in plot_level too (same source,
    # summarize_plot_disturbance_status()) -- drop here so the merge below doesn't collide and
    # silently rename both copies with _x/_y suffixes.
    plot_level = plot_level.drop(
        columns=["any_ambiguous_disturbance", "max_height_drop_frac", "max_canopy_drop_frac", "max_volume_drop_frac"],
        errors="ignore",
    )

    # k (p4), p (p5), yldc, blk are not in build_plot_level_table's own output -- pull from the
    # raw growth-curve table instead, one row per plot (these are static per plot, so any survey
    # year's row gives the same p4/p5/yldc/blk value).
    raw = load_filtered_growth_curve_table(cohort)
    per_plot_params = raw[["identification", "blk", "yldc", "p4", "p5", "Age"]].groupby("identification").agg(
        blk=("blk", "first"), yldc=("yldc", "first"), k=("p4", "first"), p=("p5", "first"),
        mean_age=("Age", "mean"),
    ).reset_index()

    disturbance = summarize_plot_disturbance_status(cohort)
    resid_with_years = fit_all_years_and_compute_residuals(cohort)
    resid_range = per_plot_residual_range(resid_with_years)

    table = (
        plot_level
        .merge(per_plot_params, on="identification", how="left")
        .merge(disturbance, on="identification", how="left")
        .merge(resid_range, on="identification", how="left")
    )
    # Same top-1% cutoff convention already established for extreme_trajectory_flag elsewhere in
    # this project's outlier diagnosis work.
    extreme_cutoff = table["residual_range"].quantile(0.99)
    table["extreme_trajectory_flag"] = table["residual_range"] >= extreme_cutoff

    table, available_env_columns = merge_environmental_features(table, ENVIRONMENT_COLUMNS)
    return table, available_env_columns


def classify_compartments(plot_table, env_columns=None):
    plot_table = plot_table.copy()
    plot_table["compartment_key"] = plot_table["cpmt"].astype(str)

    stable_target_band = plot_table[TARGET].abs().quantile(0.35)
    large_target_band = plot_table[TARGET].abs().quantile(0.80)
    large_residual_band = plot_table["residual_range"].quantile(0.90)

    agg_spec = dict(
        cpmt=("cpmt", "first"),
        x=("x", "mean"), y=("y", "mean"),
        n_plots=("identification", "nunique"),
        mean_target=(TARGET, "mean"),
        mean_abs_target=(TARGET, lambda v: float(np.mean(np.abs(v)))),
        mean_residual_range=("residual_range", "mean"),
        flagged_share=("extreme_trajectory_flag", "mean"),
        clearfell_share=("any_clearfell_like", "mean"),
        measurement_issue_share=("any_measurement_inconsistent", "mean"),
        ambiguous_disturbance_share=("any_ambiguous_disturbance", "mean"),
        mean_yldc=("yldc", "mean"),
        mean_k=("k", "mean"),
        mean_p=("p", "mean"),
        mean_age=("mean_age", "mean"),
    )
    for col in (env_columns or []):
        agg_spec[f"mean_{col}"] = (col, "mean")

    compartment_summary = plot_table.groupby("compartment_key").agg(**agg_spec).reset_index()

    # Fraction of the curve's own eventual asymptote reached at this compartment's mean survey
    # age -- (1 - exp(-k*Age))^p evaluated at the compartment's own mean k/p/Age. 1.0 = fully at
    # the flat top of the sigmoid; well under 1.0 = still on the steep early-growth part.
    compartment_summary["frac_of_asymptote_at_mean_age"] = compute_shape_term(
        pd.DataFrame({"p4": compartment_summary["mean_k"], "p5": compartment_summary["mean_p"], "Age": compartment_summary["mean_age"]})
    )

    compartment_summary["pattern"] = "moderate mismatch / mixed"
    compartment_summary.loc[
        (compartment_summary["n_plots"] >= 3)
        & (compartment_summary["mean_abs_target"] <= stable_target_band)
        & (compartment_summary["flagged_share"] <= 0.10)
        & (compartment_summary["ambiguous_disturbance_share"] <= 0.10)
        & (compartment_summary["measurement_issue_share"] == 0),
        "pattern",
    ] = "close to yldc / stable"
    compartment_summary.loc[
        (compartment_summary["n_plots"] >= 3)
        & (compartment_summary["mean_target"] >= large_target_band)
        & (compartment_summary["mean_residual_range"] < large_residual_band)
        & (compartment_summary["flagged_share"] <= 0.15),
        "pattern",
    ] = "consistently above yldc"
    compartment_summary.loc[
        (compartment_summary["n_plots"] >= 3)
        & (compartment_summary["mean_target"] <= -large_target_band)
        & (compartment_summary["mean_residual_range"] < large_residual_band)
        & (compartment_summary["flagged_share"] <= 0.15),
        "pattern",
    ] = "consistently below yldc"
    compartment_summary.loc[
        (compartment_summary["clearfell_share"] >= 0.10) | (compartment_summary["measurement_issue_share"] >= 0.10),
        "pattern",
    ] = "possible reset / measurement issue"
    compartment_summary.loc[
        (compartment_summary["ambiguous_disturbance_share"] >= 0.15)
        | (compartment_summary["flagged_share"] >= 0.20)
        | (compartment_summary["mean_residual_range"] >= large_residual_band),
        "pattern",
    ] = "trajectory-break outlier"

    bands = {
        "stable_target_band": stable_target_band,
        "large_target_band": large_target_band,
        "large_residual_band": large_residual_band,
    }
    return compartment_summary, bands


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="4survey", choices=["4survey", "6survey"])
    args = parser.parse_args()

    plot_table, env_columns = build_plot_table(args.cohort)
    print(f"n plots (after Age/yldc/disturbance filtering): {len(plot_table):,}")
    print(f"environmental columns merged: {env_columns}")

    compartment_summary, bands = classify_compartments(plot_table, env_columns=env_columns)
    print(f"n compartments: {len(compartment_summary):,}")
    print(f"Bands: {bands}")
    print()
    print("Pattern counts:")
    print(compartment_summary["pattern"].value_counts())
    print()

    print("Environmental means by archetype:")
    env_cols_summary = [f"mean_{c}" for c in env_columns]
    print(compartment_summary.groupby("pattern")[env_cols_summary].mean().round(2))
    print()

    pattern_order = [
        "close to yldc / stable", "consistently above yldc", "consistently below yldc",
        "trajectory-break outlier", "possible reset / measurement issue",
    ]
    for pattern in pattern_order:
        subset = compartment_summary[compartment_summary["pattern"] == pattern]
        print(f"--- {pattern}: {len(subset)} compartments ---")
        if len(subset) > 0:
            cols = ["cpmt", "n_plots", "mean_target", "mean_residual_range", "mean_yldc", "mean_k", "mean_p", "frac_of_asymptote_at_mean_age"]
            print(subset[cols].round(3).head(5).to_string(index=False))
        print()


if __name__ == "__main__":
    main()

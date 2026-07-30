# Purpose: Load data for the spatial-attribution methods in this folder -- residuals from ANY
# model (not just Chapman-Richards), the master per-plot-per-year dataset, and a safe way to
# join extra columns (e.g. terrain/wind features, once extracted) onto either one.
# Key logic: every model in this repo writes predictions.csv in the exact same shape
# (identification, LiDAR_year, residual, split, ...), so one loader works for all of them --
# just pass the model's folder name in.

from pathlib import Path

import pandas as pd

from models.common.geo import load_plot_coordinates

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Columns known to leak the height target, or to be an alternate/derived version of it --
# see the Dissertation Plan's "Height-derived predictors" row for the full reasoning. Kept here
# as one shared, named list so every future join can exclude them the same way, rather than
# each analysis re-deciding (and possibly forgetting one) from scratch.
LEAKAGE_RISK_COLUMNS = [
    "Vol95", "Vol99", "Vol_RM95", "GYCspec95", "GYCspec99",
    "elev_percentile_95th", "elev_percentile_99th",  # "raw height percentiles" in the plan
    "Top_Height95",  # the alternate/fallback height target, not a predictor
]


def _load_test_predictions(model_name, cohort, split_type):
    # Shared first step for both residual-loading functions below: read predictions.csv, keep
    # only the test split. Real held-out residual, not a training-fit residual -- if we used
    # training rows, the model would look artificially good there (it was literally fitted to
    # match them), so any spatial pattern we found could just be an artifact of the fitting
    # process, not something real about the forest.
    predictions_path = PROJECT_ROOT / "outputs" / split_type / model_name / cohort / "predictions.csv"
    predictions_df = pd.read_csv(predictions_path)
    return predictions_df[predictions_df["split"] == "test"]


def load_residuals(model_name, cohort="4survey", split_type="spatial_block"):
    # One row per plot: mean residual across its survey years, joined to x/y and compartment.
    # This is the shape Moran's I / LISA / SHAP / NLME / GWR all need -- one value per map
    # location, since a spatial clustering test can't work if several different residuals sit
    # at the same x/y point. `model_name` is the same folder name used everywhere else in this
    # repo -- "chapman_richards", "dnn_noenv", "pinn_noenv_pw0.05_tw0.05", "rf_baseline", etc.
    test_only = _load_test_predictions(model_name, cohort, split_type)
    mean_residual = test_only.groupby("identification")["residual"].mean().reset_index()

    coordinates_df = load_plot_coordinates()
    plots = join_by_plot(coordinates_df, mean_residual)

    print(f"Loaded {len(plots):,} plots with a real {model_name} residual ({cohort}, {split_type}, test split only)")
    return plots


def load_residuals_by_year(model_name, cohort="4survey", split_type="spatial_block"):
    # Same idea as load_residuals(), but keeps every survey year as its own row instead of
    # averaging them away. Needed for the storm/windthrow diagnostic idea (see progress_notes.md):
    # telling apart a plot that's consistently bad every year (a chronic wind-exposure signature)
    # from one that was fine until a specific year and then dropped (more likely a storm event) --
    # that distinction is impossible to see once the years have been averaged into one number.
    test_only = _load_test_predictions(model_name, cohort, split_type)

    coordinates_df = load_plot_coordinates()
    plots_by_year = join_by_plot(coordinates_df, test_only[["identification", "LiDAR_year", "residual"]])

    n_plots = plots_by_year["identification"].nunique()
    print(f"Loaded {len(plots_by_year):,} plot-year rows across {n_plots:,} plots "
          f"({model_name}, {cohort}, {split_type}, test split only, years kept separate)")
    return plots_by_year


def load_master(cohort="4survey"):
    # The full cleaned per-plot-per-year dataset every model's predictor table is built FROM --
    # one row per plot per survey year, every column the cleaning notebook kept. No coordinates
    # in this file (join load_plot_coordinates() separately, same as the functions above), and
    # no filtering to any particular model's predictor list -- this is the raw material to build
    # a new feature table from, not a ready-to-use one.
    master_path = PROJECT_ROOT / "data" / "processed" / "master" / f"clean_master_{cohort}.parquet"
    master_df = pd.read_parquet(master_path)
    print(f"Loaded master dataset: {len(master_df):,} rows, {master_df['identification'].nunique():,} plots ({cohort})")
    return master_df


def join_by_plot(base_df, other_df, on="identification"):
    # A safety-checked version of pd.merge() -- reports how many rows from EACH side failed to
    # find a match, so a bad join (wrong ID type, missing plots, a typo in a column name) fails
    # loudly and gets noticed, instead of silently dropping rows and only showing up much later
    # as "why does my analysis have fewer plots than I expected".
    merged = base_df.merge(other_df, on=on, how="inner")

    unmatched_in_base = len(base_df) - len(base_df.merge(other_df, on=on, how="inner"))
    unmatched_in_other = len(other_df) - len(other_df.merge(base_df, on=on, how="inner"))

    print(f"Joined on '{on}': {len(merged):,} matched rows")
    if unmatched_in_base > 0:
        print(f"  WARNING: {unmatched_in_base:,} rows from the first table had no match and were dropped")
    if unmatched_in_other > 0:
        print(f"  WARNING: {unmatched_in_other:,} rows from the second table had no match and were dropped")

    return merged

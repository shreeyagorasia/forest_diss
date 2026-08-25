# Purpose: answer the user's actual question. For the Stage 2 growth-curve attribution work,
# is PER-PLOT the right unit of analysis, or should plots be aggregated up to the subcompartment
# (scpt) level first? This mirrors the same comparison already sketched in
# notebooks/growth_curve_attribution/local_growth_curve_grouped_importance.ipynb (sections 8-10),
# but built as its own standalone script here. Not editing that notebook, since it's actively
# being worked on separately.
#
# Uses the SAME target construction as that notebook (local_y_max_difference = fitted per-plot
# y_max minus the yldc-implied y_max) and the established TERRAIN_AND_WIND_COLUMNS feature set
# from xgb_environmental.py (the narrow, already-vetted 16-column terrain+wind list. Not the
# wider experimental candidate columns the notebook is separately trying out), so the resulting
# R2 numbers are directly comparable to Stage 1's own established terrain/wind ceiling
# (R2~0.16-0.19).

import numpy as np
import pandas as pd

from models.common.metrics import compute_metrics
from models.common.splits import SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, SPLIT_SEED, spatial_block_split
from models.elasticnet_environmental.elasticnet_environmental import drop_rows_with_missing_features
from models.elasticnet_environmental.elasticnet_environmental import fit_with_columns as elasticnet_fit
from models.elasticnet_environmental.elasticnet_environmental import predict_with_columns as elasticnet_predict
from models.growth_curve_attribution.data import load_filtered_growth_curve_table
from models.growth_curve_attribution.disturbance_checks import summarize_plot_disturbance_status
from models.growth_curve_attribution.temporal_stability_check import fit_y_max_per_plot
from models.xgb_environmental.data import load_environmental_features
from models.xgb_environmental.xgb_environmental import TERRAIN_AND_WIND_COLUMNS
from models.xgb_environmental.xgb_environmental import fit_with_columns as xgb_fit
from models.xgb_environmental.xgb_environmental import predict_with_columns as xgb_predict

TARGET = "local_y_max_difference"


DISTURBANCE_METADATA_COLUMNS = [
    "any_ambiguous_disturbance", "max_height_drop_frac", "max_canopy_drop_frac", "max_volume_drop_frac",
]


def build_plot_level_table(cohort, apply_disturbance_cleaning=True):
    # Same construction as the notebook's section 1: fixed-shape per-plot y_max (from that
    # plot's own repeated Top_Height95 observations) minus the FC's own static yldc-implied
    # y_max. Only plots with a valid p1-p5/yldc combination are kept (dropna on y_max_yldc).
    growth_rows = load_filtered_growth_curve_table(cohort).dropna(subset=["y_max_yldc"]).copy()

    # disturbance_status is computed from the FULL (unfiltered-by-maturity) growth-curve table
    # inside summarize_plot_disturbance_status(). Consecutive-survey intervals need every
    # survey year a plot has, not just the ones that survive the Age/yldc gate here. Merging back
    # onto growth_rows (already maturity-filtered) keeps only the plots this check actually uses.
    disturbance_status = summarize_plot_disturbance_status(cohort)

    if apply_disturbance_cleaning:
        excluded_ids = set(disturbance_status.loc[disturbance_status["exclude_from_curve_fit"], "identification"])
        n_before = growth_rows["identification"].nunique()
        growth_rows = growth_rows[~growth_rows["identification"].isin(excluded_ids)]
        n_after = growth_rows["identification"].nunique()
        print(
            f"  Disturbance cleaning: excluded {n_before - n_after:,} of {n_before:,} plots "
            "(clearfell-like or measurement-inconsistent) before fitting y_max"
        )

    fitted_y_max = fit_y_max_per_plot(growth_rows)

    plot_static = (
        growth_rows.sort_values("LiDAR_year")
        .groupby("identification", as_index=False)
        .first()[["identification", "cpmt", "scpt", "x", "y", "y_max_yldc"]]
    )
    plot_table = plot_static.merge(fitted_y_max, on="identification", how="inner")
    plot_table[TARGET] = plot_table["y_max_fit"] - plot_table["y_max_yldc"]

    # ambiguous_disturbance plots are NOT excluded. Kept in the modelling population, with the
    # flag/drop-fractions attached as metadata a downstream attribution model can use as a
    # feature, rather than silently ignoring plots with a possible genuine disturbance event.
    plot_table = plot_table.merge(
        disturbance_status[["identification"] + DISTURBANCE_METADATA_COLUMNS], on="identification", how="left"
    )
    return plot_table


def merge_environmental_features(plot_table, feature_columns=TERRAIN_AND_WIND_COLUMNS):
    # feature_columns defaults to the established 16 (unchanged behaviour for every existing
    # caller). Exposed as a parameter so representation checks (e.g. swapping
    # gwa_wind_speed_10m for the 50m version) can reuse this same merge step without duplicating
    # it. None of TERRAIN_AND_WIND_COLUMNS are cohort-specific (that only applies to
    # mean_cr_residual/tas_mean/groundfrost_mean. See xgb_environmental.data.py), so this is a
    # plain merge on identification, no per-cohort renaming needed.
    environment = load_environmental_features()
    available_columns = [column for column in feature_columns if column in environment.columns]
    merged = plot_table.merge(
        environment[["identification"] + available_columns], on="identification", how="left", validate="one_to_one"
    )
    return merged, available_columns


def build_subcompartment_table(plot_table_with_features, feature_columns):
    # Aggregates plot-level rows up to one row per (cpmt, scpt) subcompartment: the target and
    # every continuous feature become their median (matching the notebook's own choice, robust to
    # the same long-tail plots discussed elsewhere in this project), coordinates become the mean
    # so a subcompartment still has a real (x, y) position for the spatial buffer step.
    group_keys = ["cpmt", "scpt"]
    aggregation = {column: "median" for column in feature_columns}
    area_table = plot_table_with_features.groupby(group_keys, as_index=False).agg(
        n_plots=("identification", "nunique"),
        x=("x", "mean"),
        y=("y", "mean"),
        **{TARGET: (TARGET, "median")},
        **{column: (column, method) for column, method in aggregation.items()},
    )
    # spatial_block_split()/apply_spatial_buffer() both key off an "identification" column --
    # a synthetic one-per-subcompartment id lets the exact same split function run unchanged at
    # this coarser scale.
    area_table["identification"] = area_table["cpmt"].astype(str) + "::" + area_table["scpt"].astype(str)
    return area_table


def run_validation_screen(table, feature_columns, scale_name, cohort, split_seed=SPLIT_SEED):
    # BUG FIX (2026-08-04): this used to evaluate BOTH models on "val". The exact same rows
    # XGBoost's val_df=val early-stopping already used to decide how many boosting rounds to run.
    # That's a real leak: the model's own fit was tuned to minimise loss on the rows its
    # performance was then measured on, optimistically biasing every XGBoost R2 this function
    # produced. spatial_block_split() already builds a genuine third partition ("test") that was
    # sitting unused. "val" now ONLY drives early stopping; both methods are scored on "test".
    table = table.copy()
    coordinates = table[["identification", "x", "y"]]
    table["split"] = spatial_block_split(
        table, block_col=SPATIAL_BLOCK_COL, buffer_distance=SPATIAL_BUFFER_METRES,
        coordinates_df=coordinates, seed=split_seed,
    )

    train = drop_rows_with_missing_features(table[table["split"] == "train"], feature_columns)
    val = drop_rows_with_missing_features(table[table["split"] == "val"], feature_columns)
    test = drop_rows_with_missing_features(table[table["split"] == "test"], feature_columns)

    elastic = elasticnet_fit(train, feature_columns, target_col=TARGET)
    elastic_predictions = elasticnet_predict(test, elastic, feature_columns)
    elastic_metrics = compute_metrics(test[TARGET], elastic_predictions)

    tree = xgb_fit(train, feature_columns, val_df=val, target_col=TARGET, n_estimators=500, max_depth=4, learning_rate=0.04)
    tree_predictions = xgb_predict(test, tree, feature_columns)
    tree_metrics = compute_metrics(test[TARGET], tree_predictions)

    return pd.DataFrame([
        {"cohort": cohort, "scale": scale_name, "method": "Elastic Net", "split_seed": split_seed, "n_train": len(train), "n_val": len(val), "n_test": len(test), **elastic_metrics},
        {"cohort": cohort, "scale": scale_name, "method": "XGBoost", "split_seed": split_seed, "n_train": len(train), "n_val": len(val), "n_test": len(test), **tree_metrics},
    ])


def run_for_cohort(cohort, split_seed=SPLIT_SEED, scales=("plot", "subcompartment"), apply_disturbance_cleaning=True):
    plot_table = build_plot_level_table(cohort, apply_disturbance_cleaning=apply_disturbance_cleaning)
    plot_table_with_features, feature_columns = merge_environmental_features(plot_table)

    results = []
    if "plot" in scales:
        results.append(run_validation_screen(plot_table_with_features, feature_columns, "plot", cohort, split_seed=split_seed))
    if "subcompartment" in scales:
        area_table = build_subcompartment_table(plot_table_with_features, feature_columns)
        results.append(run_validation_screen(area_table, feature_columns, "subcompartment", cohort, split_seed=split_seed))
    return pd.concat(results, ignore_index=True)

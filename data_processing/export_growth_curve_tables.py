# Run as: python -m data_processing.export_growth_curve_tables
#
# Builds the data foundation for "Stage 2" -- the new per-plot-growth-curve-vs-yldc model family
# (see documentation/experiment_log.md's Stage 2 section). This is a SEPARATE table from Stage
# 1's data_processing/export_model_tables.py / model_table.parquet -- it does not replace that
# pipeline, which stays exactly as-is for the already-reported baseline/DNN/PINN results.
#
# WHY THIS NEEDS ITS OWN SCRIPT, NOT A WIDER model_table.parquet
# ------------------------------------------------------------------
# export_model_tables.py's AUDIT_COLUMNS/ALL_FEATURE_COLUMNS lists deliberately only keep the
# columns Stage 1's models actually use -- p1-p5, g1, g3, area, scpt, spis, plyr are all dropped
# there, not because they're bad data, just because no Stage 1 model needed them. Stage 2 needs
# p1-p5 (to compute each plot's own yldc-implied growth curve) and plot coordinates (for any
# distance-based Stage 2 model), so this script reads directly from the already-cleaned master
# export (clean_master_<cohort>.parquet, built by clean_master_data.py -- no raw GeoPackage
# re-read needed) and keeps every column that table already has.
#
# WHAT THIS SCRIPT ADDS ON TOP OF THE MASTER TABLE
# ------------------------------------------------------
#   1. Plot coordinates (x, y, EPSG:27700) -- every Stage 2 model needs location, unlike Stage 1
#      where only spatial_block_split's own buffering needed it.
#   2. y_max_yldc -- each plot's OWN yldc-implied growth-curve asymptote, in Top_Height95 units
#      (matching the FC's own GYC convention -- see the Stage 2 planning discussion for why
#      comparisons happen in this space, not the raw elev_percentile_95th Stage 1 trains on).
#   3. top_height95_yldc_predicted -- the yldc curve's predicted height at THAT row's own Age, so
#      every Stage 2 model can see "how far is this row from its own yldc curve" without
#      recomputing the formula itself.
#
# TERRAIN/WIND/CLIMATE FEATURES ARE DELIBERATELY NOT MERGED IN HERE
# ------------------------------------------------------------------------
# Stage 1's own ENV_TERRAIN_FEATURE_SETS (models/common/torch_data.py) already gives a
# well-vetted, swappable-by-name choice of which environmental columns to use, decided
# separately from this export. Baking one hard-coded terrain merge into this table would freeze
# that choice prematurely -- a Stage 2 model that needs terrain merges
# plot_environmental_features.parquet in at load time instead, the same way Stage 1's own
# load_split_table_with_terrain() already does.

from pathlib import Path

import numpy as np
import pandas as pd

from models.common.geo import load_plot_coordinates

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_DIR = PROJECT_ROOT / "data" / "processed" / "master"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "growth_curve"
COHORTS = ["4survey", "6survey"]


def load_master(cohort):
    path = MASTER_DIR / f"clean_master_{cohort}.parquet"
    return pd.read_parquet(path)


def add_coordinates(master_df, coordinates_df):
    # plot_coordinates.csv.gz also carries its own "cpmt" column (checked directly -- it has
    # identification/cpmt/x/y), which collides with master_df's own "cpmt" -- merging without
    # handling this doesn't error at the merge itself, it silently renames both to cpmt_x/cpmt_y
    # (caught by actually inspecting this export's output columns, not assumed). Same bug shape
    # already found and fixed once for "whcl" in models/common/torch_data.py -- same general fix
    # here: drop any coordinates column that already exists in master_df (other than the
    # "identification" join key) before merging, so master_df's own version always wins and
    # nothing ends up suffixed.
    columns_already_present = [
        c for c in coordinates_df.columns if c != "identification" and c in master_df.columns
    ]
    if columns_already_present:
        coordinates_df = coordinates_df.drop(columns=columns_already_present)

    # Left join, master_df's own rows are never dropped or duplicated -- coordinates_df is one
    # row per plot (checked below), so this merge should add exactly two new columns (x, y) and
    # change nothing else about the row count.
    rows_before = len(master_df)
    merged = master_df.merge(coordinates_df, on="identification", how="left")

    if len(merged) != rows_before:
        raise ValueError(
            f"Merging in plot coordinates changed the row count ({rows_before:,} -> "
            f"{len(merged):,}) -- plot_coordinates.csv.gz should have exactly one row per plot; "
            "check it for duplicate 'identification' values before trusting this export."
        )

    missing_coordinates = merged["x"].isna().sum()
    if missing_coordinates > 0:
        print(
            f"  WARNING: {missing_coordinates:,} of {rows_before:,} rows have no matching plot "
            "coordinate -- check plot_coordinates.csv.gz covers every plot in the master table."
        )

    return merged


def add_yldc_curve_columns(df):
    # The yldc-implied growth curve, rearranged from the same formula the FC's own GYCspec95
    # column uses (documentation/data_exploration_gpkg's cheat sheet, section 8.4):
    #   GYCspec95 = (Top_Height95 / (1 - exp(-p4 * Age))^p5 - p1 - p3*2) / p2
    # Solved the other way round -- given a plot's own yldc (not solving FOR yldc), what height
    # does the curve predict:
    #   Top_Height95_yldc(Age) = (yldc * p2 + p1 + p3*2) * (1 - exp(-p4 * Age))^p5
    # y_max_yldc is the Age-independent part of that -- the plot's own yldc-implied asymptote,
    # using THAT SAME plot's own recorded (p1..p5) tuple, not one universal species constant.
    # (p1-p5 takes 7 distinct value-sets in this data, constant per plot but not tied to that
    # plot's own yldc -- using each plot's own tuple is the only way to stay faithful to how the
    # FC itself computed GYCspec95/Vol95 for that specific plot; see experiment_log.md's Stage 2
    # planning notes.)
    df = df.copy()
    df["y_max_yldc"] = df["yldc"] * df["p2"] + df["p1"] + df["p3"] * 2
    df["top_height95_yldc_predicted"] = df["y_max_yldc"] * (1 - np.exp(-df["p4"] * df["Age"])) ** df["p5"]

    # p1-p5 is missing for a small number of plots (not every row got a valid FR species/yield
    # lookup) -- these rows will have y_max_yldc/top_height95_yldc_predicted as NaN, same
    # silent-NaN-propagation convention as the rest of this pipeline, but worth printing so this
    # isn't discovered by surprise later.
    n_missing = int(df["y_max_yldc"].isna().sum())
    if n_missing > 0:
        n_plots_missing = df.loc[df["y_max_yldc"].isna(), "identification"].nunique()
        print(
            f"  NOTE: {n_missing:,} rows ({n_plots_missing:,} plots) have no valid p1-p5/yldc "
            "combination, so y_max_yldc/top_height95_yldc_predicted are NaN for them -- these "
            "plots can't be compared against a yldc curve, but are still kept in this export "
            "(every other column is still valid for them)."
        )

    return df


def build_growth_curve_table(cohort, coordinates_df):
    master_df = load_master(cohort)
    with_coordinates = add_coordinates(master_df, coordinates_df)
    with_curve_columns = add_yldc_curve_columns(with_coordinates)
    return with_curve_columns


def main():
    coordinates_df = load_plot_coordinates()

    for cohort in COHORTS:
        print(f"Building growth_curve_table for {cohort} ...")
        table = build_growth_curve_table(cohort, coordinates_df)

        output_path = OUTPUT_DIR / cohort / "growth_curve_table.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(output_path, index=False)
        print(f"  Saved {len(table):,} rows, {table.shape[1]} cols -> {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

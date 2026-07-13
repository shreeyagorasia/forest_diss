# Run as: python -m data_processing.export_model_tables
#
# WHAT THIS SCRIPT DOES
# ----------------------
# Reads the already-cleaned master exports (clean_master_4survey.parquet,
# clean_master_6survey.parquet) and builds every table the models/ package
# reads: one "current-state" table per baseline model, plus two "transition"
# tables for later growth-rate modelling.
#
# WHY THIS IS A SEPARATE SCRIPT, NOT PART OF THE NOTEBOOK
# ----------------------------------------------------------
# The master exports are the actual output of the cleaning funnel: species
# filtering to Sitka spruce, age/height validity checks, duplicate removal,
# and balancing each cohort to exactly 4 or 6 surveys per plot. That cleaning
# work only happens in data_exploration_gpkg/notebooks/lidar_years_all_data_cleaning.ipynb,
# and it's a large, actively-edited notebook -- not somewhere you want the
# one thing every model depends on (which columns go into which table) to
# also live, if there's a simpler and more robust place for it.
#
# Once the master exports exist on disk, this script needs nothing else from
# that notebook: choosing predictor columns, deriving target/fallback
# columns, and building the transition tables is pure pandas column
# selection, not anything that depends on the raw GeoPackage or the cleaning
# funnel. So if the notebook is ever broken or mid-edit, every table below
# can still be regenerated from the stable master files alone.
#
# WHICH COLUMNS ARE FEATURES VS EVALUATION-ONLY
# ------------------------------------------------
# The master exports deliberately carry more columns than any model should
# ever see as a predictor. The ones NOT listed as predictors below are kept
# in the master file for a reason, but must never be fed into a model:
#   - Vol95, Vol99, Vol_RM95, GYCspec95, GYCspec99: derived FROM height
#     and/or age, so using them as predictors would leak the answer.
#   - area, p1-p5, g1, g3: formula ingredients needed to recompute those
#     derived quantities after a model makes a height prediction (to check
#     whether the prediction implies a sensible volume/yield class), not
#     independent plot measurements.
#   - Top_Height95: the fallback target. Never a predictor for Top_Height99.
#   - whcl: a management-linked windthrow hazard class from the forest
#     inventory, not a raw measurement of wind exposure -- kept for
#     audit/stratification only.
#   - LAI: kept for canopy data-quality checks; not used alongside
#     CanopyCover in baseline models, to avoid two redundant canopy inputs.
#   - identification, LiDAR_year, blk: row identity and the spatial
#     grouping column used for splitting -- metadata, not predictors.
#     blk in particular must NEVER be passed to a model as a feature.

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
COHORTS = ["4survey", "6survey"]

# Row identity and the spatial split column. Kept in every table for
# splitting/evaluation, but never passed to a model as a feature.
METADATA_COLUMNS = ["identification", "LiDAR_year", "blk"]

# Top_Height99 is the primary modelling target (see the cleaning notebook's
# section 11.2 and documentation/plans_md's Accuracy Review for the reasoning
# and the known Top_Height99 < Top_Height95 caveat). Top_Height95 rides along
# as a fallback/audit target in every table, but is never itself a feature.
TARGET_COLUMN = "Top_Height99"
FALLBACK_TARGET_COLUMNS = ["Top_Height95"]

# One dictionary per model, mapping model name -> the predictor columns that
# model actually uses. This is the single place that decides what each
# model is allowed to see.
MODEL_FEATURE_SETS = {
    # Age-only Chapman-Richards growth curve -- no stand-management inputs.
    "cr_age": [
        "Age",
    ],
    # Interpretable linear baseline. thinning_status is a categorical
    # summary of last_thinn (never_thinned / recent_0_5yr / mid_6_10yr /
    # old_10plus_yr / not_yet_thinned_at_survey / unknown_timing) -- one
    # categorical term is easier to interpret than several overlapping
    # numeric thinning columns. Must be one-hot encoded before fitting.
    "linear_baseline": [
        "Age",
        "CanopyCover",
        "thinning_status",
        "yldc",
    ],
    # Non-linear baseline. Numeric time_since_thinning is useful for a tree
    # model, but time_since_thinning_missing is required alongside it,
    # because never-thinned plots have no valid elapsed time to report.
    "rf_baseline": [
        "Age",
        "CanopyCover",
        "Thin",
        "time_since_thinning",
        "time_since_thinning_missing",
        "recent_thinning_5yr",
        "yldc",
    ],
}

# DNN and PINN (no-environment) tables share one feature set, so there is
# only one predictor list to keep in sync between them. Not consumed by any
# model script yet -- exported now so the tables are ready when that work
# starts, matching what the cleaning notebook already produced.
FULL_NOENV_FEATURES = [
    "Age",
    "CanopyCover",
    "Thin",
    "time_since_thinning",
    "time_since_thinning_missing",
    "recent_thinning_5yr",
    "thinning_status",
    "yldc",
]
MODEL_FEATURE_SETS["dnn_noenv"] = FULL_NOENV_FEATURES
MODEL_FEATURE_SETS["pinn_noenv"] = FULL_NOENV_FEATURES

# Columns kept in the transition (growth-between-surveys) tables. This is a
# different question to the current-state tables above: the target here is
# annual_height99_increment, not Top_Height99 itself.
TRANSITION_COLUMNS = [
    "identification",
    "LiDAR_year",
    "blk",
    "previous_lidar_year",
    "survey_interval_years",
    "previous_age",
    "Age",
    "previous_top_height99",
    "Top_Height99",
    "height99_increment",
    "annual_height99_increment",
    "previous_top_height95",
    "Top_Height95",
    "previous_canopy_cover",
    "CanopyCover",
    "canopy_change",
    "Thin",
    "time_since_thinning",
    "time_since_thinning_missing",
    "recent_thinning_5yr",
    "thinning_status",
    "yldc",
]


def load_master(cohort):
    path = PROCESSED_DIR / "master" / f"clean_master_{cohort}.parquet"
    return pd.read_parquet(path)


def select_model_columns(master_df, feature_columns):
    # Metadata + this model's predictors + the primary target + the
    # fallback target, with duplicates removed while keeping the first
    # occurrence (a feature list could in principle repeat a metadata
    # column name, though none currently do).
    selected_columns = METADATA_COLUMNS + feature_columns + [TARGET_COLUMN] + FALLBACK_TARGET_COLUMNS
    selected_columns = list(dict.fromkeys(selected_columns))
    return master_df[selected_columns].copy()


def build_current_state_tables(master_df):
    # One table per entry in MODEL_FEATURE_SETS, e.g. {"cr_age": <table>, ...}
    tables = {}
    for model_name, feature_columns in MODEL_FEATURE_SETS.items():
        tables[model_name] = select_model_columns(master_df, feature_columns)
    return tables


def build_transition_table(master_df):
    # Growth between consecutive surveys for the same plot, not a
    # current-height table. previous_* columns come from shifting each
    # plot's own history forward by one survey (groupby + shift), so the
    # first survey of every plot has no "previous" and is dropped at the end.
    data = master_df.sort_values(["identification", "LiDAR_year"]).copy()
    grouped = data.groupby("identification")

    data["previous_age"] = grouped["Age"].shift()
    data["previous_top_height99"] = grouped["Top_Height99"].shift()
    data["previous_top_height95"] = grouped["Top_Height95"].shift()
    data["previous_canopy_cover"] = grouped["CanopyCover"].shift()

    data["height99_increment"] = data["Top_Height99"] - data["previous_top_height99"]
    data["annual_height99_increment"] = data["height99_increment"] / data["survey_interval_years"]
    data["canopy_change"] = data["CanopyCover"] - data["previous_canopy_cover"]

    # A row with no previous survey has nothing to compute growth from.
    data = data.dropna(subset=["previous_top_height99"])

    return data[TRANSITION_COLUMNS].copy()


def build_export_plan():
    # Maps every output Path to the table that should be written there.
    # current_state/ keeps the cohort in the DIRECTORY (so the same model
    # code reads either cohort by just changing a path); master/ and
    # transitions/ keep the cohort in the FILENAME, matching how the
    # cleaning notebook already names its own master exports.
    export_plan = {}

    for cohort in COHORTS:
        master_df = load_master(cohort)

        current_state_tables = build_current_state_tables(master_df)
        for model_name, table in current_state_tables.items():
            output_path = PROCESSED_DIR / "current_state" / cohort / f"{model_name}.parquet"
            export_plan[output_path] = table

        transition_table = build_transition_table(master_df)
        output_path = PROCESSED_DIR / "transitions" / f"transition_growth_{cohort}.parquet"
        export_plan[output_path] = transition_table

    return export_plan


def main():
    export_plan = build_export_plan()

    for output_path, table in export_plan.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(output_path, index=False)
        rel_path = output_path.relative_to(PROJECT_ROOT)
        print(f"Saved {len(table):,} rows, {table.shape[1]} cols -> {rel_path}")


if __name__ == "__main__":
    main()

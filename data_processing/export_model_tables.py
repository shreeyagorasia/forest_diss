# Run as: python -m data_processing.export_model_tables
#
# Reads the already-cleaned master exports (clean_master_4survey.parquet,
# clean_master_6survey.parquet, built by data_processing/clean_master_data.py) and builds the
# tables the models/ package reads: ONE consolidated "current-state" table per cohort, plus one
# "transition" table per cohort for growth-rate modelling.
#
# CONSOLIDATED, NOT ONE FILE PER MODEL (changed 2026-07-28)
# ------------------------------------------------------------
# Previously this script wrote a separate parquet per model (cr_age.parquet, linear_baseline.
# parquet, rf_baseline.parquet, dnn_noenv.parquet, pinn_noenv.parquet). Checked directly:
# dnn_noenv.parquet and pinn_noenv.parquet were byte-for-byte IDENTICAL, and the other three were
# each a strict subset of the same core columns -- five files carrying the same handful of
# columns in slightly different combinations. Now there is ONE table per cohort
# (current_state/<cohort>/model_table.parquet) containing every column any model needs; each
# model's own module (e.g. models/rf_baseline/rf_baseline.py's FEATURE_COLUMNS) selects its own
# subset at load time, the same way it already selects columns out of any dataframe.
#
# TARGET CHANGED FROM Top_Height99 TO elev_percentile_95th (2026-07-28)
# ------------------------------------------------------------------------
# See data_processing/clean_master_data.py's own header comment for the full reasoning.
# Top_Height99 is retired everywhere; there is no fallback target column anymore (Top_Height95
# is kept in the master table, but only as an audit ingredient for Vol95/GYCspec95, never a
# target or feature -- see clean_master_data.py).
#
# yldc REMOVED AS A FEATURE (2026-07-28)
# ------------------------------------------
# Confirmed via real held-out ablation (not just theory) that yldc hurts generalisation in every
# model checked (see documentation/progress_notes.md's 2026-07-28 entry for the numbers). Still
# present in the master table and in this consolidated export (useful for audit/stratification,
# same treatment as whcl), just no longer in any model's own FEATURE_COLUMNS list.
#
# WHY THIS IS A SEPARATE SCRIPT, NOT PART OF THE CLEANING SCRIPT
# -------------------------------------------------------------------
# clean_master_data.py does the expensive, one-way work (reading the raw GeoPackage, applying
# the cleaning funnel). Everything here is pure pandas column selection on the already-cleaned
# master files, so it can be re-run instantly whenever a model's column needs change, without
# ever touching the raw GeoPackage again.

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
COHORTS = ["4survey", "6survey"]

# Row identity and the spatial split columns. Kept in every table for splitting/evaluation, but
# never passed to a model as a feature. blk and cpmt in particular must NEVER be passed to a
# model as a feature -- cpmt (forestry compartment) is the block_col spatial_block_split()
# groups by, so a model that saw cpmt as a feature could effectively memorise which compartment
# a row came from.
METADATA_COLUMNS = ["identification", "LiDAR_year", "blk", "cpmt"]

TARGET_COLUMN = "elev_percentile_95th"

# Audit-only columns carried into the consolidated table but never selected as a feature by any
# model -- same treatment as whcl already had. yldc moved here 2026-07-28 (see header comment).
AUDIT_COLUMNS = ["yldc", "whcl", "Top_Height95", "Vol95", "GYCspec95"]

# The union of every feature column any current model uses (cr_age uses only Age; rf_baseline/
# dnn_noenv/pinn_noenv use the full set; linear_baseline uses a subset). One shared list here,
# not a separate per-model dict -- each model's own FEATURE_COLUMNS (rf_baseline.py,
# linear_baseline.py, models/common/torch_data.py's NOENV_FEATURE_COLUMNS) is the actual source
# of truth for which of these columns that specific model reads.
ALL_FEATURE_COLUMNS = [
    "Age",
    "CanopyCover",
    "Thin",
    "time_since_thinning",
    "time_since_thinning_missing",
    "recent_thinning_5yr",
    "thinning_status",
]

# Columns kept in the transition (growth-between-surveys) table. Different question to the
# current-state table above: the target here is annual_height_increment, not the target column
# itself. previous_* columns give the EARLIER survey's values, alongside the plain (unprefixed)
# LATER survey's values -- both endpoints of a transition pair carry the full feature set, which
# is what the PINN's trajectory loss needs (a forward pass on each endpoint using that
# endpoint's own real feature values, not just its Age and height).
TRANSITION_COLUMNS = [
    "identification", "LiDAR_year", "blk", "cpmt",
    "previous_lidar_year", "survey_interval_years",
    "previous_age", "Age",
    "previous_height", TARGET_COLUMN, "height_increment", "annual_height_increment",
    "previous_canopy_cover", "CanopyCover", "canopy_change",
    "previous_Thin", "Thin",
    "previous_time_since_thinning", "time_since_thinning",
    "previous_time_since_thinning_missing", "time_since_thinning_missing",
    "previous_recent_thinning_5yr", "recent_thinning_5yr",
    "previous_thinning_status", "thinning_status",
    "previous_yldc", "yldc",
]


def load_master(cohort):
    path = PROCESSED_DIR / "master" / f"clean_master_{cohort}.parquet"
    return pd.read_parquet(path)


def build_current_state_table(master_df):
    # Metadata + every feature any model needs + the target + audit-only columns, with
    # duplicates removed while keeping the first occurrence.
    selected_columns = METADATA_COLUMNS + ALL_FEATURE_COLUMNS + [TARGET_COLUMN] + AUDIT_COLUMNS
    selected_columns = list(dict.fromkeys(selected_columns))
    return master_df[selected_columns].copy()


def build_transition_table(master_df):
    # Growth between consecutive surveys for the same plot, not a current-height table.
    # previous_* columns come from shifting each plot's own history forward by one survey
    # (groupby + shift), so the first survey of every plot has no "previous" and is dropped.
    data = master_df.sort_values(["identification", "LiDAR_year"]).copy()
    grouped = data.groupby("identification")

    data["previous_age"] = grouped["Age"].shift()
    data["previous_height"] = grouped[TARGET_COLUMN].shift()
    data["previous_canopy_cover"] = grouped["CanopyCover"].shift()
    data["previous_Thin"] = grouped["Thin"].shift()
    data["previous_time_since_thinning"] = grouped["time_since_thinning"].shift()
    data["previous_time_since_thinning_missing"] = grouped["time_since_thinning_missing"].shift()
    data["previous_recent_thinning_5yr"] = grouped["recent_thinning_5yr"].shift()
    data["previous_thinning_status"] = grouped["thinning_status"].shift()
    data["previous_yldc"] = grouped["yldc"].shift()

    data["height_increment"] = data[TARGET_COLUMN] - data["previous_height"]
    data["annual_height_increment"] = data["height_increment"] / data["survey_interval_years"]
    data["canopy_change"] = data["CanopyCover"] - data["previous_canopy_cover"]

    # A row with no previous survey has nothing to compute growth from.
    data = data.dropna(subset=["previous_height"])

    return data[TRANSITION_COLUMNS].copy()


def build_export_plan():
    export_plan = {}

    for cohort in COHORTS:
        master_df = load_master(cohort)

        current_state_table = build_current_state_table(master_df)
        output_path = PROCESSED_DIR / "current_state" / cohort / "model_table.parquet"
        export_plan[output_path] = current_state_table

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

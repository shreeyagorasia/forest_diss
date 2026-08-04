# Run as: python -m data_processing.clean_master_data
#
# Plain-Python replacement for lidar_years_all_data_cleaning.ipynb's actual cleaning/filtering/
# export logic (that notebook is archived at legacy/2026-07-28/, its many diagnostic/plotting
# cells are not reproduced here -- they never changed the exported data, they only checked it).
# Reads the raw GPKG, applies the same six-stage cleaning funnel, and writes
# data/processed/master/clean_master_4survey.parquet / clean_master_6survey.parquet.
#
# Two changes from the retired notebook version, both explicit decisions (2026-07-28, see
# progress_notes.md):
#   1. Target column is now `elev_percentile_95th` (raw, no correction), not `Top_Height99`.
#      `Top_Height99`, `Vol99`, and `GYCspec99` (the whole "99th percentile" family) are dropped
#      entirely -- not used anywhere going forward.
#   2. `Top_Height95` (= elev_percentile_95th * 1.1) is kept, but ONLY as an ingredient for the
#      pre-computed FR-inventory evaluation fields `Vol95`/`GYCspec95` (useful to foresters when
#      reporting results) -- it is never a model target or feature.
#   3. `yldc` stays in the master export (it's still a real, recorded FR inventory field, useful
#      for audit/stratification) -- it is removed as a MODEL FEATURE separately, in each model's
#      own feature-column list, not here.

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPKG_PATH = PROJECT_ROOT / "data" / "raw" / "LiDAR_Years_All_7jul.gpkg"
LAYER_ALL = "LiDAR_Years"
MASTER_DIR = PROJECT_ROOT / "data" / "processed" / "master"

COHORTS = {
    "four_year": [2008, 2012, 2021, 2023],
    "six_year": [2002, 2006, 2008, 2012, 2021, 2023],
}
TARGET_SPECIES = "SS"
# AGE_MIN/AGE_MAX, HEIGHT_MAX, MIN_PLYR: generous sanity bounds (catch a corrupt/impossible
# record, e.g. a negative age or a 300-year-old planting year), not domain-derived limits --
# no citation grounds these specific numbers, they're wide enough to never bind on real Sitka
# spruce data at this site (2026-07-30 note, not yet changed: worth confirming this stage
# actually removes 0 rows on the current data, the same way other filter stages already print
# their own counts).
AGE_MIN, AGE_MAX = 0, 200
HEIGHT_MIN_EXCL = 0.0
HEIGHT_MAX = 60.0
MIN_PLYR = 1850

# The modelling target -- raw, unadjusted 95th-percentile LiDAR height. Kept under its raw
# source name (matches this project's convention for other merged-in raw fields, e.g. whcl).
TARGET_COLUMN = "elev_percentile_95th"

# Columns kept in the master export. Wider than any single model's feature list -- evaluation
# and audit columns (Vol95, GYCspec95, area, p1-p5, g1, g3) are kept for forestry-relevant
# reporting, never as model predictors (see FEATURE_COLUMNS lists in each model's own module).
MASTER_COLUMNS = [
    # Plot, survey and spatial grouping identifiers.
    "identification", "LiDAR_year", "blk", "cpmt", "scpt",
    # Recorded stand information.
    "spis", "plyr", "yldc", "Thin", "last_thinn", "next_thin_",
    # Primary modelling target.
    "Age", TARGET_COLUMN, "CanopyCover", "LAI",
    # Top_Height95 is kept ONLY as the ingredient Vol95/GYCspec95's existing formulas were
    # already computed from (by the FR inventory, not by this script) -- never a target/feature.
    "Top_Height95",
    # Audit/evaluation outcomes and formula ingredients, useful to foresters when reporting
    # results (see documentation/progress_notes.md for the Vol95/GYCspec95 formulas) -- not
    # model features. Vol99/GYCspec99 (the Top_Height99 family) are deliberately NOT kept.
    "Vol95", "GYCspec95", "area",
    "p1", "p2", "p3", "p4", "p5", "g1", "g3",
    # Wind hazard class -- audit/sensitivity only, never a baseline model feature.
    "whcl",
]


def clean_cohort(data, cohort_years):
    # Coarse-to-fine plot-level cleaning, preserving balanced trajectories: a plot is only kept
    # once every one of its survey rows (not just some) passes each stage's condition. Returns
    # the cleaned rows, a funnel summary (row/plot counts at each stage), and each stage's
    # surviving plot IDs (the funnel/stage_ids are diagnostic only -- not needed by the script's
    # own export, kept here so this function stays a faithful copy of the notebook's original).
    summary_rows = []
    stage_ids = {}

    def save_stage(label, current_data):
        summary_rows.append({
            "step": label,
            "rows": len(current_data),
            "unique_plots": current_data["identification"].nunique(),
        })
        stage_ids[label] = set(current_data["identification"].unique())

    def keep_plots_where_all(current_data, row_condition):
        passes_by_plot = row_condition.groupby(current_data["identification"]).all()
        retained_ids = passes_by_plot[passes_by_plot].index
        return current_data[current_data["identification"].isin(retained_ids)].copy()

    current = data.copy()
    save_stage("0. Raw dataset", current)

    current = current[current["LiDAR_year"].isin(cohort_years)].copy()
    save_stage("1. Cohort years", current)

    survey_count = current.groupby("identification")["LiDAR_year"].nunique()
    complete_ids = survey_count[survey_count.eq(len(cohort_years))].index
    current = current[current["identification"].isin(complete_ids)].copy()
    save_stage("2. Complete survey coverage", current)

    current = keep_plots_where_all(current, current["spis"].eq(TARGET_SPECIES))
    save_stage("3. Sitka spruce", current)

    planting_year_ok = current["plyr"].ge(MIN_PLYR) & current["plyr"].lt(current["LiDAR_year"])
    current = keep_plots_where_all(current, planting_year_ok)
    save_stage("4. Valid planting year", current)

    current = keep_plots_where_all(current, current["Age"].between(AGE_MIN, AGE_MAX))
    save_stage("5. Plausible age", current)

    # Height plausibility now checks the TARGET column directly (elev_percentile_95th), not
    # Top_Height95 -- same 0-60m bounds, kept as-is per an explicit decision (2026-07-28): the
    # raw column is systematically ~9% smaller than the old x1.1-adjusted one, so this is a
    # slightly more permissive filter than before, which is fine (60m was already a generous
    # ceiling for Sitka spruce top height).
    height_ok = current[TARGET_COLUMN].gt(HEIGHT_MIN_EXCL) & current[TARGET_COLUMN].le(HEIGHT_MAX)
    current = keep_plots_where_all(current, height_ok)
    save_stage("6. Plausible height", current)

    funnel = pd.DataFrame(summary_rows)
    return current, funnel, stage_ids


def add_export_metrics(exported):
    # Simple derived metrics for modelling and audit tables, added after cleaning so every
    # exported table inherits the same balanced cohort and the same assumptions.
    exported = exported.sort_values(["identification", "LiDAR_year"]).copy()

    valid_last_thinning = exported["last_thinn"].gt(0) & exported["last_thinn"].le(exported["LiDAR_year"])
    # Thin is a per-survey feature: it must not reveal a thinning event that happens later.
    exported["Thin"] = valid_last_thinning.astype(int)
    exported["time_since_thinning"] = np.where(
        valid_last_thinning, exported["LiDAR_year"] - exported["last_thinn"], np.nan,
    )
    exported["time_since_thinning_missing"] = exported["time_since_thinning"].isna()
    exported["recent_thinning_5yr"] = valid_last_thinning & exported["time_since_thinning"].le(5)
    exported["recent_thinning_10yr"] = valid_last_thinning & exported["time_since_thinning"].le(10)

    exported["thinning_status"] = np.select(
        [
            exported["last_thinn"].eq(0),
            valid_last_thinning & exported["time_since_thinning"].le(5),
            valid_last_thinning & exported["time_since_thinning"].between(6, 10),
            valid_last_thinning & exported["time_since_thinning"].gt(10),
            exported["last_thinn"].gt(exported["LiDAR_year"]),
        ],
        ["never_thinned", "recent_0_5yr", "mid_6_10yr", "old_10plus_yr", "not_yet_thinned_at_survey"],
        default="unknown_timing",
    )

    # unknown_timing is the catch-all for a corrupt/unexpected last_thinn value (e.g. negative) --
    # every other filter/bucketing stage in this file prints how many rows it affects; this one
    # didn't (2026-07-30 fix). Currently 0 rows in both cohorts, but a real, silent population
    # could hide here undetected if the raw data ever changes -- printed loudly (not just at high
    # count) so it's never missed by scrolling past a quiet log line.
    n_unknown_timing = int((exported["thinning_status"] == "unknown_timing").sum())
    print(f"  thinning_status = 'unknown_timing' (corrupt/unexpected last_thinn): {n_unknown_timing:,} rows")
    if n_unknown_timing > 0:
        print(
            f"  WARNING: {n_unknown_timing:,} rows have a corrupt/unexpected last_thinn value -- "
            "check the raw data before trusting thinning_status as a feature."
        )

    exported["previous_lidar_year"] = exported.groupby("identification")["LiDAR_year"].shift()
    exported["survey_interval_years"] = exported["LiDAR_year"] - exported["previous_lidar_year"]
    exported["age_band_5yr"] = (np.floor(exported["Age"] / 5) * 5).astype(int)

    return exported


def make_master_export(cleaned_data, expected_surveys):
    exported = cleaned_data[MASTER_COLUMNS].copy()
    exported = add_export_metrics(exported)

    # Stops execution if a future cleaning edit breaks the balanced-cohort guarantee.
    rows_per_plot = exported.groupby("identification").size()
    assert rows_per_plot.eq(expected_surveys).all()
    assert not exported.duplicated(["identification", "LiDAR_year"]).any()

    return exported


def clean_master_data():
    print(f"Reading {GPKG_PATH} ...")
    if not GPKG_PATH.exists():
        raise FileNotFoundError(f"GeoPackage not found: {GPKG_PATH}")
    df = gpd.read_file(GPKG_PATH, layer=LAYER_ALL, ignore_geometry=True)
    print(f"  Read {len(df):,} rows, {df.shape[1]} columns")

    d4, funnel4, _ = clean_cohort(df, COHORTS["four_year"])
    d6, funnel6, _ = clean_cohort(df, COHORTS["six_year"])
    print("\nFour-survey cleaning funnel:")
    print(funnel4.to_string(index=False))
    print("\nSix-survey cleaning funnel:")
    print(funnel6.to_string(index=False))

    clean_export_4 = make_master_export(d4, expected_surveys=4)
    clean_export_6 = make_master_export(d6, expected_surveys=6)

    MASTER_DIR.mkdir(parents=True, exist_ok=True)
    for filename, table in [
        ("clean_master_4survey.parquet", clean_export_4),
        ("clean_master_6survey.parquet", clean_export_6),
    ]:
        output_path = MASTER_DIR / filename
        table.to_parquet(output_path, index=False)
        print(f"\nSaved {len(table):,} rows, {table['identification'].nunique():,} plots -> {output_path}")


if __name__ == "__main__":
    clean_master_data()

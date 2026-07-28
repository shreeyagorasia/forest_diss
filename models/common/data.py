from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Both the Chapman-Richards and average-by-age baselines only need these columns, out of the one
# consolidated table every model now reads (see data_processing/export_model_tables.py -- one
# model_table.parquet per cohort replaced five near-duplicate per-model files 2026-07-28).
COLUMNS_NEEDED = ["identification", "LiDAR_year", "blk", "cpmt", "Age", "yldc", "elev_percentile_95th"]


def load_cohort_data(cohort):
    # cohort is either "4survey" or "6survey"
    path = PROJECT_ROOT / "data" / "processed" / "current_state" / cohort / "model_table.parquet"
    full_table = pd.read_parquet(path)
    return full_table[COLUMNS_NEEDED].copy()


def load_model_table(cohort):
    # Returns every column in the one consolidated table -- different models select their own
    # feature subset from it (e.g. models/rf_baseline/rf_baseline.py's own FEATURE_COLUMNS).
    path = PROJECT_ROOT / "data" / "processed" / "current_state" / cohort / "model_table.parquet"
    return pd.read_parquet(path)


def filter_data(df, maturity_age_min=30, reference_year=2023, yldc_min=2, yldc_max=50):
    # Age is now a per-PLOT gate, not a per-row threshold. Both the 4survey
    # and 6survey cohorts give every plot the exact same fixed set of survey
    # years (e.g. 2008/2012/2021/2023), so every plot has one real, measured
    # row at reference_year -- no need to estimate "age in 2023" by adding
    # (2023 - LiDAR_year) to some other row's Age.
    #
    # We look up each plot's actual Age at its reference_year survey. If that
    # is < maturity_age_min, the WHOLE plot is dropped -- every survey year,
    # not just the young one -- because a plot that fails the "will this be a
    # mature stand by 2023" test shouldn't contribute rows at any age. If it
    # passes, every one of its survey years is kept, including early rows
    # where the plot's raw Age is well under maturity_age_min (e.g. Age=8 in
    # 2008), so growth-curve models like Chapman-Richards still see the full
    # trajectory for stands that are mature today.
    rows_before = len(df)
    print(f"  Rows before filtering: {rows_before:,}")

    reference_rows = df.loc[df["LiDAR_year"] == reference_year, ["identification", "Age"]]
    missing_reference = set(df["identification"]) - set(reference_rows["identification"])
    if missing_reference:
        print(
            f"  WARNING: {len(missing_reference):,} plots have no {reference_year} survey row "
            "and cannot be checked against the maturity filter -- they are dropped."
        )

    mature_plot_ids = set(reference_rows.loc[reference_rows["Age"] >= maturity_age_min, "identification"])

    after_age_filter = df[df["identification"].isin(mature_plot_ids)]
    removed_by_age = rows_before - len(after_age_filter)
    print(
        f"  Removed by Age < {maturity_age_min} at the {reference_year} survey "
        f"(whole plot dropped): {removed_by_age:,} rows"
    )

    after_yldc_filter = after_age_filter[
        (after_age_filter["yldc"] >= yldc_min) & (after_age_filter["yldc"] <= yldc_max)
    ]
    removed_by_yldc = len(after_age_filter) - len(after_yldc_filter)
    print(f"  Removed by yield class between {yldc_min} and {yldc_max}: {removed_by_yldc:,} rows")

    rows_after = len(after_yldc_filter)
    print(f"  Rows after filtering: {rows_after:,}")

    # Flag it loudly if a filter removed a suspiciously large share of rows.
    if rows_before > 0 and (rows_before - rows_after) / rows_before > 0.3:
        percent_removed = (rows_before - rows_after) / rows_before * 100
        print(f"  WARNING: filtering removed {percent_removed:.1f}% of rows — check this is expected.")

    return after_yldc_filter.copy()

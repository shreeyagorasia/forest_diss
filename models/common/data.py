from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Both the Chapman-Richards and average-by-age baselines only need these
# columns. linear_baseline.parquet is the smallest exported table that has
# yldc sitting alongside Age and Top_Height99, so we read from there.
COLUMNS_NEEDED = ["identification", "LiDAR_year", "blk", "Age", "yldc", "Top_Height99"]


def load_cohort_data(cohort):
    # cohort is either "4survey" or "6survey"
    path = PROJECT_ROOT / "data" / "processed" / "current_state" / cohort / "linear_baseline.parquet"
    full_table = pd.read_parquet(path)
    return full_table[COLUMNS_NEEDED].copy()


def load_model_table(cohort, table_name):
    # table_name is one of "cr_age", "linear_baseline", "rf_baseline" -- the
    # exported table names under data/processed/current_state/<cohort>/.
    # Unlike load_cohort_data, this returns every column in the file, since
    # different models need different feature columns.
    path = PROJECT_ROOT / "data" / "processed" / "current_state" / cohort / f"{table_name}.parquet"
    return pd.read_parquet(path)


def filter_data(df, age_min=20, yldc_min=2, yldc_max=50):
    # Apply the same filters the prior dissertation used, and print how many
    # rows each filter step removes so mistakes are easy to spot.
    rows_before = len(df)
    print(f"  Rows before filtering: {rows_before:,}")

    after_age_filter = df[df["Age"] >= age_min]
    removed_by_age = rows_before - len(after_age_filter)
    print(f"  Removed by Age >= {age_min}: {removed_by_age:,} rows")

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

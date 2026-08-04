# Purpose: shared data loading for "Stage 2" -- the per-plot growth-curve-vs-yldc model family
# (see documentation/experiment_log.md's Stage 2 section). Mirrors
# models/spatial_attribution/data.py's shape: one place every Stage 2 model/diagnostic reads
# from, instead of five separate ad-hoc loaders that could drift apart.

from pathlib import Path

import pandas as pd

from models.common.data import filter_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_growth_curve_table(cohort):
    # One row per plot per survey year -- built by data_processing/export_growth_curve_tables.py.
    # Not yet filtered (same convention as model_table.parquet) -- callers apply filter_data()
    # themselves, same as every other model in this project.
    path = PROJECT_ROOT / "data" / "processed" / "growth_curve" / cohort / "growth_curve_table.parquet"
    return pd.read_parquet(path)


def load_filtered_growth_curve_table(cohort):
    # Same maturity_age_min=30 / yldc 2-50 gate used everywhere else in this project
    # (models/common/data.py::filter_data()) -- reused here, not reimplemented, so Stage 2 works
    # on the exact same plot population Stage 1 already filtered to.
    table = load_growth_curve_table(cohort)
    return filter_data(table)

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Built once by notebooks/environmental_data/aux_data_resolution_check.ipynb's final export
# cell -- one row per plot, every environmental variable already extracted/derived there, plus
# both cohorts' mean Chapman-Richards residual already computed. This script never re-downloads
# or re-derives a feature from a raw raster/API; it only reads this one file, matching the
# repo's "notebook builds the messy validated features once, plain code reads the export from
# then on" convention (see data_processing/export_model_tables.py for the same pattern used
# elsewhere in this repo).
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "environmental" / "plot_environmental_features.parquet"


def load_environmental_features():
    return pd.read_parquet(FEATURES_PATH)


# Base feature names that get a separate _4survey/_6survey column pair in the export, instead of
# one shared column -- mirrors mean_cr_residual's own pattern (see
# aux_data_resolution_check.ipynb): each cohort's value is averaged across only that cohort's own
# real survey years, not a single value shared across cohorts. Everything else in the export
# (terrain, static climate/soil, etc.) doesn't depend on which years a plot was surveyed in, so it
# only ever gets one column. 2026-07-30: tas_mean/groundfrost_mean added, same reasoning as the
# residual target -- see models/common/download_haduk_multi_year.py.
COHORT_SPECIFIC_COLUMNS = ["mean_cr_residual", "tas_mean", "groundfrost_mean"]


def load_plots_for_cohort(cohort):
    # One row per plot: every environmental feature, plus plain (not cohort-suffixed) versions of
    # COHORT_SPECIFIC_COLUMNS for this specific cohort -- renamed from the cohort-specific columns
    # the notebook exported, so the rest of this package doesn't need to know which cohort it's
    # working with. 6survey is a strict subset of 4survey's plots (see the aux notebook's own
    # setup check), so 6survey's residual column is NaN for every plot outside that subset --
    # dropped here via dropna, since those rows have no real target to fit or evaluate against.
    features_df = load_environmental_features()
    other_cohort = "6survey" if cohort == "4survey" else "4survey"

    plots_df = features_df.dropna(subset=[f"mean_cr_residual_{cohort}"]).copy()

    rename_map = {f"{base}_{cohort}": base for base in COHORT_SPECIFIC_COLUMNS}
    drop_columns = [f"{base}_{other_cohort}" for base in COHORT_SPECIFIC_COLUMNS]

    plots_df = plots_df.rename(columns=rename_map)
    plots_df = plots_df.drop(columns=drop_columns)
    return plots_df

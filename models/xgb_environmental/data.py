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


def load_plots_for_cohort(cohort):
    # One row per plot: every environmental feature, plus a plain "mean_cr_residual" target
    # column for this specific cohort (renamed from the cohort-specific column the notebook
    # exported, so the rest of this package doesn't need to know which cohort it's working with).
    # 6survey is a strict subset of 4survey's plots (see the aux notebook's own setup check), so
    # 6survey's residual column is NaN for every plot outside that subset -- dropped here via
    # dropna, since those rows have no real target to fit or evaluate against.
    features_df = load_environmental_features()
    residual_col = f"mean_cr_residual_{cohort}"
    other_residual_col = "mean_cr_residual_6survey" if cohort == "4survey" else "mean_cr_residual_4survey"

    plots_df = features_df.dropna(subset=[residual_col]).copy()
    plots_df = plots_df.rename(columns={residual_col: "mean_cr_residual"})
    plots_df = plots_df.drop(columns=[other_residual_col])
    return plots_df

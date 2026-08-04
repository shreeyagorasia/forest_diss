# Purpose: two of the LLM Council's remaining pre-SHAP checks (documentation/experiment_log.md,
# 2026-08-04 variable council entry) -- both flagged as "waved through without an actual number"
# by multiple advisors, so this computes the real numbers rather than assuming an answer:
#   1. Are TPI at 100m (native), 250m, 500m, and local_relief_500m redundant with each other, or
#      genuinely distinct spatial scales? (Only GWA-vs-GWA correlation had been checked before.)
#   2. Is inverse_slope_proxy really a near-exact duplicate of slope_degrees, as its own code
#      comment in xgb_environmental.py claims -- confirmed with a number, not narrative.

import numpy as np
import pandas as pd

from models.xgb_environmental.data import load_environmental_features

TPI_COLUMNS = ["tpi", "tpi_250m", "tpi_500m", "local_relief_500m"]
SLOPE_COLUMNS = ["slope_degrees", "inverse_slope_proxy"]


def compute_correlation_matrix(columns, method="spearman"):
    environment = load_environmental_features()
    available_columns = [column for column in columns if column in environment.columns]
    missing_columns = [column for column in columns if column not in environment.columns]
    correlation = environment[available_columns].corr(method=method)
    return correlation, missing_columns


def summarize_pairwise(correlation):
    # Flattens the correlation matrix into one row per pair (excluding a variable with itself),
    # sorted by absolute correlation -- easier to read than a full matrix when there are only a
    # handful of columns.
    pairs = []
    for i, row_name in enumerate(correlation.index):
        for j, col_name in enumerate(correlation.columns):
            if j <= i:
                continue
            pairs.append({"variable_1": row_name, "variable_2": col_name, "spearman_rho": correlation.iloc[i, j]})
    return pd.DataFrame(pairs).sort_values("spearman_rho", key=lambda values: values.abs(), ascending=False)

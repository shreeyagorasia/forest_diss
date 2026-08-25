# Purpose: explain the confirmed 4survey terrain/wind signal cheaply. ONE model fit on the
# full cleaned population (not a CV sweep), using a feature list already justified by checks
# already on disk, not a new expensive comparison:
#   - inverse_slope_proxy dropped (confirmed exact -1.0 duplicate of slope_degrees,
#     correlation_screen.py)
#   - gwa_wind_speed_10m -> gwa_wind_speed_50m (confirmed doesn't hurt, physically more correct,
#     run_wind_height_swap_check.py)
#   - tpi_500m and local_relief_500m added (confirmed NOT redundant with native tpi,
#     correlation_screen.py: rho=0.62 native-vs-500m, local_relief essentially uncorrelated with
#     any TPI scale)
# No held-out split here. This is an INTERPRETATION model (what did a model trained on
# everything lean on), not a performance-evaluation model (that question is already answered by
# spatial_cv_check.py). Standard practice: CV for performance, full-data fit for interpretation.

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr

from models.growth_curve_attribution.run_wind_height_swap_check import SWAPPED_COLUMNS
from models.growth_curve_attribution.scale_comparison_check import TARGET, build_plot_level_table, merge_environmental_features
from models.xgb_environmental.xgb_environmental import fit_with_columns as xgb_fit

FINAL_FEATURE_COLUMNS = (
    [column for column in SWAPPED_COLUMNS if column != "inverse_slope_proxy"]
    + ["tpi_500m", "local_relief_500m"]
)


def build_full_table(cohort):
    plot_table = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
    plot_table_with_features, available_columns = merge_environmental_features(plot_table, feature_columns=FINAL_FEATURE_COLUMNS)
    plot_table_with_features = plot_table_with_features.dropna(subset=available_columns + [TARGET])
    return plot_table_with_features, available_columns


def spearman_with_target(table, feature_columns):
    rows = []
    for column in feature_columns:
        rho, _ = spearmanr(table[column], table[TARGET])
        rows.append({"variable": column, "spearman_rho": rho, "abs_rho": abs(rho)})
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False)


def fit_and_explain(table, feature_columns):
    model = xgb_fit(table, feature_columns, target_col=TARGET, n_estimators=500, max_depth=4, learning_rate=0.04)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(table[feature_columns])

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.DataFrame({"variable": feature_columns, "mean_abs_shap": mean_abs_shap}).sort_values(
        "mean_abs_shap", ascending=False
    )
    return model, importance

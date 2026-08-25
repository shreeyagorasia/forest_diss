# Purpose: re-run the wind/terrain representation comparison already sketched in
# notebooks/growth_curve_attribution/local_growth_curve_grouped_importance.ipynb (the "Compare
# alternative wind and terrain representations" section), but on the CLEANED plot population
# (clearfell-like/measurement-inconsistent plots excluded before the y_max fit. See
# disturbance_checks.summarize_plot_disturbance_status()). That notebook comparison found local
# shelter (topex+windward_topex+whcl) beating every GWA wind variant, and 500m TPI beating other
# terrain scales. But it was computed on the UNCLEANED target, which the 2026-08-03 cleaning
# comparison found materially changes the result (4survey's R2 roughly halved after cleaning).
# This checks whether the "local shelter wins" conclusion survives that same cleaning.

import pandas as pd

from models.common.splits import SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, SPLIT_SEED, spatial_block_split
from models.elasticnet_environmental.elasticnet_environmental import drop_rows_with_missing_features
from models.elasticnet_environmental.elasticnet_environmental import fit_with_columns as elasticnet_fit
from models.elasticnet_environmental.elasticnet_environmental import predict_with_columns as elasticnet_predict
from models.common.metrics import compute_metrics
from models.growth_curve_attribution.scale_comparison_check import TARGET, build_plot_level_table
from models.xgb_environmental.data import load_environmental_features
from models.xgb_environmental.xgb_environmental import TERRAIN_AND_WIND_COLUMNS
from models.xgb_environmental.xgb_environmental import fit_with_columns as xgb_fit
from models.xgb_environmental.xgb_environmental import predict_with_columns as xgb_predict

# Same additional candidate columns the notebook compares. Not in the narrow, established
# TERRAIN_AND_WIND_COLUMNS list, but already extracted and present in the environmental export
# (confirmed directly: 0% missing for all seven, checked before writing this script).
EXTRA_WIND_TERRAIN_COLUMNS = [
    "gwa_weibull_a_50m", "gwa_weibull_k_50m", "gwa_wind_p95_50m", "gwa_prob_above_critical_50m",
    "tpi_250m", "tpi_500m", "local_relief_500m",
]

# base_columns: TERRAIN_AND_WIND_COLUMNS minus the wind-specific ones. Matches the notebook's
# own "base_without_wind" idea, so adding a wind representation is a fair one-at-a-time test
# against a fixed non-wind foundation.
WIND_COLUMNS_IN_BASE_SET = {"topex", "windward_topex", "whcl", "gwa_wind_speed_10m"}
NON_WIND_BASE_COLUMNS = [column for column in TERRAIN_AND_WIND_COLUMNS if column not in WIND_COLUMNS_IN_BASE_SET]

WIND_REPRESENTATIONS = {
    "no wind variables": [],
    "local shelter only": ["topex", "windward_topex", "whcl"],
    "GWA mean wind at 10m": ["gwa_wind_speed_10m"],
    "GWA Weibull A+k at 50m": ["gwa_weibull_a_50m", "gwa_weibull_k_50m"],
    "GWA p95+k at 50m": ["gwa_wind_p95_50m", "gwa_weibull_k_50m"],
    "GWA probability above 20 m/s at 50m": ["gwa_prob_above_critical_50m"],
    "local shelter plus GWA A+k at 50m": ["topex", "windward_topex", "whcl", "gwa_weibull_a_50m", "gwa_weibull_k_50m"],
}

NEW_TERRAIN_COLUMNS = ["tpi_250m", "tpi_500m", "local_relief_500m"]
TERRAIN_BASE_COLUMNS = [column for column in NON_WIND_BASE_COLUMNS if column not in NEW_TERRAIN_COLUMNS]
TERRAIN_REPRESENTATIONS = {
    "no new multiscale terrain": [],
    "TPI 250m": ["tpi_250m"],
    "TPI 500m": ["tpi_500m"],
    "local relief 500m": ["local_relief_500m"],
    "TPI 250m plus local relief 500m": ["tpi_250m", "local_relief_500m"],
    "all new multiscale terrain": NEW_TERRAIN_COLUMNS,
}


def build_cleaned_table_with_extra_columns(cohort, split_seed=SPLIT_SEED):
    plot_table = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
    environment = load_environmental_features()
    all_candidate_columns = list(dict.fromkeys(TERRAIN_AND_WIND_COLUMNS + EXTRA_WIND_TERRAIN_COLUMNS))
    available_columns = [column for column in all_candidate_columns if column in environment.columns]

    merged = plot_table.merge(
        environment[["identification"] + available_columns], on="identification", how="left", validate="one_to_one"
    )
    merged["split"] = spatial_block_split(
        merged, block_col=SPATIAL_BLOCK_COL, buffer_distance=SPATIAL_BUFFER_METRES,
        coordinates_df=merged[["identification", "x", "y"]], seed=split_seed,
    )
    return merged


def compare_representations(train, val, test, representations, base_columns, family):
    # BUG FIX (2026-08-04): never run/reported, but had the same leak found and fixed elsewhere
    # this session. Val must only drive XGBoost's early stopping, not double as the reported
    # evaluation set. Both methods are now scored on the untouched "test" partition.
    rows = []
    for representation, added_columns in representations.items():
        columns = list(dict.fromkeys(base_columns + added_columns))
        train_rows = drop_rows_with_missing_features(train, columns)
        val_rows = drop_rows_with_missing_features(val, columns)
        test_rows = drop_rows_with_missing_features(test, columns)

        elastic = elasticnet_fit(train_rows, columns, target_col=TARGET)
        elastic_predictions = elasticnet_predict(test_rows, elastic, columns)
        elastic_metrics = compute_metrics(test_rows[TARGET], elastic_predictions)
        rows.append({"family": family, "representation": representation, "method": "Elastic Net", "n_features": len(columns), **elastic_metrics})

        tree = xgb_fit(train_rows, columns, val_df=val_rows, target_col=TARGET, n_estimators=500, max_depth=4, learning_rate=0.04)
        tree_predictions = xgb_predict(test_rows, tree, columns)
        tree_metrics = compute_metrics(test_rows[TARGET], tree_predictions)
        rows.append({"family": family, "representation": representation, "method": "XGBoost", "n_features": len(columns), **tree_metrics})

    result = pd.DataFrame(rows)
    for method in result["method"].unique():
        baseline_mask = (result["method"] == method) & (result["representation"].str.startswith("no "))
        baseline_r2 = result.loc[baseline_mask, "r2"].iloc[0]
        result.loc[result["method"] == method, "r2_change_from_baseline"] = result.loc[result["method"] == method, "r2"] - baseline_r2
    return result


def run_for_cohort(cohort, split_seed=SPLIT_SEED):
    merged = build_cleaned_table_with_extra_columns(cohort, split_seed=split_seed)
    train = merged[merged["split"] == "train"]
    val = merged[merged["split"] == "val"]
    test = merged[merged["split"] == "test"]

    wind_comparison = compare_representations(train, val, test, WIND_REPRESENTATIONS, NON_WIND_BASE_COLUMNS, "wind")
    terrain_comparison = compare_representations(train, val, test, TERRAIN_REPRESENTATIONS, TERRAIN_BASE_COLUMNS, "terrain")
    wind_comparison["cohort"] = cohort
    terrain_comparison["cohort"] = cohort
    return wind_comparison, terrain_comparison

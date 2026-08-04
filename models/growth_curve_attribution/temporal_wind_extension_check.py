"""Test whether interval wind metrics help AV2 when converted into exposure interactions.

Raw MIDAS interval metrics are constant within a cohort under the current design, so they cannot
explain cross-plot variation in `local_y_max_difference` on their own. This script keeps the AV2
target unchanged and tests the only coherent extension compatible with that target: interactions
between cohort-level storminess summaries and plot-level exposure variables.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.growth_curve_attribution.explain_signal import FINAL_FEATURE_COLUMNS
from models.growth_curve_attribution.scale_comparison_check import build_plot_level_table, merge_environmental_features
from models.growth_curve_attribution.spatial_cv_check import run_spatial_cv, summarize_spatial_cv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIDAS_INTERVAL_PATH = PROJECT_ROOT / "data" / "processed" / "environmental" / "midas_wind_interval_metrics.parquet"

COHORTS = ["4survey", "6survey"]
METHODS = {"elastic_net_predicted": "Elastic Net", "xgboost_predicted": "XGBoost"}
DEFAULT_THRESHOLD_MS = 25.0

TEMPORAL_BASE_COLUMNS = [
    "midas_max_gust_ms",
    "midas_gust_p95_ms",
    "midas_mean_wind_p95_ms",
    "midas_hours_above_critical_per_year",
    "midas_cumulative_gust_excess_per_year",
    "midas_independent_storm_count_per_year",
    "midas_time_since_last_major_storm_days",
]

EXPOSURE_COLUMNS = ["topex", "windward_topex", "whcl", "gwa_wind_speed_50m"]


def cohort_intervals_from_growth_rows(cohort: str) -> pd.DataFrame:
    growth_rows = (
        pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "growth_curve" / cohort / "growth_curve_table.parquet")
        [["previous_lidar_year", "LiDAR_year", "survey_interval_years"]]
        .dropna()
        .drop_duplicates()
        .sort_values(["previous_lidar_year", "LiDAR_year"])
    )
    return growth_rows


def build_cohort_storm_profile(cohort: str, threshold_ms: float = DEFAULT_THRESHOLD_MS) -> dict[str, float]:
    interval_metrics = pd.read_parquet(MIDAS_INTERVAL_PATH)
    interval_metrics = interval_metrics.loc[interval_metrics["midas_gust_threshold_ms"] == threshold_ms].copy()
    interval_metrics = interval_metrics.loc[interval_metrics["midas_observation_coverage"] >= 0.8].copy()

    if interval_metrics.empty:
        raise ValueError("No MIDAS interval rows remain after threshold and coverage filtering")

    cohort_intervals = cohort_intervals_from_growth_rows(cohort)
    merged = interval_metrics.merge(
        cohort_intervals,
        on=["previous_lidar_year", "LiDAR_year"],
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError(f"No MIDAS interval rows matched the {cohort} survey intervals")

    station_weights = 1.0 / merged["station_distance_km"].clip(lower=1.0)
    interval_weights = merged["survey_interval_years"].astype(float)
    weights = station_weights * interval_weights

    profile = {
        "midas_max_gust_ms": float(merged["midas_max_gust_ms"].max()),
        "midas_gust_p95_ms": float(np.average(merged["midas_gust_p95_ms"], weights=weights)),
        "midas_mean_wind_p95_ms": float(np.average(merged["midas_mean_wind_p95_ms"], weights=weights)),
        "midas_hours_above_critical_per_year": float(np.average(merged["midas_hours_above_critical_per_year"], weights=weights)),
        "midas_cumulative_gust_excess_per_year": float(np.average(merged["midas_cumulative_gust_excess_per_year"], weights=weights)),
        "midas_independent_storm_count_per_year": float(np.average(merged["midas_independent_storm_count_per_year"], weights=weights)),
        "midas_time_since_last_major_storm_days": float(np.average(merged["midas_time_since_last_major_storm_days"], weights=weights)),
    }
    return profile


def add_temporal_wind_columns(table: pd.DataFrame, cohort: str, threshold_ms: float = DEFAULT_THRESHOLD_MS):
    table = table.copy()
    profile = build_cohort_storm_profile(cohort, threshold_ms=threshold_ms)
    for column, value in profile.items():
        table[column] = value

    interaction_columns = []
    for temporal_column in TEMPORAL_BASE_COLUMNS:
        for exposure_column in EXPOSURE_COLUMNS:
            interaction_column = f"{temporal_column}__x__{exposure_column}"
            table[interaction_column] = table[temporal_column] * table[exposure_column]
            interaction_columns.append(interaction_column)
    return table, TEMPORAL_BASE_COLUMNS + interaction_columns, profile


def run_for_cohort(cohort: str, threshold_ms: float = DEFAULT_THRESHOLD_MS):
    plot_table = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
    plot_table_with_features, available_columns = merge_environmental_features(
        plot_table, feature_columns=FINAL_FEATURE_COLUMNS
    )

    temporal_table, temporal_columns, profile = add_temporal_wind_columns(
        plot_table_with_features, cohort, threshold_ms=threshold_ms
    )
    varying_temporal_columns = [
        column for column in temporal_columns if temporal_table[column].nunique(dropna=False) > 1
    ]

    comparison_sets = {
        "terrain_wind_final": available_columns,
        "terrain_wind_plus_temporal_wind_interactions": available_columns + varying_temporal_columns,
    }

    rows = []
    for label, columns in comparison_sets.items():
        out_of_fold_predictions, _ = run_spatial_cv(temporal_table, columns)
        for predicted_col, method_name in METHODS.items():
            summary = summarize_spatial_cv(out_of_fold_predictions, predicted_col)
            rows.append({
                "cohort": cohort,
                "feature_set": label,
                "method": method_name,
                "n_features": len(columns),
                "threshold_ms": threshold_ms,
                "n_temporal_columns_added": len(columns) - len(available_columns),
                "pooled_r2": summary["pooled_r2"],
                "per_fold_r2_mean": summary["per_fold_r2_mean"],
                "per_fold_r2_std": summary["per_fold_r2_std"],
                **profile,
            })
    return pd.DataFrame(rows)


def main():
    all_results = []
    for cohort in COHORTS:
        print(f"\n===== {cohort}: temporal wind interaction extension =====")
        all_results.append(run_for_cohort(cohort))

    results = pd.concat(all_results, ignore_index=True)
    output_path = PROJECT_ROOT / "outputs" / "growth_curve_attribution" / "temporal_wind_extension_check.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    pd.set_option("display.width", 200)
    print()
    print(
        results[
            ["cohort", "feature_set", "method", "n_features", "n_temporal_columns_added", "pooled_r2", "per_fold_r2_mean", "per_fold_r2_std"]
        ].to_string(index=False)
    )
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()

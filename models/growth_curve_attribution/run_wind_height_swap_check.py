# Run as: python -m models.growth_curve_attribution.run_wind_height_swap_check
#
# The LLM Council's "one thing to do first" (documentation/experiment_log.md, 2026-08-04 variable
# council entry): gwa_wind_speed_10m sits in the established 16-column terrain/wind list, but this
# project has separately flagged 10m GWA wind as likely measuring wind WITHIN/BELOW a mature
# canopy -- a physically backwards proxy for a mature stand's actual crown-level exposure, versus
# 50m which sits above it. Does the confirmed 4survey signal (pooled R2=0.118 EN / 0.174 XGBoost,
# 5-fold spatial CV, established 16) survive swapping 10m for 50m, or was some of that signal
# riding on a known-contaminated variable?

import pandas as pd

from models.growth_curve_attribution.scale_comparison_check import build_plot_level_table, merge_environmental_features
from models.growth_curve_attribution.spatial_cv_check import run_spatial_cv, summarize_spatial_cv
from models.xgb_environmental.xgb_environmental import TERRAIN_AND_WIND_COLUMNS

COHORTS = ["4survey", "6survey"]
METHODS = {"elastic_net_predicted": "Elastic Net", "xgboost_predicted": "XGBoost"}

# The one swap the council asked for -- everything else about the established 16 stays untouched,
# so any R2 change is attributable to this single column, not a bundle of changes.
SWAPPED_COLUMNS = [
    "gwa_wind_speed_50m" if column == "gwa_wind_speed_10m" else column
    for column in TERRAIN_AND_WIND_COLUMNS
]


def run_for_cohort(cohort, feature_columns, label):
    plot_table = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
    plot_table_with_features, available_columns = merge_environmental_features(plot_table, feature_columns=feature_columns)

    out_of_fold_predictions, _ = run_spatial_cv(plot_table_with_features, available_columns)

    rows = []
    for predicted_col, method_name in METHODS.items():
        summary = summarize_spatial_cv(out_of_fold_predictions, predicted_col)
        rows.append({
            "cohort": cohort, "feature_set": label, "method": method_name,
            "pooled_r2": summary["pooled_r2"], "per_fold_r2_mean": summary["per_fold_r2_mean"],
            "per_fold_r2_std": summary["per_fold_r2_std"],
        })
    return pd.DataFrame(rows)


def main():
    all_results = []
    for cohort in COHORTS:
        print(f"\n===== {cohort}: established 16 (gwa_wind_speed_10m) =====")
        all_results.append(run_for_cohort(cohort, TERRAIN_AND_WIND_COLUMNS, "10m wind (established)"))

        print(f"\n===== {cohort}: swapped (gwa_wind_speed_50m) =====")
        all_results.append(run_for_cohort(cohort, SWAPPED_COLUMNS, "50m wind (swapped)"))

    results = pd.concat(all_results, ignore_index=True)
    pd.set_option("display.width", 160)
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

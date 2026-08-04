# Run as: python -m models.growth_curve_attribution.run_representation_cv_check
#
# Re-confirms two of the LLM Council's remaining pre-SHAP checks (documentation/experiment_log.md,
# 2026-08-04 variable council entry), but under the 5-fold spatial CV framework -- the council's
# peer review specifically flagged that no correlation/representation check had been run inside
# this project's own spatial CV, only on a single pooled/non-spatial split (which is exactly what
# the ORIGINAL "local shelter beats GWA" comparison did, on the uncleaned target, in the notebook).
#
#   1. Does "local shelter" (topex + windward_topex + whcl -- already inside the established,
#      wind-swapped 16) earn its place, or does the signal hold up fine without it?
#   2. Do the extra TPI scales (500m) and local_relief_500m -- confirmed genuinely NOT redundant
#      with native TPI by the correlation screen (rho=0.62 native-vs-500m, local_relief
#      essentially uncorrelated with any TPI scale) -- actually help under CV, not just "aren't
#      redundant on paper"?

import pandas as pd

from models.growth_curve_attribution.run_wind_height_swap_check import SWAPPED_COLUMNS
from models.growth_curve_attribution.scale_comparison_check import build_plot_level_table, merge_environmental_features
from models.growth_curve_attribution.spatial_cv_check import run_spatial_cv, summarize_spatial_cv

COHORTS = ["4survey", "6survey"]
METHODS = {"elastic_net_predicted": "Elastic Net", "xgboost_predicted": "XGBoost"}

LOCAL_SHELTER_COLUMNS = {"topex", "windward_topex", "whcl"}

FEATURE_SET_VARIANTS = {
    "swapped baseline (established 16, 50m wind)": SWAPPED_COLUMNS,
    "baseline minus local shelter": [c for c in SWAPPED_COLUMNS if c not in LOCAL_SHELTER_COLUMNS],
    "baseline plus GWA Weibull A+k at 50m": SWAPPED_COLUMNS + ["gwa_weibull_a_50m", "gwa_weibull_k_50m"],
    "baseline plus TPI 500m": SWAPPED_COLUMNS + ["tpi_500m"],
    "baseline plus local_relief_500m": SWAPPED_COLUMNS + ["local_relief_500m"],
    "baseline plus TPI 500m and local_relief_500m": SWAPPED_COLUMNS + ["tpi_500m", "local_relief_500m"],
}


def run_for_cohort(cohort, feature_columns, label):
    plot_table = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
    plot_table_with_features, available_columns = merge_environmental_features(plot_table, feature_columns=feature_columns)
    out_of_fold_predictions, _ = run_spatial_cv(plot_table_with_features, available_columns)

    rows = []
    for predicted_col, method_name in METHODS.items():
        summary = summarize_spatial_cv(out_of_fold_predictions, predicted_col)
        rows.append({
            "cohort": cohort, "feature_set": label, "n_features": len(available_columns), "method": method_name,
            "pooled_r2": summary["pooled_r2"], "per_fold_r2_mean": summary["per_fold_r2_mean"],
            "per_fold_r2_std": summary["per_fold_r2_std"],
        })
    return pd.DataFrame(rows)


def main():
    all_results = []
    for cohort in COHORTS:
        for label, columns in FEATURE_SET_VARIANTS.items():
            print(f"\n===== {cohort}: {label} =====")
            all_results.append(run_for_cohort(cohort, columns, label))

    results = pd.concat(all_results, ignore_index=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 100)
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()

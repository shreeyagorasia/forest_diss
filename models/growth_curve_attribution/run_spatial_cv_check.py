# Run as: python -m models.growth_curve_attribution.run_spatial_cv_check
#
# Rotates the held-out compartments across K folds instead of using one fixed split, so the
# reported R2 uses every compartment in the forest, not just the ~17% that happened to land in
# val under split_seed=42. See spatial_cv_check.py's own header for the full reasoning.

from models.growth_curve_attribution.scale_comparison_check import build_plot_level_table, merge_environmental_features
from models.growth_curve_attribution.spatial_cv_check import DEFAULT_K_FOLDS, run_spatial_cv, summarize_spatial_cv

COHORTS = ["4survey", "6survey"]
METHODS = {"elastic_net_predicted": "Elastic Net", "xgboost_predicted": "XGBoost"}


def run_for_cohort(cohort, k=DEFAULT_K_FOLDS):
    print(f"\n===== {cohort} ({k}-fold spatial CV) =====")
    plot_table = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
    plot_table_with_features, feature_columns = merge_environmental_features(plot_table)

    out_of_fold_predictions, fold_row_counts = run_spatial_cv(plot_table_with_features, feature_columns, k=k)
    print(f"  Rows per fold: {fold_row_counts.tolist()}")

    for predicted_col, method_name in METHODS.items():
        summary = summarize_spatial_cv(out_of_fold_predictions, predicted_col)
        print(f"\n  {method_name}:")
        print(f"    n_compartments (all, pooled across folds): {summary['n_compartments']}")
        print(f"    n_plots: {summary['n_plots']:,}")
        print(f"    pooled_r2 (headline, uses every compartment): {summary['pooled_r2']:.4f}")
        print(f"    per_fold_r2_mean: {summary['per_fold_r2_mean']:.4f}  per_fold_r2_std: {summary['per_fold_r2_std']:.4f}")
        print(f"    per_fold_r2_values: { {k: round(v, 4) for k, v in summary['per_fold_r2_values'].items()} }")


def main():
    for cohort in COHORTS:
        run_for_cohort(cohort)


if __name__ == "__main__":
    main()

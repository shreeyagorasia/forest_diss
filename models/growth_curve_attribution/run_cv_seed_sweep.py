# Run as: python -m models.growth_curve_attribution.run_cv_seed_sweep
#
# The user's own next check: how much does the 5-fold spatial CV pooled R2 itself vary as a
# function of which seed governs the fold assignment? The bootstrap CI (bootstrap_ci_check.py)
# quantifies uncertainty for ONE fixed CV partition; this instead re-runs the whole CV design
# under several different fold-assignment seeds, giving a genuine error bound on the CURRENT
# headline metric (5-fold pooled R2), not the older single-split metric the earlier seed sweep
# (run_seed_sweep_check.py) covers.

import pandas as pd

from models.growth_curve_attribution.explain_signal import FINAL_FEATURE_COLUMNS
from models.growth_curve_attribution.scale_comparison_check import build_plot_level_table, merge_environmental_features
from models.growth_curve_attribution.spatial_cv_check import run_spatial_cv, summarize_spatial_cv

COHORTS = ["4survey", "6survey"]
SEEDS_TO_CHECK = [42, 43, 44, 45, 46, 47, 48, 49]
METHODS = {"elastic_net_predicted": "Elastic Net", "xgboost_predicted": "XGBoost"}


def run_for_cohort_and_seed(cohort, table, feature_columns, seed):
    table_with_features, available_columns = merge_environmental_features(table, feature_columns=feature_columns)
    out_of_fold_predictions, _ = run_spatial_cv(table_with_features, available_columns, seed=seed)

    rows = []
    for predicted_col, method_name in METHODS.items():
        summary = summarize_spatial_cv(out_of_fold_predictions, predicted_col)
        rows.append({
            "cohort": cohort, "cv_fold_seed": seed, "method": method_name,
            "pooled_r2": summary["pooled_r2"],
            "per_fold_r2_mean": summary["per_fold_r2_mean"],
            "per_fold_r2_std": summary["per_fold_r2_std"],
        })
    return pd.DataFrame(rows)


def main():
    all_results = []
    for cohort in COHORTS:
        table = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
        for seed in SEEDS_TO_CHECK:
            print(f"\n===== {cohort}, CV fold-assignment seed={seed} =====")
            all_results.append(run_for_cohort_and_seed(cohort, table, FINAL_FEATURE_COLUMNS, seed))

    results = pd.concat(all_results, ignore_index=True)
    pd.set_option("display.width", 160)
    print()
    print(results.to_string(index=False))

    print("\n===== Summary across CV fold-assignment seeds =====")
    summary = results.groupby(["cohort", "method"])["pooled_r2"].agg(["mean", "std", "min", "max"])
    print(summary.round(4))

    output_path = "outputs/growth_curve_attribution/cv_seed_sweep.csv"
    import os
    os.makedirs("outputs/growth_curve_attribution", exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()

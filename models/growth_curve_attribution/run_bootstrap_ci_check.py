# Run as: python -m models.growth_curve_attribution.run_bootstrap_ci_check
#
# Answers the LLM Council's "one thing to do first": what's the real uncertainty on the
# plot-level attribution R2, and how many independent compartments is it actually averaging
# over? See bootstrap_ci_check.py's own header for the full reasoning.

from models.growth_curve_attribution.bootstrap_ci_check import cluster_bootstrap_r2_ci, get_val_predictions_with_cpmt

COHORTS = ["4survey", "6survey"]
METHODS = {"elastic_net_predicted": "Elastic Net", "xgboost_predicted": "XGBoost"}


def run_for_cohort(cohort):
    print(f"\n===== {cohort} =====")
    val_predictions = get_val_predictions_with_cpmt(cohort)

    for predicted_col, method_name in METHODS.items():
        result = cluster_bootstrap_r2_ci(val_predictions, predicted_col)
        print(f"\n  {method_name}:")
        print(f"    n_val_compartments: {result['n_val_compartments']}")
        print(f"    n_val_rows: {result['n_val_rows']:,}")
        print(f"    point_estimate_r2: {result['point_estimate_r2']:.4f}")
        print(f"    bootstrap_mean_r2: {result['bootstrap_mean_r2']:.4f}")
        print(f"    bootstrap_std_r2: {result['bootstrap_std_r2']:.4f}")
        print(f"    95% CI: [{result['ci_95_lower']:.4f}, {result['ci_95_upper']:.4f}]")
        print(f"    fraction of bootstrap resamples with R2 < 0: {result['fraction_of_bootstrap_resamples_below_zero']:.1%}")


def main():
    for cohort in COHORTS:
        run_for_cohort(cohort)


if __name__ == "__main__":
    main()

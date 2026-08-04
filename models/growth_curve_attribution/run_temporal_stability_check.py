# Run as: python -m models.growth_curve_attribution.run_temporal_stability_check
#
# Prints the temporal stability check for both cohorts -- see
# models/growth_curve_attribution/temporal_stability_check.py's own header for what this checks
# and why it needs to happen before any spatial attribution work.

from models.growth_curve_attribution.temporal_stability_check import evaluate_temporal_stability

COHORTS = ["4survey", "6survey"]


def print_metrics(label, metrics):
    print(
        f"    {label}: R2={metrics['r2']:.4f}  RMSE={metrics['rmse']:.3f}m  "
        f"MAE={metrics['mae']:.3f}m  Bias={metrics['bias']:.3f}m"
    )


def main():
    for cohort in COHORTS:
        print(f"\n===== {cohort} =====")
        results = evaluate_temporal_stability(cohort)

        for evaluation_year, year_results in results.items():
            print(f"\n  -- Held-out year {evaluation_year} ({year_results['n_plots']:,} plots) --")
            print_metrics("Early-years-fit y_max curve", year_results["early_fit"])
            print_metrics("Static yldc curve          ", year_results["yldc_curve"])


if __name__ == "__main__":
    main()

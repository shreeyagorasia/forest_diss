# Purpose: the llm-council's third diagnostic for pinn_env_terrain_k's -0.71 y_max/k correlation
# (see documentation/experiment_log.md's 2026-08-03/04 entries). Every peer reviewer in that
# council independently flagged the same gap: nobody checked whether -0.71 is even statistically
# distinguishable from noise, given the likely small number of held-out compartments. This
# answers that directly with a 95% confidence interval, reusing the exact cluster-bootstrap
# pattern already proven in models/growth_curve_attribution/bootstrap_ci_check.py.
#
# Uses a CLUSTER bootstrap, not a plain row-level bootstrap. Same reasoning as that file's own
# header: plots in the same compartment are NOT independent (they share the same held-out/train
# fate under spatial_block_split, and often similar terrain), so resampling individual PLOT rows
# would pretend there are far more independent data points than there really are. Resampling
# whole COMPARTMENTS is the statistically honest unit here.
#
# Run as: python -m models.pinn_env_terrain_k.bootstrap_correlation_check --cohort 4survey --split-type spatial_block

import argparse
import json

import numpy as np
import pandas as pd

from models.common.saving import model_output_dir


def load_one_row_per_plot(predictions_path):
    # predictions.csv has one row per plot-YEAR (the usual long format every model in this repo
    # saves). But learned_y_max/learned_k are static PER PLOT (same value repeated across a
    # plot's own survey years, since terrain/wind inputs don't vary by year). Bootstrapping the
    # long table directly would silently let a 4-survey plot count 4x as much as it should in
    # every resample. Deduplicating to one row per plot first is what makes each resampled
    # unit a genuine, equally-weighted plot.
    predictions_df = pd.read_csv(predictions_path)
    one_row_per_plot = predictions_df.drop_duplicates(subset="identification")[
        ["identification", "cpmt", "learned_y_max", "learned_k"]
    ]
    return one_row_per_plot


def compute_correlation(learned_y_max, learned_k):
    return float(np.corrcoef(learned_y_max, learned_k)[0, 1])


def cluster_bootstrap_correlation_ci(plot_table, n_bootstrap=2000, seed=0):
    # Resamples whole COMPARTMENTS with replacement (not individual plots). See this module's
    # own header for why that's the statistically honest unit here.
    learned_y_max = plot_table["learned_y_max"].to_numpy()
    learned_k = plot_table["learned_k"].to_numpy()
    cpmt_values = plot_table["cpmt"].to_numpy()

    compartments = np.unique(cpmt_values)
    n_compartments = len(compartments)
    indices_by_compartment = {compartment: np.where(cpmt_values == compartment)[0] for compartment in compartments}

    rng = np.random.default_rng(seed)
    bootstrap_correlations = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resampled_compartments = rng.choice(compartments, size=n_compartments, replace=True)
        resampled_indices = np.concatenate([indices_by_compartment[compartment] for compartment in resampled_compartments])
        bootstrap_correlations[i] = compute_correlation(learned_y_max[resampled_indices], learned_k[resampled_indices])

    point_estimate = compute_correlation(learned_y_max, learned_k)

    return {
        "n_plots": len(plot_table),
        "n_compartments": n_compartments,
        "point_estimate_correlation": point_estimate,
        "bootstrap_mean_correlation": float(np.nanmean(bootstrap_correlations)),
        "bootstrap_std_correlation": float(np.nanstd(bootstrap_correlations)),
        "ci_95_lower": float(np.nanpercentile(bootstrap_correlations, 2.5)),
        "ci_95_upper": float(np.nanpercentile(bootstrap_correlations, 97.5)),
        "fraction_of_bootstrap_resamples_with_wrong_sign": float(np.nanmean(bootstrap_correlations > 0)),
    }


def check_for_cohort(cohort, split_type, run_name, n_bootstrap):
    output_model_name = run_name if run_name else "pinn_env_terrain_k"
    predictions_path = model_output_dir(output_model_name, cohort, split_type=split_type) / "predictions.csv"
    if not predictions_path.exists():
        print(f"  (No predictions.csv found at {predictions_path}. Run evaluate_pinn_env_terrain_k.py first.)")
        return None

    plot_table = load_one_row_per_plot(predictions_path)
    result = cluster_bootstrap_correlation_ci(plot_table, n_bootstrap=n_bootstrap)

    print(f"  n_plots={result['n_plots']:,}  n_compartments={result['n_compartments']}")
    print(f"  Point estimate correlation: {result['point_estimate_correlation']:+.4f}")
    print(
        f"  Cluster-bootstrap ({n_bootstrap} resamples): mean={result['bootstrap_mean_correlation']:+.4f}  "
        f"std={result['bootstrap_std_correlation']:.4f}"
    )
    print(f"  95% CI: [{result['ci_95_lower']:+.4f}, {result['ci_95_upper']:+.4f}]")
    print(
        f"  Fraction of resamples with the WRONG (positive) sign: "
        f"{result['fraction_of_bootstrap_resamples_with_wrong_sign']:.1%}"
    )
    if result["ci_95_upper"] < 0:
        print("  READ: 95% CI excludes zero and stays negative. The correlation's sign is not just noise.")
    else:
        print("  READ: 95% CI includes zero or crosses into positive territory. The correlation's sign/magnitude is not well-pinned-down by this test set alone.")

    output_path = model_output_dir(output_model_name, cohort, split_type=split_type) / "y_max_k_bootstrap_ci.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default=None, help="Omit to run both cohorts.")
    parser.add_argument(
        "--split-type",
        choices=["temporal", "spatial_block", "spatial_block_kfold", "temporal_narrow_gap"],
        default="spatial_block",
    )
    parser.add_argument("--run-name", default=None, help="Must match the --run-name used when fitting, if one was used.")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]
    for cohort in cohorts:
        print(f"===== {cohort} ({args.split_type}) =====")
        check_for_cohort(cohort, args.split_type, args.run_name, args.n_bootstrap)
        print()


if __name__ == "__main__":
    main()

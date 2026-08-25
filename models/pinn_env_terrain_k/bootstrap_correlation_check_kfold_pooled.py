# Purpose: rerun the y_max/k cluster-bootstrap CI (bootstrap_correlation_check.py) on the CURRENT
# tier's pooled 5-fold w=0 predictions. The flagship -0.935 correlation cited in RQ2a item 3 is
# a pooled 5-fold spatial_block_kfold number, but the existing bootstrap CI (95% CI [-0.77, -0.62])
# was computed on the OLD, pre-restructuring tier's single-split predictions, flagged as "not yet
# rerun current" in the draft. bootstrap_correlation_check.py's own file-reading logic expects ONE
# predictions.csv at a single path, but a kfold run saves 5 separate predictions.csv files (one per
# fold_N subfolder). This script pools those 5 files first (matching the exact pooling approach
# already used to recompute the -0.9345 point estimate, see
# TEMP_rq1_physicsablation_results_2026-08-11.tex's "What does the -0.935 correlation actually look
# like" section), then reuses cluster_bootstrap_correlation_ci() directly, unmodified.
#
# Run as: python -m models.pinn_env_terrain_k.bootstrap_correlation_check_kfold_pooled --cohort 4survey

import argparse
import json
from pathlib import Path

import pandas as pd

from models.pinn_env_terrain_k.bootstrap_correlation_check import cluster_bootstrap_correlation_ci

RUN_NAME = "rq1_pinn_env_terrain_k_nested_set3_gated_terrain_wind_vif_w0_seed42"
N_FOLDS = 5


def load_pooled_one_row_per_plot(cohort):
    frames = []
    for fold in range(N_FOLDS):
        path = Path("outputs/spatial_block_kfold") / RUN_NAME / cohort / f"fold_{fold}" / "predictions.csv"
        frames.append(pd.read_csv(path))
    pooled = pd.concat(frames, ignore_index=True)
    # learned_y_max/learned_k are static per plot (same value repeated across a plot's own survey
    # years and, since each plot appears in exactly one fold's test set under spatial_block_kfold,
    # repeated only within that one fold). Dedupe to one row per plot so each plot counts once.
    one_row_per_plot = pooled.drop_duplicates(subset="identification")[["identification", "cpmt", "learned_y_max", "learned_k"]]
    return one_row_per_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default=None)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    args = parser.parse_args()

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]
    for cohort in cohorts:
        print(f"===== {cohort} (pooled 5-fold, w=0, current tier) =====")
        plot_table = load_pooled_one_row_per_plot(cohort)
        result = cluster_bootstrap_correlation_ci(plot_table, n_bootstrap=args.n_bootstrap)
        print(f"  n_plots={result['n_plots']:,}  n_compartments={result['n_compartments']}")
        print(f"  Point estimate correlation: {result['point_estimate_correlation']:+.4f}")
        print(f"  95% CI: [{result['ci_95_lower']:+.4f}, {result['ci_95_upper']:+.4f}]")
        print(f"  Fraction of resamples with the WRONG (positive) sign: {result['fraction_of_bootstrap_resamples_with_wrong_sign']:.1%}")

        output_path = Path("outputs/spatial_block_kfold") / RUN_NAME / cohort / "y_max_k_bootstrap_ci_pooled.json"
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved -> {output_path}")
        print()


if __name__ == "__main__":
    main()

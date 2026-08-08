# Purpose: is GNNWR's R2 advantage over Elastic Net/XGBoost (found in the full-population 5-fold
# CV, see documentation/experiment_log.md's 2026-08-06 entry) actually statistically supported, or
# just two point estimates that happen to differ? Two complementary checks, on purpose -- they
# have very different statistical power and it is worth seeing both:
#
# 1. A PAIRED comparison across the 5 folds themselves (Wilcoxon signed-rank + paired t-test on
#    the 5 fold-level R2 values). Cheap, and directly answers "does GNNWR win in most folds, not
#    just on average" -- but n=5 has very little power (Wilcoxon's smallest possible p-value at
#    n=5 is 0.0625, so this test can basically never reach conventional significance even with a
#    real, large effect). Reported anyway for the fold-by-fold pattern, not for its p-value.
#
# 2. A CLUSTER BOOTSTRAP on the R2 DIFFERENCE, resampling whole COMPARTMENTS (not plot rows) --
#    the same statistically-honest unit this project's own bootstrap_ci_check.py already
#    established for a single model's CI, extended here to bootstrap the PAIRED DIFFERENCE
#    between two models' R2 on the SAME resampled compartments each iteration (not two separate
#    CIs eyeballed for overlap, which is a weaker, less correct comparison). This has real power
#    (231 compartments to resample from, not 5 folds), and is the more trustworthy of the two.
#
# Both checks reuse already-saved data -- EN/XGBoost via a fresh (but cheap, local) run of
# run_spatial_cv, GNNWR via the 5 already-saved fold-level test-prediction CSVs -- no cluster
# resubmission needed.

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from models.common.metrics import compute_metrics
from models.growth_curve_attribution.broad_environmental_check import columns_for_groups
from models.growth_curve_attribution.gnnwr_check import OUTPUT_DIR as GNNWR_OUTPUT_DIR
from models.growth_curve_attribution.gnnwr_check import SCOPES
from models.growth_curve_attribution.scale_comparison_check import TARGET, build_plot_level_table, merge_environmental_features
from models.growth_curve_attribution.spatial_cv_check import run_spatial_cv, summarize_spatial_cv
from models.elasticnet_environmental.elasticnet_environmental import drop_rows_with_missing_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def compute_r2(actual, predicted):
    # Same formula as models/common/metrics.py -- kept local so this file has no other
    # dependency for the one calculation the bootstrap loop calls thousands of times.
    sum_squared_error = np.sum((actual - predicted) ** 2)
    sum_squared_total = np.sum((actual - actual.mean()) ** 2)
    return 1 - sum_squared_error / sum_squared_total if sum_squared_total > 0 else np.nan


def load_en_xgb_out_of_fold(cohort: str, scope: str):
    """Fresh, local, cheap rerun of the SAME 5-fold spatial CV EN/XGBoost's headline numbers
    come from -- returns per-plot out-of-fold predictions (identification, cpmt, fold, target,
    elastic_net_predicted, xgboost_predicted) for every plot, plus the per-fold R2 summary."""
    feature_columns = columns_for_groups(SCOPES[scope])
    plot_table = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
    merged, feature_columns = merge_environmental_features(plot_table, feature_columns=feature_columns)
    merged = drop_rows_with_missing_features(merged, feature_columns)
    out_of_fold, _ = run_spatial_cv(merged, feature_columns)
    return out_of_fold


def load_gnnwr_pooled(cohort: str, scope: str):
    """Loads and concatenates the 5 already-saved full-population fold CSVs -- every plot
    appears exactly once, in the fold where it was held out, so this is directly comparable to
    EN/XGBoost's own out-of-fold predictions."""
    paths = sorted(glob.glob(str(GNNWR_OUTPUT_DIR / f"gnnwr_{scope}_{cohort}_reffull_fold*_test_predictions.csv")))
    if not paths:
        raise FileNotFoundError(f"No full-population GNNWR fold CSVs found for {cohort}/{scope} in {GNNWR_OUTPUT_DIR}")
    fold_tables = []
    for fold_index, path in enumerate(paths):
        table = pd.read_csv(path)
        table["fold"] = fold_index
        fold_tables.append(table)
    return pd.concat(fold_tables, ignore_index=True)


def paired_fold_test(en_xgb_oof: pd.DataFrame, gnnwr_pooled: pd.DataFrame):
    """Check 1: paired comparison of the 5 fold-level R2 values."""
    en_fold_r2, xgb_fold_r2, gnnwr_fold_r2 = [], [], []
    for fold in sorted(en_xgb_oof["fold"].unique()):
        fold_rows_en_xgb = en_xgb_oof[en_xgb_oof["fold"] == fold]
        en_fold_r2.append(compute_r2(fold_rows_en_xgb[TARGET].to_numpy(), fold_rows_en_xgb["elastic_net_predicted"].to_numpy()))
        xgb_fold_r2.append(compute_r2(fold_rows_en_xgb[TARGET].to_numpy(), fold_rows_en_xgb["xgboost_predicted"].to_numpy()))
        fold_rows_gnnwr = gnnwr_pooled[gnnwr_pooled["fold"] == fold]
        gnnwr_fold_r2.append(compute_r2(fold_rows_gnnwr[TARGET].to_numpy(), fold_rows_gnnwr["denormalized_pred_result"].to_numpy()))

    en_fold_r2, xgb_fold_r2, gnnwr_fold_r2 = np.array(en_fold_r2), np.array(xgb_fold_r2), np.array(gnnwr_fold_r2)

    results = {}
    for label, other_r2 in [("GNNWR vs Elastic Net", en_fold_r2), ("GNNWR vs XGBoost", xgb_fold_r2)]:
        diffs = gnnwr_fold_r2 - other_r2
        # Wilcoxon needs at least one non-zero difference and n>=1; with n=5 it can only ever
        # report a handful of distinct p-values (its minimum possible p-value at n=5 is 0.0625).
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diffs) if np.any(diffs != 0) else (np.nan, np.nan)
        ttest_stat, ttest_p = stats.ttest_rel(gnnwr_fold_r2, other_r2)
        results[label] = {
            "fold_r2_gnnwr": gnnwr_fold_r2.tolist(),
            "fold_r2_other": other_r2.tolist(),
            "fold_diffs": diffs.tolist(),
            "n_folds_gnnwr_wins": int(np.sum(diffs > 0)),
            "mean_diff": float(diffs.mean()),
            "wilcoxon_stat": float(wilcoxon_stat) if not np.isnan(wilcoxon_stat) else None,
            "wilcoxon_p": float(wilcoxon_p) if not np.isnan(wilcoxon_p) else None,
            "paired_ttest_stat": float(ttest_stat),
            "paired_ttest_p": float(ttest_p),
        }
    return results


def cluster_bootstrap_paired_difference(combined: pd.DataFrame, col_a: str, col_b: str, n_bootstrap: int = 2000, seed: int = 0):
    """Check 2: bootstrap the R2 DIFFERENCE (col_a's R2 minus col_b's R2), resampling whole
    compartments with replacement -- same compartment draw used for BOTH models each iteration,
    so this is a genuine paired bootstrap on the difference, not two separate CIs."""
    target_values = combined[TARGET].to_numpy()
    pred_a = combined[col_a].to_numpy()
    pred_b = combined[col_b].to_numpy()
    cpmt_values = combined["cpmt"].to_numpy()

    compartments = np.unique(cpmt_values)
    n_compartments = len(compartments)
    indices_by_compartment = {compartment: np.where(cpmt_values == compartment)[0] for compartment in compartments}

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)
    r2_a_samples = np.empty(n_bootstrap)
    r2_b_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resampled_compartments = rng.choice(compartments, size=n_compartments, replace=True)
        resampled_indices = np.concatenate([indices_by_compartment[compartment] for compartment in resampled_compartments])
        r2_a = compute_r2(target_values[resampled_indices], pred_a[resampled_indices])
        r2_b = compute_r2(target_values[resampled_indices], pred_b[resampled_indices])
        r2_a_samples[i] = r2_a
        r2_b_samples[i] = r2_b
        diffs[i] = r2_a - r2_b

    point_r2_a = compute_r2(target_values, pred_a)
    point_r2_b = compute_r2(target_values, pred_b)

    return {
        "n_compartments": n_compartments,
        "n_plots": len(combined),
        "point_r2_a": point_r2_a,
        "point_r2_b": point_r2_b,
        "point_diff": point_r2_a - point_r2_b,
        "bootstrap_mean_diff": float(np.nanmean(diffs)),
        "diff_ci_95_lower": float(np.nanpercentile(diffs, 2.5)),
        "diff_ci_95_upper": float(np.nanpercentile(diffs, 97.5)),
        "fraction_bootstrap_a_beats_b": float(np.nanmean(diffs > 0)),
    }


def run_check(cohort: str, scope: str, n_bootstrap: int = 2000, seed: int = 0):
    print(f"\n=== {cohort} / {scope} ===")
    en_xgb_oof = load_en_xgb_out_of_fold(cohort, scope)
    gnnwr_pooled = load_gnnwr_pooled(cohort, scope)

    print(f"  EN/XGBoost out-of-fold plots: {len(en_xgb_oof):,}  GNNWR pooled plots: {len(gnnwr_pooled):,}")

    print("\n  --- Check 1: paired fold-level test (n=5, low power -- read the fold pattern, not the p-value) ---")
    fold_results = paired_fold_test(en_xgb_oof, gnnwr_pooled)
    for label, result in fold_results.items():
        print(f"  {label}: GNNWR wins {result['n_folds_gnnwr_wins']}/5 folds, mean diff={result['mean_diff']:.4f}, "
              f"Wilcoxon p={result['wilcoxon_p']}, paired t-test p={result['paired_ttest_p']:.4f}")

    print("\n  --- Check 2: cluster bootstrap on the R2 difference (231 compartments, real power) ---")
    combined = en_xgb_oof.merge(
        gnnwr_pooled[["identification", "denormalized_pred_result"]].rename(columns={"denormalized_pred_result": "gnnwr_predicted"}),
        on="identification", how="inner",
    )
    print(f"  Matched plots for bootstrap (EN/XGBoost/GNNWR all present): {len(combined):,}")

    bootstrap_results = {}
    for label, col_a, col_b in [
        ("GNNWR vs Elastic Net", "gnnwr_predicted", "elastic_net_predicted"),
        ("GNNWR vs XGBoost", "gnnwr_predicted", "xgboost_predicted"),
    ]:
        result = cluster_bootstrap_paired_difference(combined, col_a, col_b, n_bootstrap=n_bootstrap, seed=seed)
        bootstrap_results[label] = result
        print(f"  {label}: R2 {result['point_r2_a']:.4f} vs {result['point_r2_b']:.4f}, "
              f"diff={result['point_diff']:.4f}, 95% CI on diff=[{result['diff_ci_95_lower']:.4f}, {result['diff_ci_95_upper']:.4f}], "
              f"P(GNNWR better)={result['fraction_bootstrap_a_beats_b']:.3f}")

    return fold_results, bootstrap_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default="4survey")
    parser.add_argument("--scope", choices=list(SCOPES) + ["both"], default="both")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scopes = list(SCOPES) if args.scope == "both" else [args.scope]
    for scope in scopes:
        run_check(args.cohort, scope, n_bootstrap=args.n_bootstrap, seed=args.seed)


if __name__ == "__main__":
    main()

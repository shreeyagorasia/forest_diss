# Purpose: pool a DNN/PINN model's k-fold spatial CV results into one whole-population score,
# plus the per-fold spread. The same "precision fix" idea already proven for the Elastic Net/
# XGBoost attribution work in models/growth_curve_attribution/spatial_cv_check.py
# (summarize_spatial_cv), generalised here to read the DNN/PINN family's own predictions.csv
# format (which run_*.py/evaluate_*.py already save per fold, one file per
# outputs/spatial_block_kfold/<model_name>/<cohort>/fold_<i>/predictions.csv).
#
# Run as: python -m models.common.kfold_summary --model-name pinn_env_terrain_k --cohort 4survey --n-folds 5

import argparse
import json

import pandas as pd

from models.common.metrics import compute_metrics
from models.common.saving import model_output_dir
from models.common.bootstrap_ci import cluster_bootstrap_ci, r2_statistic


def load_all_folds(model_name, cohort, n_folds, split_type="spatial_block_kfold"):
    # Every fold's evaluate_*.py already wrote its own test-set predictions.csv. This just
    # reads all of them and stacks them into one table. Every plot appears in exactly one fold's
    # test set (spatial_kfold_split's own guarantee, see models/common/splits.py), so stacking
    # them gives one row per plot-year across the WHOLE population, not one arbitrary ~20% slice.
    all_fold_predictions = []
    for fold_index in range(n_folds):
        predictions_path = model_output_dir(model_name, cohort, f"fold_{fold_index}", split_type=split_type) / "predictions.csv"
        if not predictions_path.exists():
            raise FileNotFoundError(
                f"Missing {predictions_path}. Run both run_{model_name}.py and "
                f"evaluate_{model_name}.py for fold {fold_index} first."
            )
        fold_predictions = pd.read_csv(predictions_path)
        fold_predictions["fold"] = fold_index
        all_fold_predictions.append(fold_predictions)

    return pd.concat(all_fold_predictions, ignore_index=True)


def summarize_kfold(model_name, cohort, n_folds, split_type="spatial_block_kfold"):
    pooled_predictions = load_all_folds(model_name, cohort, n_folds, split_type=split_type)

    # Pooled metrics: every plot's own out-of-fold prediction, all folds concatenated, ONE score
    # computed over the whole population. The headline number, since it uses every compartment
    # rather than one arbitrary ~20% test slice.
    pooled_metrics = compute_metrics(
        pooled_predictions["observed_top_height"].values,
        pooled_predictions["predicted_top_height"].values,
        age=pooled_predictions["Age"].values,
    )

    # Per-fold metrics: how much the estimate would have varied if only ONE fold's worth of
    # compartments had been used as the held-out set (i.e. what a single spatial_block_split
    # call is actually doing). Shown alongside the pooled number so the precision gain from
    # k-fold is visible directly, not just asserted.
    per_fold_r2 = []
    for fold_index in range(n_folds):
        fold_rows = pooled_predictions[pooled_predictions["fold"] == fold_index]
        fold_metrics = compute_metrics(fold_rows["observed_top_height"].values, fold_rows["predicted_top_height"].values)
        per_fold_r2.append(fold_metrics["r2"])

    per_fold_r2_series = pd.Series(per_fold_r2)

    # A single pooled R2 number invites over-reading ("0.42 beats 0.38") without knowing whether
    # that gap is even distinguishable from resampling noise. Cluster-bootstrap the pooled R2
    # itself (resampling whole compartments. See models/common/bootstrap_ci.py for why), so
    # every model's headline number comes with a 95% CI, not just a point estimate.
    r2_ci = cluster_bootstrap_ci(pooled_predictions, "cpmt", r2_statistic, n_resamples=1000, seed=42)

    summary = {
        "model_name": model_name,
        "cohort": cohort,
        "n_folds": n_folds,
        "n_plot_years": len(pooled_predictions),
        "n_plots": pooled_predictions["identification"].nunique(),
        "pooled_r2": pooled_metrics["r2"],
        "pooled_r2_ci_low": r2_ci["ci_low"],
        "pooled_r2_ci_high": r2_ci["ci_high"],
        "pooled_r2_ci_n_clusters": r2_ci["n_clusters"],
        "pooled_rmse": pooled_metrics["rmse"],
        "pooled_mae": pooled_metrics["mae"],
        "pooled_bias": pooled_metrics["bias"],
        "per_fold_r2_mean": float(per_fold_r2_series.mean()),
        "per_fold_r2_std": float(per_fold_r2_series.std()),
        "per_fold_r2_values": per_fold_r2,
    }

    # If this model also learns a per-plot y_max/k map (pinn_env_terrain/pinn_env_terrain_k),
    # pool that too. Gives a much larger, less noisy sample of the y_max/k correlation than
    # any single fold's ~20% slice could.
    if "learned_y_max" in pooled_predictions.columns and "learned_k" in pooled_predictions.columns:
        one_row_per_plot = pooled_predictions.drop_duplicates(subset="identification")
        summary["pooled_y_max_k_correlation"] = float(
            one_row_per_plot["learned_y_max"].corr(one_row_per_plot["learned_k"])
        )

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="e.g. pinn_env_terrain_k, dnn_noenv (matches --run-name if one was used when fitting).")
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default=None, help="Omit to summarize both cohorts.")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--split-type", choices=["spatial_block_kfold"], default="spatial_block_kfold",
        help="Only spatial_block_kfold has per-fold outputs to pool.",
    )
    args = parser.parse_args()

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]
    for cohort in cohorts:
        print(f"===== {args.model_name} ({cohort}, {args.n_folds}-fold) =====")
        summary = summarize_kfold(args.model_name, cohort, args.n_folds, split_type=args.split_type)
        print(f"  n_plots={summary['n_plots']:,}  n_plot_years={summary['n_plot_years']:,}")
        print(f"  Pooled (whole population): R2={summary['pooled_r2']:.4f}  RMSE={summary['pooled_rmse']:.4f}  MAE={summary['pooled_mae']:.4f}  Bias={summary['pooled_bias']:+.4f}")
        print(f"  Pooled R2 95% CI (cluster bootstrap, {summary['pooled_r2_ci_n_clusters']} compartments): [{summary['pooled_r2_ci_low']:.4f}, {summary['pooled_r2_ci_high']:.4f}]")
        print(f"  Per-fold R2: mean={summary['per_fold_r2_mean']:.4f}  std={summary['per_fold_r2_std']:.4f}  values={[round(v, 4) for v in summary['per_fold_r2_values']]}")
        if "pooled_y_max_k_correlation" in summary:
            print(f"  Pooled y_max/k correlation (all {summary['n_plots']:,} plots, one row per plot): {summary['pooled_y_max_k_correlation']:+.4f}")

        output_path = model_output_dir(args.model_name, cohort, split_type=args.split_type) / "kfold_summary.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved -> {output_path}")
        print()


if __name__ == "__main__":
    main()

# Run as: python -m models.elasticnet_environmental.run_elasticnet_environmental --cohort 4survey
#     or: python -m models.elasticnet_environmental.run_elasticnet_environmental --cohort 6survey
#     or: python -m models.elasticnet_environmental.run_elasticnet_environmental (both cohorts)
#
# Fits Elastic Net (regularized linear regression) to predict each plot's mean Chapman-Richards
# residual from its environmental variables -- the SAME feature sets, SAME data, SAME
# spatial_block_split() as models/xgb_environmental/run_xgb_environmental.py, so the two
# methods' results are directly comparable, not run under different conditions.
#
# Why build this alongside XGBoost+SHAP at all: SHAP values are known to be unreliable when
# input features are correlated (confirmed for this exact dataset -- see
# notebooks/environmental_data/env_variable_importance.ipynb section 6, where elevation's SHAP
# rank turned out to be misleading once tested by ablation). Elastic Net's penalty behaves
# differently under correlation -- it tends to SPREAD credit across a correlated group rather
# than letting one member's SHAP value arbitrarily win -- so comparing the two methods' answers
# for the same correlated clusters is a genuine, independent cross-check, not just a second
# opinion built the same way.

import argparse
import json

from models.common.metrics import compute_metrics
from models.common.run_logging import RunTimer, format_error, write_run_log, write_started_marker
from models.common.saving import model_output_dir as output_dir
from models.common.splits import SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, spatial_block_split
from models.elasticnet_environmental.elasticnet_environmental import (
    FEATURE_SETS,
    drop_rows_with_missing_features,
    fit,
    get_coefficients_table,
    predict,
)
from models.xgb_environmental.data import load_plots_for_cohort

COHORTS = ["4survey", "6survey"]
SEED = 42
DEVICE = "cpu"


def run_one_feature_set(cohort, feature_set_name, plots_df):
    print(f"\n--- {cohort} / {feature_set_name} ---")

    # Same split function, same block column, same buffer, same seed as
    # run_xgb_environmental.py -- so a comparison between the two methods' test R2/coefficients
    # is never confounded by the two methods secretly seeing different train/val/test rows.
    split_labels = spatial_block_split(
        plots_df, block_col=SPATIAL_BLOCK_COL, buffer_distance=SPATIAL_BUFFER_METRES, seed=SEED,
    )
    plots_df = plots_df.copy()
    plots_df["split"] = split_labels

    train_df = plots_df[plots_df["split"] == "train"]
    val_df = plots_df[plots_df["split"] == "val"]
    test_df = plots_df[plots_df["split"] == "test"]
    print(f"  train rows: {len(train_df):,} | val rows: {len(val_df):,} | test rows: {len(test_df):,}")

    # Unlike XGBoost, Elastic Net cannot fit or predict on a row with any NaN feature value --
    # see drop_rows_with_missing_features()'s own comment for why this is safe to do here.
    feature_columns = FEATURE_SETS[feature_set_name]
    train_df = drop_rows_with_missing_features(train_df, feature_columns)
    val_df = drop_rows_with_missing_features(val_df, feature_columns)
    test_df = drop_rows_with_missing_features(test_df, feature_columns)

    hyperparameters = {"seed": SEED, "feature_set": feature_set_name, "n_features": len(FEATURE_SETS[feature_set_name])}
    timer = RunTimer().start()
    attempt_id = write_started_marker(
        model_name=f"elasticnet_environmental_{feature_set_name}", cohort=cohort, split_type="spatial_block",
        run_phase="fit_and_evaluate", is_test_run=False, device=DEVICE, hyperparameters=hyperparameters,
    )

    try:
        fitted = fit(train_df, feature_set_name, seed=SEED)
        print(f"  ElasticNetCV picked alpha={fitted['model'].alpha_:.4f}, l1_ratio={fitted['model'].l1_ratio_:.2f}")

        def scored_metrics(df):
            predictions = predict(df, fitted, feature_set_name)
            row_metrics = compute_metrics(df["mean_cr_residual"].values, predictions)
            del row_metrics["mre"], row_metrics["accuracy_pct"]  # nonsensical for a residual target, see xgb_environmental's own comment on this
            return row_metrics, predictions

        # val is for feature-selection/comparison decisions; test is reported once, as the final
        # number -- the same train/val/test discipline run_xgb_environmental.py now follows.
        val_metrics, _ = scored_metrics(val_df)
        test_metrics, test_predictions = scored_metrics(test_df)
        metrics = test_metrics
        print(f"  val R2:  {val_metrics['r2']:.3f} | val RMSE:  {val_metrics['rmse']:.3f}")
        print(f"  test R2: {test_metrics['r2']:.3f} | test RMSE: {test_metrics['rmse']:.3f}")

        save_dir = output_dir("elasticnet_environmental", feature_set_name, cohort, split_type="spatial_block")
        save_dir.mkdir(parents=True, exist_ok=True)

        predictions_out = test_df[["identification", "cpmt", "mean_cr_residual"]].copy()
        predictions_out["predicted"] = test_predictions
        predictions_out["residual"] = predictions_out["mean_cr_residual"] - predictions_out["predicted"]
        predictions_out.to_csv(save_dir / "predictions.csv", index=False)

        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(save_dir / "val_metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=2)

        # The whole point of this model: a directly-readable, signed, per-variable effect size --
        # no SHAP/TreeExplainer step needed, the coefficient IS the answer. Saved for every
        # feature set (unlike XGBoost's SHAP, which is only computed for two of the three).
        coefficients = get_coefficients_table(fitted)
        coefficients.to_csv(save_dir / "coefficients.csv", header=True)
        print(f"  Saved metrics + predictions + coefficients -> {save_dir}")

        write_run_log(
            attempt_id=attempt_id,
            model_name=f"elasticnet_environmental_{feature_set_name}", cohort=cohort, split_type="spatial_block",
            run_phase="fit_and_evaluate", status="success", is_test_run=False, device=DEVICE,
            hyperparameters=hyperparameters, metrics=metrics, error=None,
            output_dir=save_dir, runtime_seconds=timer.elapsed_seconds(), n_rows_fit=len(train_df),
        )
    except Exception as error:
        write_run_log(
            attempt_id=attempt_id,
            model_name=f"elasticnet_environmental_{feature_set_name}", cohort=cohort, split_type="spatial_block",
            run_phase="fit_and_evaluate", status="failed", is_test_run=False, device=DEVICE,
            hyperparameters=hyperparameters, metrics=None, error=format_error(error),
            output_dir=None, runtime_seconds=timer.elapsed_seconds(), n_rows_fit=None,
        )
        print(f"  WARNING: {cohort}/{feature_set_name} failed: {error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=COHORTS, default=None, help="Omit to run both cohorts.")
    args = parser.parse_args()

    cohorts_to_run = [args.cohort] if args.cohort else COHORTS

    for cohort in cohorts_to_run:
        print(f"\n=== Cohort: {cohort} ===")
        plots_df = load_plots_for_cohort(cohort)
        print(f"Loaded {len(plots_df):,} plots")

        for feature_set_name in FEATURE_SETS:
            run_one_feature_set(cohort, feature_set_name, plots_df)


if __name__ == "__main__":
    main()

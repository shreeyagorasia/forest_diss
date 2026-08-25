# Run as: python -m models.env_deviation.run_env_deviation --cohort 4survey --split-type spatial_block --base cr
#     or: python -m models.env_deviation.run_env_deviation --cohort 4survey --split-type spatial_block --base dnn_noenv
#
# Decoupled deviation model. See models/env_deviation/env_deviation.py's own top-of-file note
# and documentation/model_instructions/env_deviation_decoupled_instructions.md for the full
# reasoning. No physics loss, no joint training: fit the base model's prediction (or read the
# frozen CR curve), compute what's left over, then fit XGBoost to predict THAT from terrain/wind
# alone. Fits in seconds. Like xgb_environmental.py's own run script, there is no separate
# cluster-fit/local-evaluate split for this model.

import argparse
import json
from datetime import datetime, timezone

import pandas as pd

from models.common.metrics import compute_metrics
from models.common.run_logging import RunTimer, format_error, write_run_log, write_started_marker
from models.common.saving import get_git_commit, load_cr_params, model_output_dir
from models.common.torch_data import (
    DEFAULT_ENV_TERRAIN_FEATURE_SET,
    ENV_TERRAIN_FEATURE_SETS,
    TARGET_COLUMN,
    fill_missing_time_since_thinning,
    load_split_table_with_terrain,
    select_device,
)
from models.env_deviation.env_deviation import (
    build_cr_residual_target,
    build_dnn_noenv_residual_target,
    fit_residual_model,
    predict_residual,
    split_val_for_residual_early_stopping,
)

MODEL_NAME = "env_deviation"
SEED = 42


def run_for_cohort(cohort, split_type, base_model, feature_set_name, run_name=None):
    default_name = f"{MODEL_NAME}_{base_model}"
    if feature_set_name != DEFAULT_ENV_TERRAIN_FEATURE_SET:
        default_name = f"{default_name}_{feature_set_name}"
    output_model_name = run_name if run_name else default_name

    print(f"===== {cohort} ({output_model_name}, {split_type}) — base={base_model} =====")

    timer = RunTimer().start()
    attempt_id = write_started_marker(
        model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="fit",
        is_test_run=False, device="cpu", hyperparameters={"base_model": base_model, "feature_set_name": feature_set_name},
    )

    try:
        feature_columns = ENV_TERRAIN_FEATURE_SETS[feature_set_name]
        print(f"  Terrain/wind feature set: {feature_set_name} = {feature_columns}")

        split_df = load_split_table_with_terrain(cohort, split_type, feature_columns)
        split_df = fill_missing_time_since_thinning(split_df)

        if base_model == "cr":
            cr_params = load_cr_params(cohort, split_type)
            base_prediction, residual = build_cr_residual_target(split_df, cr_params, TARGET_COLUMN)
        elif base_model == "dnn_noenv":
            device = select_device()
            base_prediction, residual = build_dnn_noenv_residual_target(
                split_df, cohort, split_type, device, TARGET_COLUMN
            )
        else:
            raise ValueError(f"Unknown base_model: {base_model!r}")

        split_df = split_df.copy()
        split_df["base_prediction"] = base_prediction
        split_df["residual"] = residual

        train_df = split_df[split_df["split"] == "train"]
        val_df = split_df[split_df["split"] == "val"]
        test_df = split_df[split_df["split"] == "test"]

        if base_model == "cr":
            # CR has only 3 global parameters. Minimal overfitting risk, so its train-set
            # residuals are a fair, representative target. Standard train-fits/val-for-early-
            # stopping/test-read-once discipline, same as everything else in this repo.
            residual_fit_df, residual_early_stopping_df = train_df, val_df
        else:
            # dnn_noenv overfits (train_loss << val_loss, every training curve this session shows
            # this). Fitting the residual model to its TRAIN-set residuals would fit an
            # artificially small, optimistic error. Use VAL-set residuals instead (see
            # split_val_for_residual_early_stopping()'s own note). Test stays untouched either way.
            residual_fit_df, residual_early_stopping_df = split_val_for_residual_early_stopping(val_df)

        print(f"  residual model: fit on {len(residual_fit_df):,} rows, "
              f"early-stopping on {len(residual_early_stopping_df):,} rows, "
              f"test on {len(test_df):,} rows")

        residual_model = fit_residual_model(residual_fit_df, residual_early_stopping_df, feature_columns, seed=SEED)

        predicted_residual_test = predict_residual(residual_model, test_df, feature_columns)
        predicted_height_test = test_df["base_prediction"].values + predicted_residual_test
        observed_height_test = test_df[TARGET_COLUMN].values

        metrics = compute_metrics(observed_height_test, predicted_height_test, age=test_df["Age"].values)

        output_dir = model_output_dir(output_model_name, cohort, split_type=split_type)
        output_dir.mkdir(parents=True, exist_ok=True)

        predictions_df = pd.DataFrame({
            "identification": test_df["identification"].values,
            "blk": test_df["blk"].values,
            "cpmt": test_df["cpmt"].values,
            "LiDAR_year": test_df["LiDAR_year"].values,
            "Age": test_df["Age"].values,
            "base_prediction": test_df["base_prediction"].values,
            "predicted_residual": predicted_residual_test,
            "observed_top_height": observed_height_test,
            "predicted_top_height": predicted_height_test,
            "residual": observed_height_test - predicted_height_test,
            "split": "test",
        })
        predictions_df.to_csv(output_dir / "predictions.csv", index=False)

        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        run_metadata = {
            "cohort": cohort,
            "base_model": base_model,
            "feature_set_name": feature_set_name,
            "n_rows_residual_fit": len(residual_fit_df),
            "n_rows_residual_early_stopping": len(residual_early_stopping_df),
            "n_rows_test": len(test_df),
            "xgb_best_iteration": residual_model.best_iteration,
            "fit_date": datetime.now(timezone.utc).isoformat(),
            "git_commit": get_git_commit(),
        }
        with open(output_dir / "run_metadata.json", "w") as f:
            json.dump(run_metadata, f, indent=2)

        write_run_log(
            attempt_id=attempt_id,
            model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="fit",
            status="success", is_test_run=False, device="cpu",
            hyperparameters={"base_model": base_model, "feature_set_name": feature_set_name}, metrics=metrics, error=None,
            output_dir=output_dir, runtime_seconds=timer.elapsed_seconds(), n_rows_fit=len(residual_fit_df),
        )

        print(f"  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}  Bias={metrics['bias']:+.4f}")
        print(f"  Saved -> {output_dir / 'predictions.csv'}")
        print(f"  Saved -> {output_dir / 'metrics.json'}")
        print()
        return metrics

    except Exception as error:
        write_run_log(
            attempt_id=attempt_id,
            model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="fit",
            status="failed", is_test_run=False, device="cpu",
            hyperparameters={"base_model": base_model, "feature_set_name": feature_set_name}, metrics=None, error=format_error(error),
            output_dir=None, runtime_seconds=timer.elapsed_seconds(),
        )
        print(f"  WARNING: {output_model_name} failed for {cohort}: {error}")
        print()
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default=None, help="Omit to run both cohorts.")
    parser.add_argument(
        "--split-type", choices=["temporal", "spatial_block", "temporal_narrow_gap"], default="spatial_block"
    )
    parser.add_argument("--base", choices=["cr", "dnn_noenv"], required=True, help="Which base prediction to compute the residual against.")
    parser.add_argument(
        "--feature-set", choices=list(ENV_TERRAIN_FEATURE_SETS.keys()), default=DEFAULT_ENV_TERRAIN_FEATURE_SET,
    )
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]

    all_metrics = {}
    for cohort in cohorts:
        all_metrics[cohort] = run_for_cohort(cohort, args.split_type, args.base, args.feature_set, args.run_name)

    print("===== Summary: test-split metrics =====")
    for cohort, metrics in all_metrics.items():
        print(f"  {cohort}: MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}  Bias={metrics['bias']:+.4f}")


if __name__ == "__main__":
    main()

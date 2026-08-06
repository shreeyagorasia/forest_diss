# Run as: python -m models.pinn_noenv.evaluate_pinn_noenv --cohort 4survey
#
# EVALUATES an already-trained PINN on the held-out test split (2023). Does
# not train anything -- just loads the checkpoint + scalers that
# run_pinn_noenv.py already saved, makes predictions on the test rows, and
# computes accuracy metrics (MAE, RMSE, R2, Bias, etc). The physics and
# trajectory loss terms are a TRAINING-time concept only -- there is
# nothing to evaluate about them here, this is a plain forward pass.
#
# Deliberately cheap and CPU-friendly: this is a small network doing a
# single forward pass over a few tens of thousands of rows, not a training
# loop -- there is no need for a GPU or a SLURM job for this step. Meant
# to be run locally, after copying the trained checkpoint down from the
# cluster. If evaluating on the cluster instead, use
# jobs/pinn_noenv/evaluate_pinn_noenv.sh.

import argparse
import json
import time

import joblib
import pandas as pd

from models.common.metrics import compute_metrics
from models.common.run_logging import RunTimer, format_error, write_run_log, write_started_marker
from models.common.saving import model_output_dir
from models.common.splits import DEFAULT_K_FOLDS, SPLIT_SEED
from models.common.torch_data import TARGET_COLUMN, build_tensors, load_split_table, select_device
from models.pinn_noenv.pinn_noenv import load_best_model, predict

MODEL_NAME = "pinn_noenv"


def run_for_cohort(
    cohort, split_type, run_name=None, split_seed=SPLIT_SEED,
    k_folds=DEFAULT_K_FOLDS, held_out_fold=0,
):
    # run_name only changes where the checkpoint is READ from -- see the
    # matching note in run_pinn_noenv.py. The underlying data table to
    # evaluate on always uses the plain MODEL_NAME.
    output_model_name = run_name if run_name else MODEL_NAME
    fold_suffix = f", fold={held_out_fold}/{k_folds}" if split_type == "spatial_block_kfold" else ""
    print(f"===== {cohort} ({output_model_name}, {split_type}{fold_suffix}) — EVALUATE ONLY =====")
    device = select_device()
    print(f"  Using device: {device}")

    timer = RunTimer().start()
    attempt_id = write_started_marker(
        model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="evaluate",
        is_test_run=False, device=str(device), hyperparameters={},
    )

    try:
        if split_type == "spatial_block_kfold":
            output_dir = model_output_dir(output_model_name, cohort, f"fold_{held_out_fold}", split_type=split_type)
        else:
            output_dir = model_output_dir(output_model_name, cohort, split_type=split_type)
        checkpoints_dir = output_dir / "checkpoints"
        preprocessing_dir = output_dir / "preprocessing"

        # ----- Load everything run_pinn_noenv.py already saved -----
        with open(checkpoints_dir / "architecture.json") as f:
            architecture = json.load(f)
        n_other_features = architecture["n_other_features"]
        # .get(...): a checkpoint saved before 2026-08-02 has no "hidden_layer_sizes" key at
        # all -- treated the same as an explicit None (the original 3x128 network).
        hidden_layer_sizes = architecture.get("hidden_layer_sizes")

        scaler_age = joblib.load(preprocessing_dir / "scaler_age.joblib")
        scaler_other_features = joblib.load(preprocessing_dir / "scaler_other_features.joblib")
        scaler_height = joblib.load(preprocessing_dir / "scaler_height.joblib")
        with open(preprocessing_dir / "encoded_column_names.json") as f:
            encoded_column_names = json.load(f)

        model = load_best_model(n_other_features, device, checkpoints_dir, hidden_layer_sizes=hidden_layer_sizes)

        # ----- Load ONLY the test rows -----
        split_df = load_split_table(
            cohort, split_type, split_seed=split_seed,
            k_folds=k_folds, held_out_fold=held_out_fold,
        )
        test_df = split_df[split_df["split"] == "test"]

        age_test, other_test, target_test = build_tensors(
            test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
        )

        # ----- Make predictions and unscale them back to real metres -----
        # Timed for a runtime-comparison chart -- see evaluate_dnn_noenv.py's identical comment.
        inference_start_time = time.time()
        predicted_height_test_scaled = predict(model, age_test, other_test)
        inference_elapsed_seconds = time.time() - inference_start_time
        predicted_height_test = scaler_height.inverse_transform(
            predicted_height_test_scaled.cpu().numpy()
        ).flatten()
        observed_height_test = test_df[TARGET_COLUMN].values

        metrics = compute_metrics(observed_height_test, predicted_height_test, age=test_df["Age"].values)
        metrics["inference_seconds_total"] = inference_elapsed_seconds
        metrics["inference_ms_per_plot"] = (inference_elapsed_seconds / len(test_df)) * 1000

        predictions_df = pd.DataFrame({
            "identification": test_df["identification"].values,
            "blk": test_df["blk"].values,
            "cpmt": test_df["cpmt"].values,
            "LiDAR_year": test_df["LiDAR_year"].values,
            "Age": test_df["Age"].values,
            "observed_top_height": observed_height_test,
            "predicted_top_height": predicted_height_test,
            "residual": observed_height_test - predicted_height_test,
            "split": "test",
        })

        predictions_df.to_csv(output_dir / "predictions.csv", index=False)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        write_run_log(
            attempt_id=attempt_id,
            model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="evaluate",
            status="success", is_test_run=False, device=str(device),
            hyperparameters={}, metrics=metrics, error=None,
            output_dir=output_dir, runtime_seconds=timer.elapsed_seconds(), n_rows_fit=len(test_df),
        )

        print(f"  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}  Bias={metrics['bias']:+.4f}")
        print(f"  Saved -> {output_dir / 'predictions.csv'}")
        print(f"  Saved -> {output_dir / 'metrics.json'}")
        print()
        return metrics

    except Exception as error:
        write_run_log(
            attempt_id=attempt_id,
            model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="evaluate",
            status="failed", is_test_run=False, device=str(device),
            hyperparameters={}, metrics=None, error=format_error(error),
            output_dir=None, runtime_seconds=timer.elapsed_seconds(),
        )
        print(f"  WARNING: {output_model_name} evaluation failed for {cohort}: {error}")
        run_name_flag = f" --run-name {run_name}" if run_name else ""
        print(
            f"  (Did you run 'python -m models.pinn_noenv.run_pinn_noenv "
            f"--cohort {cohort} --split-type {split_type}{run_name_flag}' first?)"
        )
        print()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default=None, help="Omit to run both cohorts.")
    parser.add_argument(
        "--split-type", choices=["temporal", "spatial_block", "spatial_block_kfold", "temporal_narrow_gap"], default="temporal"
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Must match the --run-name used when fitting, if one was used (e.g. for a physics_weight/trajectory_weight sweep).",
    )
    parser.add_argument(
        "--split-seed", type=int, default=SPLIT_SEED,
        help=f"Must match the --split-seed used when fitting (default {SPLIT_SEED}).",
    )
    parser.add_argument("--n-folds", type=int, default=DEFAULT_K_FOLDS)
    parser.add_argument("--fold-index", type=int, default=0)
    args = parser.parse_args()

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]

    all_metrics = {}
    for cohort in cohorts:
        all_metrics[cohort] = run_for_cohort(
            cohort, args.split_type, args.run_name, split_seed=args.split_seed,
            k_folds=args.n_folds, held_out_fold=args.fold_index,
        )

    print("===== Summary: test-split metrics =====")
    for cohort, metrics in all_metrics.items():
        if metrics is None:
            print(f"  {cohort}: FAILED, see outputs/run_logs/")
            continue
        print(f"  {cohort}: MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}  Bias={metrics['bias']:+.4f}")


if __name__ == "__main__":
    main()

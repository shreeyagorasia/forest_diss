# Run as: python -m models.dnn_env_terrain.evaluate_dnn_env_terrain --cohort 4survey
#
# EVALUATES an already-trained dnn_env_terrain checkpoint on the held-out test split. Mirrors
# evaluate_dnn_noenv.py's structure exactly (cheap, CPU-friendly, the one place the test split
# is touched) -- see that file's own comments for the full reasoning, not repeated here.

import argparse
import json
import time

import joblib
import pandas as pd
import torch

from models.common.metrics import compute_metrics
from models.common.run_logging import RunTimer, format_error, write_run_log, write_started_marker
from models.common.saving import model_output_dir
from models.common.splits import DEFAULT_K_FOLDS, SPLIT_SEED
from models.common.torch_data import TARGET_COLUMN, build_tensors, build_terrain_tensor, load_split_table_with_terrain, select_device
from models.dnn_env_terrain.dnn_env_terrain import load_best_model, predict

MODEL_NAME = "dnn_env_terrain"


def run_for_cohort(cohort, split_type, run_name=None, split_seed=SPLIT_SEED, k_folds=DEFAULT_K_FOLDS, held_out_fold=0):
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

        with open(checkpoints_dir / "architecture.json") as f:
            architecture = json.load(f)
        n_other_features = architecture["n_other_features"]
        # .get(...): a checkpoint saved before 2026-08-02 has no "hidden_layer_sizes" key at
        # all -- treated the same as an explicit None (the original 3x128 network).
        hidden_layer_sizes = architecture.get("hidden_layer_sizes")

        scaler_age = joblib.load(preprocessing_dir / "scaler_age.joblib")
        scaler_other_features = joblib.load(preprocessing_dir / "scaler_other_features.joblib")
        scaler_height = joblib.load(preprocessing_dir / "scaler_height.joblib")
        scaler_terrain = joblib.load(preprocessing_dir / "scaler_terrain.joblib")
        with open(preprocessing_dir / "encoded_column_names.json") as f:
            encoded_column_names = json.load(f)
        # Read back the EXACT feature_columns list run_dnn_env_terrain.py saved when this
        # checkpoint was trained -- not re-derived from --feature-set here, so evaluation always
        # matches training even if ENV_TERRAIN_FEATURE_SETS's definitions change later.
        with open(preprocessing_dir / "terrain_feature_columns.json") as f:
            feature_columns = json.load(f)

        model = load_best_model(n_other_features, device, checkpoints_dir, hidden_layer_sizes=hidden_layer_sizes)

        split_df = load_split_table_with_terrain(
            cohort, split_type, feature_columns, split_seed=split_seed, k_folds=k_folds, held_out_fold=held_out_fold,
        )
        test_df = split_df[split_df["split"] == "test"]

        age_test, other_test_noenv, target_test = build_tensors(
            test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
        )
        terrain_test = build_terrain_tensor(test_df, scaler_terrain, feature_columns, device)
        other_test = torch.cat([other_test_noenv, terrain_test], dim=1)

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
            f"  (Did you run 'python -m models.dnn_env_terrain.run_dnn_env_terrain "
            f"--cohort {cohort} --split-type {split_type}{run_name_flag}' first?)"
        )
        print()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default=None, help="Omit to run both cohorts.")
    parser.add_argument(
        "--split-type",
        choices=["temporal", "spatial_block", "spatial_block_kfold", "temporal_narrow_gap", "plot_level"],
        default="temporal",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Must match the --run-name used when fitting, if one was used.",
    )
    parser.add_argument(
        "--split-seed", type=int, default=SPLIT_SEED,
        help=f"Must match the --split-seed used when fitting (default {SPLIT_SEED}).",
    )
    parser.add_argument(
        "--n-folds", type=int, default=DEFAULT_K_FOLDS,
        help="Must match --n-folds used when fitting, for --split-type spatial_block_kfold.",
    )
    parser.add_argument(
        "--fold-index", type=int, default=0,
        help="Must match --fold-index used when fitting, for --split-type spatial_block_kfold.",
    )
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

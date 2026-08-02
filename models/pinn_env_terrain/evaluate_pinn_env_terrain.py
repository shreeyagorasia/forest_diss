# Run as: python -m models.pinn_env_terrain.evaluate_pinn_env_terrain --cohort 4survey
#
# EVALUATES an already-trained pinn_env_terrain checkpoint on the held-out test split. Mirrors
# evaluate_pinn_noenv.py's structure (cheap, CPU-friendly, the one place the test split is
# touched) -- see that file's own comments for the full reasoning, not repeated here.

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from models.common.metrics import compute_metrics
from models.common.run_logging import RunTimer, format_error, write_run_log, write_started_marker
from models.common.saving import model_output_dir
from models.common.splits import SPLIT_SEED
from models.common.torch_data import TARGET_COLUMN, build_tensors, build_terrain_tensor, load_split_table_with_terrain, select_device
from models.pinn_env_terrain.pinn_env_terrain import load_best_model, predict, predict_y_max

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "pinn_env_terrain"


def run_for_cohort(cohort, split_type, run_name=None, split_seed=SPLIT_SEED):
    output_model_name = run_name if run_name else MODEL_NAME
    print(f"===== {cohort} ({output_model_name}, {split_type}) — EVALUATE ONLY =====")
    device = select_device()
    print(f"  Using device: {device}")

    timer = RunTimer().start()
    attempt_id = write_started_marker(
        model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="evaluate",
        is_test_run=False, device=str(device), hyperparameters={},
    )

    try:
        output_dir = model_output_dir(output_model_name, cohort, split_type=split_type)
        checkpoints_dir = output_dir / "checkpoints"
        preprocessing_dir = output_dir / "preprocessing"

        with open(checkpoints_dir / "architecture.json") as f:
            architecture = json.load(f)
        n_other_features = architecture["n_other_features"]
        n_terrain_features = architecture["n_terrain_features"]
        # .get(...): a checkpoint saved before 2026-08-02 has no "hidden_layer_sizes" key at
        # all -- treated the same as an explicit None (the original 3x128 main network).
        hidden_layer_sizes = architecture.get("hidden_layer_sizes")

        scaler_age = joblib.load(preprocessing_dir / "scaler_age.joblib")
        scaler_other_features = joblib.load(preprocessing_dir / "scaler_other_features.joblib")
        scaler_height = joblib.load(preprocessing_dir / "scaler_height.joblib")
        scaler_terrain = joblib.load(preprocessing_dir / "scaler_terrain.joblib")
        with open(preprocessing_dir / "encoded_column_names.json") as f:
            encoded_column_names = json.load(f)
        with open(preprocessing_dir / "terrain_feature_columns.json") as f:
            feature_columns = json.load(f)

        # The frozen CR anchor's y_max is the same split-MATCHED anchor run_pinn_env_terrain.py's
        # load_cr_params() used to train this checkpoint -- re-read directly from the same
        # outputs/<split_type>/chapman_richards/<cohort>/params.json path rather than trusting a
        # copy. BUG FIX (2026-08-01): this used to read the unprefixed, pooled
        # outputs/chapman_richards/<cohort>/params.json path -- harmless for spatial_block (its
        # split-matched y_max happens to be nearly identical to the pooled value, checked
        # directly), but WRONG for temporal/temporal_narrow_gap, where the two differ by
        # 3-11% (e.g. 4survey temporal: pooled 51.96m vs split-matched 46.48m). Did not affect
        # any height prediction or metric (predict() never uses global_y_max at all) -- only
        # the reported "learned_y_max" column/anchor print, which a future check comparing the
        # learned y_max map against known terrain effects would rely on being correct.
        with open(PROJECT_ROOT / "outputs" / split_type / "chapman_richards" / cohort / "params.json") as f:
            global_y_max = json.load(f)["y_max"]

        model = load_best_model(
            n_other_features, n_terrain_features, device, checkpoints_dir,
            hidden_layer_sizes=hidden_layer_sizes,
        )

        split_df = load_split_table_with_terrain(cohort, split_type, feature_columns, split_seed=split_seed)
        test_df = split_df[split_df["split"] == "test"]

        age_test, other_test, target_test = build_tensors(
            test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
        )
        terrain_test = build_terrain_tensor(test_df, scaler_terrain, feature_columns, device)

        predicted_height_test_scaled = predict(model, age_test, other_test)
        predicted_height_test = scaler_height.inverse_transform(
            predicted_height_test_scaled.cpu().numpy()
        ).flatten()
        observed_height_test = test_df[TARGET_COLUMN].values

        # The learned, plot-specific y_max map -- the actual interpretable output this whole
        # model exists to produce, saved alongside the ordinary height predictions so it can be
        # mapped/compared against terrain directly (e.g. in spatial_autocorrelation_terrain.ipynb),
        # not just used internally by the physics loss.
        learned_y_max_test = predict_y_max(model, terrain_test, global_y_max).cpu().numpy().flatten()

        metrics = compute_metrics(observed_height_test, predicted_height_test, age=test_df["Age"].values)

        predictions_df = pd.DataFrame({
            "identification": test_df["identification"].values,
            "blk": test_df["blk"].values,
            "cpmt": test_df["cpmt"].values,
            "LiDAR_year": test_df["LiDAR_year"].values,
            "Age": test_df["Age"].values,
            "observed_top_height": observed_height_test,
            "predicted_top_height": predicted_height_test,
            "residual": observed_height_test - predicted_height_test,
            "learned_y_max": learned_y_max_test,
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
        print(
            f"  Learned y_max range: {learned_y_max_test.min():.2f}m to {learned_y_max_test.max():.2f}m "
            f"(global anchor: {global_y_max:.2f}m)"
        )
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
            f"  (Did you run 'python -m models.pinn_env_terrain.run_pinn_env_terrain "
            f"--cohort {cohort} --split-type {split_type}{run_name_flag}' first?)"
        )
        print()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default=None, help="Omit to run both cohorts.")
    parser.add_argument(
        "--split-type", choices=["temporal", "spatial_block", "temporal_narrow_gap"], default="temporal"
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Must match the --run-name used when fitting, if one was used.",
    )
    parser.add_argument(
        "--split-seed", type=int, default=SPLIT_SEED,
        help=f"Must match the --split-seed used when fitting (default {SPLIT_SEED}).",
    )
    args = parser.parse_args()

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]

    all_metrics = {}
    for cohort in cohorts:
        all_metrics[cohort] = run_for_cohort(cohort, args.split_type, args.run_name, split_seed=args.split_seed)

    print("===== Summary: test-split metrics =====")
    for cohort, metrics in all_metrics.items():
        if metrics is None:
            print(f"  {cohort}: FAILED, see outputs/run_logs/")
            continue
        print(f"  {cohort}: MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}  Bias={metrics['bias']:+.4f}")


if __name__ == "__main__":
    main()

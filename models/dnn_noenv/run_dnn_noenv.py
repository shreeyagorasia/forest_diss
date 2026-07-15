# Run as: python -m models.dnn_noenv.run_dnn_noenv --cohort 4survey
#     or: python -m models.dnn_noenv.run_dnn_noenv --cohort 4survey --max-epochs 5
#
# Trains the plain DNN (no-environment feature set) on temporal_split's
# training years, early-stopping on the validation year (2021), then
# evaluates once on the held-out test year (2023). See
# documentation/model_instructions/age_only_dnn_pinn_instructions.md.
#
# --max-epochs is exposed on the CLI specifically so a quick local sanity
# check (e.g. --max-epochs 5) can be run before handing the real, full-length
# training job to SLURM with a much larger --max-epochs.

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from models.common.metrics import compute_metrics
from models.common.saving import get_git_commit
from models.common.splits import TEMPORAL_YEARS
from models.common.torch_data import (
    TARGET_COLUMN,
    build_tensors,
    encode_thinning_status,
    fit_scalers,
    load_split_table,
    load_trajectory_pairs,
    print_pre_training_diagnostic,
    select_device,
)
from models.dnn_noenv.dnn_noenv import (
    BATCH_SIZE,
    L1_COEFFICIENT,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    fit,
    predict,
    save_checkpoints,
    save_run_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "dnn_noenv"
DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 500
DEFAULT_EARLY_STOPPING_PATIENCE = 20


def run_for_cohort(cohort, max_epochs, early_stopping_patience, seed):
    print(f"===== {cohort} ({MODEL_NAME}) =====")
    device = select_device()
    print(f"  Using device: {device}")

    split_df = load_split_table(cohort, MODEL_NAME)
    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]

    eligible_plot_ids = set(split_df["identification"].unique())
    train_years = TEMPORAL_YEARS[cohort]["train_years"]
    pairs_df = load_trajectory_pairs(cohort, eligible_plot_ids, train_years)
    print_pre_training_diagnostic(cohort, split_df, pairs_df)

    scaler_age, scaler_other_features, scaler_height = fit_scalers(train_df)
    encoded_column_names = encode_thinning_status(train_df).columns.tolist()

    age_train, other_train, target_train = build_tensors(
        train_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
    )
    age_val, other_val, target_val = build_tensors(
        val_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
    )
    age_test, other_test, target_test = build_tensors(
        test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
    )

    n_other_features = other_train.shape[1]
    best_model, final_model_state, history_df = fit(
        age_train, other_train, target_train,
        age_val, other_val, target_val,
        n_other_features, device, seed,
        max_epochs, early_stopping_patience,
    )
    print(f"  Trained for {len(history_df)} epochs. Final val_loss={history_df['val_loss'].iloc[-1]:.6f}")

    predicted_height_test_scaled = predict(best_model, age_test, other_test)
    predicted_height_test = scaler_height.inverse_transform(
        predicted_height_test_scaled.cpu().numpy()
    ).flatten()
    observed_height_test = test_df[TARGET_COLUMN].values

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
        "split": "test",
    })

    output_dir = PROJECT_ROOT / "outputs" / MODEL_NAME / cohort
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    save_checkpoints(best_model, final_model_state, n_other_features, output_dir / "checkpoints")

    preprocessing_dir = output_dir / "preprocessing"
    preprocessing_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler_age, preprocessing_dir / "scaler_age.joblib")
    joblib.dump(scaler_other_features, preprocessing_dir / "scaler_other_features.joblib")
    joblib.dump(scaler_height, preprocessing_dir / "scaler_height.joblib")
    with open(preprocessing_dir / "encoded_column_names.json", "w") as f:
        json.dump(encoded_column_names, f, indent=2)

    hyperparameters = {
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_factor": LR_SCHEDULER_FACTOR,
        "lr_scheduler_patience": LR_SCHEDULER_PATIENCE,
        "l1_coefficient": L1_COEFFICIENT,
        "batch_size": BATCH_SIZE,
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "seed": seed,
        "temporal_split_years": TEMPORAL_YEARS[cohort],
        "n_epochs_trained": len(history_df),
        "git_commit": get_git_commit(),
    }
    save_run_metadata(cohort, len(train_df), hyperparameters, output_dir / "run_metadata.json")

    print(f"  Saved all outputs -> {output_dir}")
    print()
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default=None, help="Omit to run both cohorts.")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]

    all_metrics = {}
    for cohort in cohorts:
        all_metrics[cohort] = run_for_cohort(cohort, args.max_epochs, args.patience, args.seed)

    print("===== Summary: test-split metrics =====")
    for cohort, metrics in all_metrics.items():
        print(f"  {cohort}: MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  R2={metrics['r2']:.4f}  Bias={metrics['bias']:+.4f}")


if __name__ == "__main__":
    main()

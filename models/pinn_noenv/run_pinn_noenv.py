# Run as: python -m models.pinn_noenv.run_pinn_noenv --cohort 4survey
#     or: python -m models.pinn_noenv.run_pinn_noenv --cohort 4survey --max-epochs 5
#     or: python -m models.pinn_noenv.run_pinn_noenv --cohort 4survey --split-type spatial_block
#
# TRAINS the CR-PINN (no-environment feature set, two physics loss terms on
# top of plain MSE) under either temporal_split or spatial_block_split --
# see --split-type below. Meant to run on the SLURM cluster where the GPU
# is -- see jobs/pinn_noenv/run_pinn_noenv.sh.
#
# The frozen CR physics anchor (load_cr_params below) always reads the
# plot_level Chapman-Richards fit, regardless of which split_type the PINN
# itself uses -- this is "Route A" (see documentation/experiment_log.md's
# Decisions log): y_max/k/p are treated as population-level biological
# constants, not something requiring the PINN's own train/test discipline,
# so the same reasoning applies whether the PINN is being tested temporally
# or spatially.
#
# This script deliberately does NOT touch the test split (2023) or compute
# any accuracy metrics. That is a separate, cheap step
# (models/pinn_noenv/evaluate_pinn_noenv.py) meant to run afterwards,
# locally on a laptop CPU. See
# documentation/model_instructions/age_only_dnn_pinn_instructions.md.
#
# --max-epochs is exposed on the CLI specifically so a quick local sanity
# check (e.g. --max-epochs 5) can be run before handing the real, full-length
# training job to SLURM with a much larger --max-epochs. A run with
# max_epochs below TEST_RUN_MAX_EPOCHS_THRESHOLD (see models/common/
# run_logging.py) is automatically tagged is_test_run=True in the log.
#
# The WHOLE run (data loading through saving) is wrapped in one
# started/success/failed log triple in outputs/run_logs/ -- including data
# loading and reading CR's frozen params, since a missing input file is
# exactly the kind of failure this log exists to catch, not just a
# training-loop error.

import argparse
import json
from pathlib import Path

import joblib

from models.common.run_logging import (
    TEST_RUN_MAX_EPOCHS_THRESHOLD,
    RunTimer,
    format_error,
    write_run_log,
    write_started_marker,
)
from models.common.saving import get_git_commit, model_output_dir
from models.common.splits import TEMPORAL_YEARS
from models.common.torch_data import (
    build_pair_tensors,
    build_tensors,
    encode_thinning_status,
    fit_scalers,
    load_split_table,
    load_trajectory_pairs,
    print_pre_training_diagnostic,
    select_device,
)
from models.pinn_noenv.pinn_noenv import (
    BATCH_SIZE,
    L1_COEFFICIENT,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    PAIRS_BATCH_SIZE,
    PHYSICS_WEIGHT,
    TRAJECTORY_WEIGHT,
    fit,
    save_checkpoints,
    save_run_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "pinn_noenv"
DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 500
DEFAULT_EARLY_STOPPING_PATIENCE = 20


def load_cr_params(cohort):
    # CR's y_max/k/p, already fitted by models/chapman_richards -- read as
    # plain floats and treated as FROZEN constants. Never refit here.
    params_path = PROJECT_ROOT / "outputs" / "chapman_richards" / cohort / "params.json"
    with open(params_path) as f:
        params = json.load(f)
    return {"y_max": params["y_max"], "k": params["k"], "p": params["p"]}


def run_for_cohort(cohort, split_type, max_epochs, early_stopping_patience, seed):
    print(f"===== {cohort} ({MODEL_NAME}, {split_type}) — FIT ONLY, no test-set evaluation =====")

    is_test_run = max_epochs < TEST_RUN_MAX_EPOCHS_THRESHOLD
    device = select_device()
    print(f"  Using device: {device}")

    hyperparameters = {
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_factor": LR_SCHEDULER_FACTOR,
        "lr_scheduler_patience": LR_SCHEDULER_PATIENCE,
        "l1_coefficient": L1_COEFFICIENT,
        "physics_weight": PHYSICS_WEIGHT,
        "trajectory_weight": TRAJECTORY_WEIGHT,
        "batch_size": BATCH_SIZE,
        "pairs_batch_size": PAIRS_BATCH_SIZE,
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "seed": seed,
    }
    # See the matching note in run_dnn_noenv.py -- only meaningful for
    # temporal_split, where train/val/test years are fixed ahead of time.
    if split_type == "temporal":
        hyperparameters["temporal_split_years"] = TEMPORAL_YEARS[cohort]

    # Write a "started" log entry BEFORE doing any real work -- if SLURM
    # kills this job (out of memory, out of time, node crash) rather than
    # it failing with a normal Python error, this entry is the only record
    # left behind, and having no matching "success"/"failed" entry later
    # IS the signal that something went wrong.
    timer = RunTimer().start()
    attempt_id = write_started_marker(
        model_name=MODEL_NAME, cohort=cohort, split_type=split_type, run_phase="fit",
        is_test_run=is_test_run, device=str(device), hyperparameters=hyperparameters,
    )

    try:
        cr_params = load_cr_params(cohort)
        print(f"  Frozen CR params: y_max={cr_params['y_max']:.4f}, k={cr_params['k']:.6f}, p={cr_params['p']:.6f}")

        # ----- Load and prepare the data -----
        split_df = load_split_table(cohort, MODEL_NAME, split_type)
        train_df = split_df[split_df["split"] == "train"]
        val_df = split_df[split_df["split"] == "val"]
        # Note: test_df is deliberately never loaded here at all.

        pairs_df = load_trajectory_pairs(cohort, split_df)
        print_pre_training_diagnostic(cohort, split_df, pairs_df)

        scaler_age, scaler_other_features, scaler_height = fit_scalers(train_df)
        encoded_column_names = encode_thinning_status(train_df).columns.tolist()

        age_train, other_train, target_train = build_tensors(
            train_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
        )
        age_val, other_val, target_val = build_tensors(
            val_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
        )
        pair_tensors = build_pair_tensors(
            pairs_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
        )

        # ----- Train -----
        n_other_features = other_train.shape[1]
        best_model, final_model_state, history_df = fit(
            age_train, other_train, target_train,
            age_val, other_val, target_val,
            pair_tensors, cr_params, scaler_age, scaler_height,
            n_other_features, device, seed,
            max_epochs, early_stopping_patience,
        )
        last_row = history_df.iloc[-1]
        best_val_loss = float(history_df["val_loss"].min())
        print(
            f"  Trained for {len(history_df)} epochs in {timer.elapsed_seconds():.1f}s. "
            f"best_val_loss={best_val_loss:.6f}  final: data_loss={last_row['data_loss']:.6f}  "
            f"physics_loss={last_row['physics_loss']:.6f}  trajectory_loss={last_row['trajectory_loss']:.6f}"
        )

        # ----- Save everything needed to evaluate this model LATER, on a
        # different machine -----
        output_dir = model_output_dir(MODEL_NAME, cohort, split_type=split_type)
        output_dir.mkdir(parents=True, exist_ok=True)

        history_df.to_csv(output_dir / "training_history.csv", index=False)
        save_checkpoints(best_model, final_model_state, n_other_features, output_dir / "checkpoints")

        preprocessing_dir = output_dir / "preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler_age, preprocessing_dir / "scaler_age.joblib")
        joblib.dump(scaler_other_features, preprocessing_dir / "scaler_other_features.joblib")
        joblib.dump(scaler_height, preprocessing_dir / "scaler_height.joblib")
        with open(preprocessing_dir / "encoded_column_names.json", "w") as f:
            json.dump(encoded_column_names, f, indent=2)

        run_metadata_hyperparameters = {
            **hyperparameters,
            "n_epochs_trained": len(history_df),
            "n_trajectory_pairs": len(pairs_df),
            "frozen_cr_params": cr_params,
            "git_commit": get_git_commit(),
        }
        save_run_metadata(cohort, len(train_df), run_metadata_hyperparameters, output_dir / "run_metadata.json")

        # "metrics" here is just the loss numbers reached during training
        # -- NOT the real MAE/RMSE/R2/Bias test-set metrics, which only
        # exist once evaluate_pinn_noenv.py has been run.
        write_run_log(
            attempt_id=attempt_id,
            model_name=MODEL_NAME, cohort=cohort, split_type=split_type, run_phase="fit",
            status="success", is_test_run=is_test_run, device=str(device),
            hyperparameters={
                **hyperparameters,
                "n_epochs_trained": len(history_df),
                "n_trajectory_pairs": len(pairs_df),
                "frozen_cr_params": cr_params,
            },
            metrics={
                "best_val_loss": best_val_loss,
                "final_val_loss": float(last_row["val_loss"]),
                "final_data_loss": float(last_row["data_loss"]),
                "final_physics_loss": float(last_row["physics_loss"]),
                "final_trajectory_loss": float(last_row["trajectory_loss"]),
            },
            error=None,
            output_dir=output_dir, runtime_seconds=timer.elapsed_seconds(), n_rows_fit=len(train_df),
        )

        print(f"  Saved checkpoint + scalers + history -> {output_dir}")
        print(f"  Next step: python -m models.pinn_noenv.evaluate_pinn_noenv --cohort {cohort} --split-type {split_type}")
        print()
        return best_val_loss

    except Exception as error:
        write_run_log(
            attempt_id=attempt_id,
            model_name=MODEL_NAME, cohort=cohort, split_type=split_type, run_phase="fit",
            status="failed", is_test_run=is_test_run, device=str(device),
            hyperparameters=hyperparameters, metrics=None, error=format_error(error),
            output_dir=None, runtime_seconds=timer.elapsed_seconds(),
        )
        print(f"  WARNING: {MODEL_NAME} fit failed for {cohort}: {error}")
        print()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default=None, help="Omit to run both cohorts.")
    parser.add_argument("--split-type", choices=["temporal", "spatial_block"], default="temporal")
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]

    results = {}
    for cohort in cohorts:
        results[cohort] = run_for_cohort(cohort, args.split_type, args.max_epochs, args.patience, args.seed)

    print("===== Summary: best validation loss reached =====")
    for cohort, best_val_loss in results.items():
        if best_val_loss is None:
            print(f"  {cohort}: FAILED, see outputs/run_logs/")
            continue
        print(f"  {cohort}: best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()

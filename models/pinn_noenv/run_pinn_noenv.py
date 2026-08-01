# Run as: python -m models.pinn_noenv.run_pinn_noenv --cohort 4survey
#     or: python -m models.pinn_noenv.run_pinn_noenv --cohort 4survey --max-epochs 5
#     or: python -m models.pinn_noenv.run_pinn_noenv --cohort 4survey --split-type spatial_block
#
# TRAINS the CR-PINN (no-environment feature set, two physics loss terms on
# top of plain MSE) under either temporal_split or spatial_block_split --
# see --split-type below. Meant to run on the SLURM cluster where the GPU
# is -- see jobs/pinn_noenv/run_pinn_noenv.sh.
#
# The frozen CR physics anchor (load_cr_params below) reads the split-MATCHED
# Chapman-Richards fit -- fit using only THIS split_type's own train-assigned plots, not the
# pooled/"cr_pooled" plot_level fit this file used before 2026-08-01. See load_cr_params()'s own
# comment for why: cr_pooled was confirmed to leak (its random 60% training plots inevitably
# overlapped with whichever plots a given split_type later assigned to test). There is no
# --cr-variant flag to opt back into cr_pooled -- it wasn't a legitimate alternative to keep
# choosable, just a bug, so both PINN models now always use cr_matched, unconditionally.
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
    GRAD_CLIP_MAX_NORM,
    L1_COEFFICIENT,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    PAIRS_BATCH_SIZE,
    PHYSICS_WEIGHT,
    TRAJECTORY_WEIGHT,
    VAL_LOSS_SMOOTHING_WINDOW,
    WEIGHT_DECAY,
    fit,
    save_checkpoints,
    save_run_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "pinn_noenv"
DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 500
DEFAULT_EARLY_STOPPING_PATIENCE = 40
DEFAULT_OPTIMIZER = "adam"


def load_cr_params(cohort, split_type):
    # Split-MATCHED CR anchor (2026-08-01 fix, switched from the pooled/"cr_pooled" version
    # this file used before) -- fit using ONLY this split_type's own train-assigned plots
    # (models/baselines/run_baselines.py's cr_train_df), not a random 60% plot_level split
    # unrelated to spatial_block/temporal. The old pooled version was confirmed to leak: its
    # random 60% training plots inevitably overlap with whichever plots a given split_type later
    # assigns to test, since the two splits were never coordinated (verified directly: the saved
    # n_rows_fit was exactly 60% of filtered rows -- plot_level_split's train share, not this
    # split's own). Read as plain floats and treated as FROZEN constants either way -- never
    # refit here, this file already exists on disk for every split_type PINN uses, built as a
    # side effect of the CR baseline's own per-split fit.
    params_path = PROJECT_ROOT / "outputs" / split_type / "chapman_richards" / cohort / "params.json"
    with open(params_path) as f:
        params = json.load(f)
    return {"y_max": params["y_max"], "k": params["k"], "p": params["p"]}


def run_for_cohort(
    cohort, split_type, max_epochs, early_stopping_patience, seed, optimizer_name,
    physics_weight, trajectory_weight, batch_size, pairs_batch_size, run_name,
):
    # run_name only changes where results are SAVED (output_dir below) and
    # how this run is labelled in outputs/run_logs/ -- it never changes
    # which underlying data table gets loaded (that always uses the plain
    # MODEL_NAME, "pinn_noenv", a few lines down), since a physics_weight/
    # trajectory_weight sweep is still the same network on the same data,
    # just a different loss-weighting choice. Without a distinct run_name,
    # every sweep run would silently overwrite the primary result at the
    # same output_dir -- see the pinn_noenv_crmatched naming precedent in
    # documentation/experiment_log.md for why a different PINN
    # configuration gets its own name rather than reusing MODEL_NAME.
    output_model_name = run_name if run_name else MODEL_NAME
    print(f"===== {cohort} ({output_model_name}, {split_type}) — FIT ONLY, no test-set evaluation =====")

    is_test_run = max_epochs < TEST_RUN_MAX_EPOCHS_THRESHOLD
    device = select_device()
    print(f"  Using device: {device}")

    hyperparameters = {
        "learning_rate": LEARNING_RATE,
        "lr_scheduler_factor": LR_SCHEDULER_FACTOR,
        "lr_scheduler_patience": LR_SCHEDULER_PATIENCE,
        "l1_coefficient": L1_COEFFICIENT,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "val_loss_smoothing_window": VAL_LOSS_SMOOTHING_WINDOW,
        "optimizer_name": optimizer_name,
        "physics_weight": physics_weight,
        "trajectory_weight": trajectory_weight,
        "batch_size": batch_size,
        "pairs_batch_size": pairs_batch_size,
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
        model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="fit",
        is_test_run=is_test_run, device=str(device), hyperparameters=hyperparameters,
    )

    try:
        cr_params = load_cr_params(cohort, split_type)
        print(f"  Frozen CR params (split-matched, {split_type}): y_max={cr_params['y_max']:.4f}, k={cr_params['k']:.6f}, p={cr_params['p']:.6f}")

        # ----- Load and prepare the data -----
        split_df = load_split_table(cohort, split_type)
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
            optimizer_name=optimizer_name,
            physics_weight=physics_weight, trajectory_weight=trajectory_weight,
            batch_size=batch_size, pairs_batch_size=pairs_batch_size,
        )
        last_row = history_df.iloc[-1]
        # The smoothed column is what actually decided which epoch's
        # weights got saved as "best" (see pinn_noenv.py::fit()) -- reporting
        # its minimum here keeps this number consistent with the checkpoint
        # this run actually kept.
        best_val_loss = float(history_df["val_loss_smoothed"].min())
        print(
            f"  Trained for {len(history_df)} epochs in {timer.elapsed_seconds():.1f}s. "
            f"best_val_loss={best_val_loss:.6f}  final: data_loss={last_row['data_loss']:.6f}  "
            f"physics_loss={last_row['physics_loss']:.6f}  trajectory_loss={last_row['trajectory_loss']:.6f}"
        )

        # ----- Save everything needed to evaluate this model LATER, on a
        # different machine -----
        output_dir = model_output_dir(output_model_name, cohort, split_type=split_type)
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
            model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="fit",
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
        run_name_flag = f" --run-name {run_name}" if run_name else ""
        print(
            f"  Next step: python -m models.pinn_noenv.evaluate_pinn_noenv "
            f"--cohort {cohort} --split-type {split_type}{run_name_flag}"
        )
        print()
        return best_val_loss

    except Exception as error:
        write_run_log(
            attempt_id=attempt_id,
            model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="fit",
            status="failed", is_test_run=is_test_run, device=str(device),
            hyperparameters=hyperparameters, metrics=None, error=format_error(error),
            output_dir=None, runtime_seconds=timer.elapsed_seconds(),
        )
        print(f"  WARNING: {output_model_name} fit failed for {cohort}: {error}")
        print()
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default=None, help="Omit to run both cohorts.")
    parser.add_argument(
        "--split-type", choices=["temporal", "spatial_block", "temporal_narrow_gap"], default="temporal"
    )
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--optimizer", choices=["adam", "sgd_momentum"], default=DEFAULT_OPTIMIZER,
        help="adam is the default everywhere so far; sgd_momentum is an A/B-test alternative.",
    )
    parser.add_argument(
        "--physics-weight", type=float, default=PHYSICS_WEIGHT,
        help=f"Weight on the instantaneous-derivative physics loss term. Default {PHYSICS_WEIGHT}, never tuned before.",
    )
    parser.add_argument(
        "--trajectory-weight", type=float, default=TRAJECTORY_WEIGHT,
        help=f"Weight on the trajectory (finite-difference) physics loss term. Default {TRAJECTORY_WEIGHT}, never tuned before.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=(
            f"Main training batch size. Default {BATCH_SIZE} -- 4x smaller than dnn_noenv's 512, "
            "never tuned/matched before; exposed here specifically to test whether that mismatch "
            "(not the physics terms) explains a DNN-vs-PINN convergence difference."
        ),
    )
    parser.add_argument(
        "--pairs-batch-size", type=int, default=PAIRS_BATCH_SIZE,
        help="Trajectory-pairs batch size. Default matches --batch-size's default.",
    )
    parser.add_argument(
        "--run-name", default=None,
        help=(
            "Only changes where results are saved and how this run is labelled in "
            "outputs/run_logs/ -- use this whenever --physics-weight/--trajectory-weight "
            "differ from the defaults, so a sweep run doesn't overwrite the primary "
            "pinn_noenv result at the same output_dir. E.g. pinn_noenv_pw2_tw2."
        ),
    )
    args = parser.parse_args()

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]

    results = {}
    for cohort in cohorts:
        results[cohort] = run_for_cohort(
            cohort, args.split_type, args.max_epochs, args.patience, args.seed, args.optimizer,
            args.physics_weight, args.trajectory_weight, args.batch_size, args.pairs_batch_size,
            args.run_name,
        )

    print("===== Summary: best validation loss reached =====")
    for cohort, best_val_loss in results.items():
        if best_val_loss is None:
            print(f"  {cohort}: FAILED, see outputs/run_logs/")
            continue
        print(f"  {cohort}: best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()

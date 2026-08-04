# Run as: python -m models.dnn_noenv.run_dnn_noenv --cohort 4survey
#     or: python -m models.dnn_noenv.run_dnn_noenv --cohort 4survey --max-epochs 5
#     or: python -m models.dnn_noenv.run_dnn_noenv --cohort 4survey --split-type spatial_block
#
# TRAINS the plain DNN (no-environment feature set) under either
# temporal_split (train on the earliest years, early-stop on 2021) or
# spatial_block_split (train on whole held-in compartments, early-stop on
# held-out val compartments) -- see --split-type below. Meant to run on the
# SLURM cluster where the GPU is -- see jobs/dnn_noenv/run_dnn_noenv.sh.
#
# This script deliberately does NOT touch the test split (2023) or compute
# any accuracy metrics. That is a separate, cheap step
# (models/dnn_noenv/evaluate_dnn_noenv.py) meant to run afterwards, locally
# on a laptop CPU -- there is no reason to spend GPU cluster time on a
# quick forward pass over the test rows once training is already done.
# See documentation/model_instructions/age_only_dnn_pinn_instructions.md.
#
# --max-epochs is exposed on the CLI specifically so a quick local sanity
# check (e.g. --max-epochs 5) can be run before handing the real, full-length
# training job to SLURM with a much larger --max-epochs. A run with
# max_epochs below TEST_RUN_MAX_EPOCHS_THRESHOLD (see models/common/
# run_logging.py) is automatically tagged is_test_run=True in the log.
#
# The WHOLE run (data loading through saving) is wrapped in one
# started/success/failed log triple in outputs/run_logs/ -- including data
# loading, since a missing input file (e.g. data/processed/ not yet
# regenerated on a fresh machine) is exactly the kind of failure this log
# exists to catch, not just a training-loop error.

import argparse
import json

import joblib

from models.common.run_logging import (
    TEST_RUN_MAX_EPOCHS_THRESHOLD,
    RunTimer,
    format_error,
    write_run_log,
    write_started_marker,
)
from models.common.saving import get_git_commit, model_output_dir
from models.common.splits import DEFAULT_K_FOLDS, SPLIT_SEED, TEMPORAL_YEARS
from models.common.torch_model import parse_hidden_layer_sizes
from models.common.torch_data import (
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
    GRAD_CLIP_MAX_NORM,
    L1_COEFFICIENT,
    LEARNING_RATE,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    VAL_LOSS_SMOOTHING_WINDOW,
    WEIGHT_DECAY,
    fit,
    save_checkpoints,
    save_run_metadata,
)

MODEL_NAME = "dnn_noenv"
DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 500
DEFAULT_EARLY_STOPPING_PATIENCE = 40
DEFAULT_OPTIMIZER = "adam"


def run_for_cohort(
    cohort, split_type, max_epochs, early_stopping_patience, seed, optimizer_name,
    run_name=None, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, dropout_rate=0.0,
    hidden_layer_sizes=None, split_seed=SPLIT_SEED, k_folds=DEFAULT_K_FOLDS, held_out_fold=0,
):
    # run_name only changes where results are SAVED (output_dir below) and
    # how this run is labelled in outputs/run_logs/ -- it never changes
    # which underlying data table gets loaded (that always uses the plain
    # MODEL_NAME, "dnn_noenv", a few lines down). Without a distinct
    # run_name, a reseed (different --seed, same everything else) would
    # silently overwrite the primary checkpoint at the same output_dir --
    # mirrors run_pinn_noenv.py's run_name handling for the physics-weight
    # sweep, same reasoning.
    output_model_name = run_name if run_name else MODEL_NAME
    fold_suffix = f", fold={held_out_fold}/{k_folds}" if split_type == "spatial_block_kfold" else ""
    print(f"===== {cohort} ({output_model_name}, {split_type}{fold_suffix}) — FIT ONLY, no test-set evaluation =====")

    # A "test run" is just a quick sanity check with very few epochs (see
    # TEST_RUN_MAX_EPOCHS_THRESHOLD) -- recorded in the log automatically,
    # no extra flag needed, so a 5-epoch debug run never gets mistaken for
    # a real result later.
    is_test_run = max_epochs < TEST_RUN_MAX_EPOCHS_THRESHOLD

    device = select_device()
    print(f"  Using device: {device}")

    hyperparameters = {
        "learning_rate": learning_rate,
        "lr_scheduler_factor": LR_SCHEDULER_FACTOR,
        "lr_scheduler_patience": LR_SCHEDULER_PATIENCE,
        "l1_coefficient": L1_COEFFICIENT,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_max_norm": GRAD_CLIP_MAX_NORM,
        "val_loss_smoothing_window": VAL_LOSS_SMOOTHING_WINDOW,
        "optimizer_name": optimizer_name,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "seed": seed,
        "dropout_rate": dropout_rate,
        "hidden_layer_sizes": hidden_layer_sizes,
    }
    # temporal_split's train/val/test years are a fixed, known-ahead-of-time
    # config worth recording; spatial_block_split has no equivalent (which
    # compartments land where is only known after the split is actually
    # computed below), so this key is only added for temporal.
    if split_type == "temporal":
        hyperparameters["temporal_split_years"] = TEMPORAL_YEARS[cohort]

    # Write a "started" log entry BEFORE doing any real work. If this job
    # gets killed by SLURM (out of memory, ran out of time, node crashed)
    # rather than failing with a normal Python error, this "started" entry
    # is the only record left behind -- there will be no matching
    # "success"/"failed" entry, and that gap is itself the signal that
    # something went wrong. attempt_id links this entry to whichever one
    # gets written at the end.
    timer = RunTimer().start()
    attempt_id = write_started_marker(
        model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="fit",
        is_test_run=is_test_run, device=str(device), hyperparameters=hyperparameters,
    )

    try:
        # ----- Load and prepare the data -----
        split_df = load_split_table(cohort, split_type, split_seed=split_seed, k_folds=k_folds, held_out_fold=held_out_fold)
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

        # ----- Train -----
        n_other_features = other_train.shape[1]
        best_model, final_model_state, history_df = fit(
            age_train, other_train, target_train,
            age_val, other_val, target_val,
            n_other_features, device, seed,
            max_epochs, early_stopping_patience,
            optimizer_name=optimizer_name,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            hidden_layer_sizes=hidden_layer_sizes,
        )
        # The smoothed column is what actually decided which epoch's
        # weights got saved as "best" (see dnn_noenv.py::fit()) -- reporting
        # its minimum here, not the raw val_loss column's, keeps this
        # number consistent with which checkpoint this run actually kept.
        best_val_loss = float(history_df["val_loss_smoothed"].min())
        final_val_loss = float(history_df["val_loss"].iloc[-1])
        print(
            f"  Trained for {len(history_df)} epochs in {timer.elapsed_seconds():.1f}s. "
            f"best_val_loss={best_val_loss:.6f}  final_val_loss={final_val_loss:.6f}"
        )

        # ----- Save everything needed to evaluate this model LATER, on a
        # different machine -----
        if split_type == "spatial_block_kfold":
            output_dir = model_output_dir(output_model_name, cohort, f"fold_{held_out_fold}", split_type=split_type)
        else:
            output_dir = model_output_dir(output_model_name, cohort, split_type=split_type)
        output_dir.mkdir(parents=True, exist_ok=True)

        history_df.to_csv(output_dir / "training_history.csv", index=False)
        save_checkpoints(
            best_model, final_model_state, n_other_features, output_dir / "checkpoints",
            hidden_layer_sizes=hidden_layer_sizes,
        )

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
            "git_commit": get_git_commit(),
        }
        save_run_metadata(cohort, len(train_df), run_metadata_hyperparameters, output_dir / "run_metadata.json")

        # "metrics" here is just the validation loss reached during
        # training -- NOT the real MAE/RMSE/R2/Bias test-set metrics,
        # which only exist once evaluate_dnn_noenv.py has been run. Kept
        # small and clearly named so the two are never confused.
        write_run_log(
            attempt_id=attempt_id,
            model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="fit",
            status="success", is_test_run=is_test_run, device=str(device),
            hyperparameters={**hyperparameters, "n_epochs_trained": len(history_df)},
            metrics={"best_val_loss": best_val_loss, "final_val_loss": final_val_loss},
            error=None,
            output_dir=output_dir, runtime_seconds=timer.elapsed_seconds(), n_rows_fit=len(train_df),
        )

        print(f"  Saved checkpoint + scalers + history -> {output_dir}")
        run_name_flag = f" --run-name {run_name}" if run_name else ""
        print(
            f"  Next step: python -m models.dnn_noenv.evaluate_dnn_noenv "
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
        "--split-type",
        choices=["temporal", "spatial_block", "spatial_block_kfold", "temporal_narrow_gap"],
        default="temporal",
    )
    parser.add_argument(
        "--n-folds", type=int, default=DEFAULT_K_FOLDS,
        help=f"Number of folds for --split-type spatial_block_kfold (default {DEFAULT_K_FOLDS}). "
             "Ignored for every other split type.",
    )
    parser.add_argument(
        "--fold-index", type=int, default=0,
        help="Which fold to hold out as test, for --split-type spatial_block_kfold (0-indexed, "
             "must be < --n-folds). Ignored for every other split type.",
    )
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--optimizer", choices=["adam", "sgd_momentum"], default=DEFAULT_OPTIMIZER,
        help="adam is the default everywhere so far; sgd_momentum is an A/B-test alternative.",
    )
    parser.add_argument(
        "--run-name", default=None,
        help="Set this whenever --seed isn't the default, so the run doesn't overwrite the "
             "primary dnn_noenv checkpoint (e.g. for a reseed variance check).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=(
            f"Training batch size. Default {BATCH_SIZE}, matching pinn_noenv's default before "
            "2026-07-29 -- exposed here (previously hardcoded) for symmetry with "
            "run_pinn_noenv.py's --batch-size, so a batch-size sweep can vary either model's "
            "batch size independently. See documentation/experiment_log.md's 2026-07-29 entry "
            "for why this matters: pinn_noenv's smaller batch size (128 vs this model's 512) "
            "was found to stall its training entirely, not just a minor confound."
        ),
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE,
        help=f"Adam/SGD starting learning rate. Default {LEARNING_RATE}, never swept before "
             "2026-08-01. See documentation/experiment_log.md's 2026-08-01 entry -- exposed to "
             "test whether a bigger starting rate changes best_val_loss, not just where in "
             "training the best epoch lands.",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.0,
        help="Dropout probability in the network's hidden layers. Default 0.0 (no dropout, "
             "matching every dnn_noenv/pinn_noenv result reported so far) -- exposed 2026-08-01 "
             "as the better-motivated fix for the overfitting-shaped training curves already on "
             "record (train_loss keeps falling while val_loss stalls early), see "
             "documentation/experiment_log.md's 2026-08-01 entry.",
    )
    parser.add_argument(
        "--hidden-layer-sizes", type=str, default=None,
        help="Comma-separated hidden layer sizes, e.g. '64,32'. Default: the original 3x128 "
             "network (unchanged). Exposed 2026-08-02 for the architecture-sensitivity sweep -- "
             "the dropout/learning-rate diagnostic found neither moved best_val_loss, raising "
             "the question of whether fixed capacity is the actual limiting factor. See "
             "documentation/experiment_log.md's 2026-08-02 entry.",
    )
    parser.add_argument(
        "--split-seed", type=int, default=SPLIT_SEED,
        help=f"Seed for spatial_block_split's own block-shuffle (default {SPLIT_SEED}, every "
             "prior result in this repo). Only affects split_type=spatial_block -- exposed "
             "2026-08-02 for the split-seed robustness check flagged in experiment_log.md: every "
             "result up to now used the SAME fixed compartment partition, never varied.",
    )
    args = parser.parse_args()

    if args.split_type == "spatial_block_kfold" and not (0 <= args.fold_index < args.n_folds):
        raise ValueError(f"--fold-index must be in [0, {args.n_folds}), got {args.fold_index}")

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]

    results = {}
    for cohort in cohorts:
        results[cohort] = run_for_cohort(
            cohort, args.split_type, args.max_epochs, args.patience, args.seed, args.optimizer, args.run_name,
            batch_size=args.batch_size, learning_rate=args.learning_rate, dropout_rate=args.dropout_rate,
            hidden_layer_sizes=parse_hidden_layer_sizes(args.hidden_layer_sizes), split_seed=args.split_seed,
            k_folds=args.n_folds, held_out_fold=args.fold_index,
        )

    print("===== Summary: best validation loss reached =====")
    for cohort, best_val_loss in results.items():
        if best_val_loss is None:
            print(f"  {cohort}: FAILED, see outputs/run_logs/")
            continue
        print(f"  {cohort}: best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()

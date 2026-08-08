# Run as: python -m models.dnn_env_terrain.run_dnn_env_terrain --cohort 4survey
#     or: python -m models.dnn_env_terrain.run_dnn_env_terrain --cohort 4survey --max-epochs 5
#     or: python -m models.dnn_env_terrain.run_dnn_env_terrain --cohort 4survey --split-type spatial_block
#
# TRAINS the DNN env_terrain control -- same no-env features as dnn_noenv.py, PLUS a chosen
# terrain/wind feature set (models/common/torch_data.py::ENV_TERRAIN_FEATURE_SETS, --feature-set
# below) concatenated in, all fed to one flat network. Mirrors run_dnn_noenv.py's structure
# exactly; see that file's own comments for the fit-only/evaluate-later split, --max-epochs
# test-run convention, and run_name handling -- not repeated here.

import argparse
import json

import joblib
import torch

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
    DEFAULT_ENV_TERRAIN_FEATURE_SET,
    ENV_TERRAIN_FEATURE_SETS,
    build_tensors,
    build_terrain_tensor,
    encode_thinning_status,
    fit_scalers,
    fit_terrain_scaler,
    load_split_table_with_terrain,
    load_trajectory_pairs,
    print_pre_training_diagnostic,
    select_device,
)
from models.dnn_env_terrain.dnn_env_terrain import (
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

MODEL_NAME = "dnn_env_terrain"
DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 500
DEFAULT_EARLY_STOPPING_PATIENCE = 40
DEFAULT_OPTIMIZER = "adam"


def run_for_cohort(
    cohort, split_type, max_epochs, early_stopping_patience, seed, optimizer_name,
    run_name=None, batch_size=BATCH_SIZE, feature_set_name=DEFAULT_ENV_TERRAIN_FEATURE_SET,
    dropout_rate=0.0, learning_rate=LEARNING_RATE, hidden_layer_sizes=None, split_seed=SPLIT_SEED,
    k_folds=DEFAULT_K_FOLDS, held_out_fold=0,
):
    output_model_name = run_name if run_name else MODEL_NAME
    fold_suffix = f", fold={held_out_fold}/{k_folds}" if split_type == "spatial_block_kfold" else ""
    print(f"===== {cohort} ({output_model_name}, {split_type}{fold_suffix}) — FIT ONLY, no test-set evaluation =====")

    is_test_run = max_epochs < TEST_RUN_MAX_EPOCHS_THRESHOLD
    device = select_device()
    print(f"  Using device: {device}")

    feature_columns = ENV_TERRAIN_FEATURE_SETS[feature_set_name]
    print(f"  Terrain/wind feature set: {feature_set_name} = {feature_columns}")

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
        "feature_set_name": feature_set_name,
        "dropout_rate": dropout_rate,
        "hidden_layer_sizes": hidden_layer_sizes,
    }
    if split_type == "temporal":
        hyperparameters["temporal_split_years"] = TEMPORAL_YEARS[cohort]

    timer = RunTimer().start()
    attempt_id = write_started_marker(
        model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="fit",
        is_test_run=is_test_run, device=str(device), hyperparameters=hyperparameters,
    )

    try:
        # ----- Load and prepare the data -- load_split_table_with_terrain() merges the chosen
        # terrain/wind columns in, on top of load_dnn_noenv's usual no-env table. -----
        split_df = load_split_table_with_terrain(
            cohort, split_type, feature_columns, split_seed=split_seed, k_folds=k_folds, held_out_fold=held_out_fold,
        )
        train_df = split_df[split_df["split"] == "train"]
        val_df = split_df[split_df["split"] == "val"]

        pairs_df = load_trajectory_pairs(cohort, split_df)
        print_pre_training_diagnostic(cohort, split_df, pairs_df)

        scaler_age, scaler_other_features, scaler_height = fit_scalers(train_df)
        scaler_terrain = fit_terrain_scaler(train_df, feature_columns)
        encoded_column_names = encode_thinning_status(train_df).columns.tolist()

        age_train, other_train_noenv, target_train = build_tensors(
            train_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
        )
        terrain_train = build_terrain_tensor(train_df, scaler_terrain, feature_columns, device)
        other_train = torch.cat([other_train_noenv, terrain_train], dim=1)

        age_val, other_val_noenv, target_val = build_tensors(
            val_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
        )
        terrain_val = build_terrain_tensor(val_df, scaler_terrain, feature_columns, device)
        other_val = torch.cat([other_val_noenv, terrain_val], dim=1)

        # ----- Train -----
        n_other_features = other_train.shape[1]
        best_model, final_model_state, history_df = fit(
            age_train, other_train, target_train,
            age_val, other_val, target_val,
            n_other_features, device, seed,
            max_epochs, early_stopping_patience,
            optimizer_name=optimizer_name,
            batch_size=batch_size,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            hidden_layer_sizes=hidden_layer_sizes,
        )
        best_val_loss = float(history_df["val_loss_smoothed"].min())
        final_val_loss = float(history_df["val_loss"].iloc[-1])
        print(
            f"  Trained for {len(history_df)} epochs in {timer.elapsed_seconds():.1f}s. "
            f"best_val_loss={best_val_loss:.6f}  final_val_loss={final_val_loss:.6f}"
        )

        # ----- Save everything needed to evaluate this model LATER -----
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
        joblib.dump(scaler_terrain, preprocessing_dir / "scaler_terrain.joblib")
        with open(preprocessing_dir / "encoded_column_names.json", "w") as f:
            json.dump(encoded_column_names, f, indent=2)
        # Saved explicitly (not just the feature_set NAME in run_metadata.json) so
        # evaluate_dnn_env_terrain.py reads back the EXACT columns used to train this specific
        # checkpoint -- safe even if ENV_TERRAIN_FEATURE_SETS's definitions ever change later.
        with open(preprocessing_dir / "terrain_feature_columns.json", "w") as f:
            json.dump(feature_columns, f, indent=2)

        run_metadata_hyperparameters = {
            **hyperparameters,
            "n_epochs_trained": len(history_df),
            "git_commit": get_git_commit(),
        }
        save_run_metadata(cohort, len(train_df), run_metadata_hyperparameters, output_dir / "run_metadata.json")

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
            f"  Next step: python -m models.dnn_env_terrain.evaluate_dnn_env_terrain "
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
        choices=["temporal", "spatial_block", "spatial_block_kfold", "temporal_narrow_gap", "plot_level"],
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
    parser.add_argument("--optimizer", choices=["adam", "sgd_momentum"], default=DEFAULT_OPTIMIZER)
    parser.add_argument(
        "--run-name", default=None,
        help="Set this whenever --seed/--feature-set/--dropout-rate isn't the default, so the "
             "run doesn't overwrite the primary dnn_env_terrain checkpoint.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--feature-set", choices=list(ENV_TERRAIN_FEATURE_SETS.keys()), default=DEFAULT_ENV_TERRAIN_FEATURE_SET,
        help=f"Named terrain/wind feature set (see ENV_TERRAIN_FEATURE_SETS in torch_data.py). "
             f"Default: {DEFAULT_ENV_TERRAIN_FEATURE_SET}.",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.0,
        help="Dropout probability in the main network's hidden layers. Default 0.0 (no dropout, "
             "matching dnn_noenv's architecture) -- a real hyperparameter to sweep, not a "
             "guessed value, see experiment_log.md.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE,
        help=f"Adam/SGD starting learning rate. Default {LEARNING_RATE}, never swept before "
             "2026-08-01. See documentation/experiment_log.md's 2026-08-01 entry.",
    )
    parser.add_argument(
        "--hidden-layer-sizes", type=str, default=None,
        help="Comma-separated hidden layer sizes, e.g. '64,32'. Default: the original 3x128 "
             "network (unchanged). See documentation/experiment_log.md's 2026-08-02 entry.",
    )
    parser.add_argument(
        "--split-seed", type=int, default=SPLIT_SEED,
        help=f"Seed for spatial_block_split's own block-shuffle (default {SPLIT_SEED}). Only "
             "affects split_type=spatial_block -- see documentation/experiment_log.md's "
             "2026-08-02 split-seed robustness entries.",
    )
    args = parser.parse_args()

    if args.split_type == "spatial_block_kfold" and not (0 <= args.fold_index < args.n_folds):
        raise ValueError(f"--fold-index must be in [0, {args.n_folds}), got {args.fold_index}")

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]

    results = {}
    for cohort in cohorts:
        results[cohort] = run_for_cohort(
            cohort, args.split_type, args.max_epochs, args.patience, args.seed, args.optimizer, args.run_name,
            batch_size=args.batch_size, feature_set_name=args.feature_set, dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate, hidden_layer_sizes=parse_hidden_layer_sizes(args.hidden_layer_sizes),
            split_seed=args.split_seed, k_folds=args.n_folds, held_out_fold=args.fold_index,
        )

    print("===== Summary: best validation loss reached =====")
    for cohort, best_val_loss in results.items():
        if best_val_loss is None:
            print(f"  {cohort}: FAILED, see outputs/run_logs/")
            continue
        print(f"  {cohort}: best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()

# Run as: python -m models.pinn_env_terrain_k.run_pinn_env_terrain_k --cohort 4survey
#     or: python -m models.pinn_env_terrain_k.run_pinn_env_terrain_k --cohort 4survey --max-epochs 5
#     or: python -m models.pinn_env_terrain_k.run_pinn_env_terrain_k --cohort 4survey --split-type spatial_block
#
# TRAINS the CR-PINN with terrain/wind-conditioned y_max AND k (growth-rate parameter) --
# extends models/pinn_env_terrain (which conditions y_max only) with a second sub-network for k,
# per an explicit 2026-08-03 supervisor-prompted extension. See pinn_env_terrain_k.py's own
# top-of-file note for the full reasoning (why k not p, the Socha et al. 2021 lineage this
# departs from, the y_max/k confound risk to check once fit). A new, separate model -- not an
# edit to pinn_env_terrain, and no shared code with models/growth_curve_attribution/ (a
# different, parallel investigation).
#
# --physics-weight/--trajectory-weight default to 1.0 (the untested base case), same reasoning as
# pinn_env_terrain.py's own default.

import argparse
import json
from pathlib import Path

import joblib
import torch

from models.common.run_logging import (
    TEST_RUN_MAX_EPOCHS_THRESHOLD,
    RunTimer,
    format_error,
    write_run_log,
    write_started_marker,
)
from models.common.saving import get_git_commit, load_cr_params, model_output_dir
from models.common.splits import DEFAULT_K_FOLDS, SPLIT_SEED, TEMPORAL_YEARS
from models.common.torch_model import parse_hidden_layer_sizes
from models.common.torch_data import (
    DEFAULT_ENV_TERRAIN_FEATURE_SET,
    ENV_TERRAIN_FEATURE_SETS,
    build_pair_terrain_tensor,
    build_pair_tensors,
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
from models.pinn_env_terrain_k.pinn_env_terrain_k import (
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
MODEL_NAME = "pinn_env_terrain_k"
DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 500
DEFAULT_EARLY_STOPPING_PATIENCE = 40
DEFAULT_OPTIMIZER = "adam"


def run_for_cohort(
    cohort, split_type, max_epochs, early_stopping_patience, seed, optimizer_name,
    physics_weight, trajectory_weight, batch_size, pairs_batch_size, run_name,
    feature_set_name=DEFAULT_ENV_TERRAIN_FEATURE_SET, dropout_rate=0.0, learning_rate=LEARNING_RATE,
    hidden_layer_sizes=None, split_seed=SPLIT_SEED, k_folds=DEFAULT_K_FOLDS, held_out_fold=0,
    freeze_y_max=False,
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
        "physics_weight": physics_weight,
        "trajectory_weight": trajectory_weight,
        "batch_size": batch_size,
        "pairs_batch_size": pairs_batch_size,
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "seed": seed,
        "feature_set_name": feature_set_name,
        "dropout_rate": dropout_rate,
        "hidden_layer_sizes": hidden_layer_sizes,
        "freeze_y_max": freeze_y_max,
    }
    if split_type == "temporal":
        hyperparameters["temporal_split_years"] = TEMPORAL_YEARS[cohort]

    timer = RunTimer().start()
    attempt_id = write_started_marker(
        model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="fit",
        is_test_run=is_test_run, device=str(device), hyperparameters=hyperparameters,
    )

    try:
        cr_held_out_fold = held_out_fold if split_type == "spatial_block_kfold" else None
        cr_params = load_cr_params(cohort, split_type, split_seed=split_seed, held_out_fold=cr_held_out_fold)
        print(f"  Frozen CR anchor (split-matched, {split_type}{fold_suffix}): y_max={cr_params['y_max']:.4f}, k={cr_params['k']:.6f}, p={cr_params['p']:.6f}")

        # ----- Load and prepare the data -----
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

        # Main network's inputs: age + no-env features ONLY -- terrain/wind is NOT concatenated
        # in here, it only reaches the model via the y_max/k sub-networks below.
        age_train, other_train, target_train = build_tensors(
            train_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
        )
        age_val, other_val, target_val = build_tensors(
            val_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
        )
        terrain_train = build_terrain_tensor(train_df, scaler_terrain, feature_columns, device)

        pair_tensors = build_pair_tensors(
            pairs_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
        )
        terrain_pairs = build_pair_terrain_tensor(pairs_df, scaler_terrain, feature_columns, device)

        # ----- Train -----
        n_other_features = other_train.shape[1]
        n_terrain_features = terrain_train.shape[1]
        best_model, final_model_state, history_df = fit(
            age_train, other_train, terrain_train, target_train,
            age_val, other_val, target_val,
            pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
            n_other_features, n_terrain_features, device, seed,
            max_epochs, early_stopping_patience,
            optimizer_name=optimizer_name,
            physics_weight=physics_weight, trajectory_weight=trajectory_weight,
            batch_size=batch_size, pairs_batch_size=pairs_batch_size,
            dropout_rate=dropout_rate, learning_rate=learning_rate, hidden_layer_sizes=hidden_layer_sizes,
            freeze_y_max=freeze_y_max,
        )
        last_row = history_df.iloc[-1]
        best_val_loss = float(history_df["val_loss_smoothed"].min())
        print(
            f"  Trained for {len(history_df)} epochs in {timer.elapsed_seconds():.1f}s. "
            f"best_val_loss={best_val_loss:.6f}  final: data_loss={last_row['data_loss']:.6f}  "
            f"physics_loss={last_row['physics_loss']:.6f}  trajectory_loss={last_row['trajectory_loss']:.6f}"
        )

        # ----- Save everything needed to evaluate this model LATER -----
        # An extra "fold_<i>" path segment under spatial_block_kfold, so a full k-fold sweep's
        # 5 (or n_folds) runs never overwrite each other -- model_output_dir() already accepts
        # arbitrary extra *parts, no change needed there.
        if split_type == "spatial_block_kfold":
            output_dir = model_output_dir(output_model_name, cohort, f"fold_{held_out_fold}", split_type=split_type)
        else:
            output_dir = model_output_dir(output_model_name, cohort, split_type=split_type)
        output_dir.mkdir(parents=True, exist_ok=True)

        history_df.to_csv(output_dir / "training_history.csv", index=False)
        save_checkpoints(
            best_model, final_model_state, n_other_features, n_terrain_features, output_dir / "checkpoints",
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
        with open(preprocessing_dir / "terrain_feature_columns.json", "w") as f:
            json.dump(feature_columns, f, indent=2)

        run_metadata_hyperparameters = {
            **hyperparameters,
            "n_epochs_trained": len(history_df),
            "n_trajectory_pairs": len(pairs_df),
            "frozen_cr_params": cr_params,
            "git_commit": get_git_commit(),
        }
        save_run_metadata(cohort, len(train_df), run_metadata_hyperparameters, output_dir / "run_metadata.json")

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
            f"  Next step: python -m models.pinn_env_terrain_k.evaluate_pinn_env_terrain_k "
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
             "must be < --n-folds). Ignored for every other split type. Requires a matching "
             "fold-specific CR anchor -- run 'python -m models.baselines.run_baselines "
             "--split-type spatial_block_kfold --n-folds <N> --fold-index <i>' first.",
    )
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_EARLY_STOPPING_PATIENCE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--optimizer", choices=["adam", "sgd_momentum"], default=DEFAULT_OPTIMIZER)
    parser.add_argument(
        "--physics-weight", type=float, default=PHYSICS_WEIGHT,
        help=f"Weight on the instantaneous-derivative physics loss term. Default {PHYSICS_WEIGHT} "
             "(untested base case).",
    )
    parser.add_argument(
        "--trajectory-weight", type=float, default=TRAJECTORY_WEIGHT,
        help=f"Weight on the trajectory (finite-difference) physics loss term. Default {TRAJECTORY_WEIGHT}.",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--pairs-batch-size", type=int, default=PAIRS_BATCH_SIZE)
    parser.add_argument(
        "--run-name", default=None,
        help="Only changes where results are saved -- use this whenever --physics-weight/"
             "--trajectory-weight/--feature-set/--dropout-rate differ from the defaults.",
    )
    parser.add_argument(
        "--feature-set", choices=list(ENV_TERRAIN_FEATURE_SETS.keys()), default=DEFAULT_ENV_TERRAIN_FEATURE_SET,
        help=f"Named terrain/wind feature set, shared by both the y_max and k sub-networks (see "
             f"ENV_TERRAIN_FEATURE_SETS in torch_data.py). Default: {DEFAULT_ENV_TERRAIN_FEATURE_SET}.",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.0,
        help="Dropout probability in the main network's and both sub-networks' hidden layers. "
             "Default 0.0 (no dropout, matching pinn_env_terrain's architecture).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=LEARNING_RATE,
        help=f"Adam/SGD starting learning rate. Default {LEARNING_RATE}.",
    )
    parser.add_argument(
        "--hidden-layer-sizes", type=str, default=None,
        help="Comma-separated hidden layer sizes for the MAIN network only, e.g. '64,32' -- "
             "does not resize either sub-network. Default: the original 3x128 main network.",
    )
    parser.add_argument(
        "--freeze-y-max", action="store_true",
        help="Council's freeze-one-vary-other ablation (2026-08-04): pins y_max to the plain "
             "global CR constant (y_max_subnetwork gets no gradient and stays at its random "
             "init -- ignore 'learned_y_max' in this run's predictions.csv, it's untrained "
             "noise) so only k is a free per-plot parameter. Tests whether a wide, implausible "
             "learned-k range persists with a single degree of freedom per plot -- use --run-name "
             "to keep this separate from the full two-parameter model's output.",
    )
    parser.add_argument(
        "--split-seed", type=int, default=SPLIT_SEED,
        help=f"Seed for spatial_block_split's own block-shuffle (default {SPLIT_SEED}). "
             "load_cr_params() reads the matching '_splitseed<N>'-suffixed CR anchor for a "
             "non-default value -- run 'python -m models.baselines.run_baselines --split-type "
             "<split_type> --split-seed <N>' first to produce it.",
    )
    args = parser.parse_args()

    if args.split_type == "spatial_block_kfold" and not (0 <= args.fold_index < args.n_folds):
        raise ValueError(f"--fold-index must be in [0, {args.n_folds}), got {args.fold_index}")

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]

    results = {}
    for cohort in cohorts:
        results[cohort] = run_for_cohort(
            cohort, args.split_type, args.max_epochs, args.patience, args.seed, args.optimizer,
            args.physics_weight, args.trajectory_weight, args.batch_size, args.pairs_batch_size,
            args.run_name, feature_set_name=args.feature_set, dropout_rate=args.dropout_rate,
            learning_rate=args.learning_rate, hidden_layer_sizes=parse_hidden_layer_sizes(args.hidden_layer_sizes),
            split_seed=args.split_seed, k_folds=args.n_folds, held_out_fold=args.fold_index,
            freeze_y_max=args.freeze_y_max,
        )

    print("===== Summary: best validation loss reached =====")
    for cohort, best_val_loss in results.items():
        if best_val_loss is None:
            print(f"  {cohort}: FAILED, see outputs/run_logs/")
            continue
        print(f"  {cohort}: best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()

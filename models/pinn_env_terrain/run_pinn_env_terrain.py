# Run as: python -m models.pinn_env_terrain.run_pinn_env_terrain --cohort 4survey
#     or: python -m models.pinn_env_terrain.run_pinn_env_terrain --cohort 4survey --max-epochs 5
#     or: python -m models.pinn_env_terrain.run_pinn_env_terrain --cohort 4survey --split-type spatial_block
#
# TRAINS the CR-PINN with terrain/wind-conditioned y_max -- same frozen CR anchor (split-matched
# "cr_matched", see load_cr_params below) as pinn_noenv.py, but y_max in the physics/trajectory
# loss is now global_y_max + a per-plot adjustment from the terrain/wind sub-network, not a
# single constant.
# Mirrors run_pinn_noenv.py's structure closely -- see that file's own comments for the
# fit-only/evaluate-later split, --max-epochs test-run convention, and run_name handling.
#
# --physics-weight/--trajectory-weight default to 1.0 (the untested base case), NOT the no-env
# pipeline's Stage 3 winning weight -- see pinn_env_terrain.py's own top-of-file note for why
# that decision does not carry over here. A dedicated env_terrain physics-weight sweep is still
# open work.

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
from models.common.saving import get_git_commit, model_output_dir
from models.common.splits import TEMPORAL_YEARS
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
from models.pinn_env_terrain.pinn_env_terrain import (
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
MODEL_NAME = "pinn_env_terrain"
DEFAULT_SEED = 42
DEFAULT_MAX_EPOCHS = 500
DEFAULT_EARLY_STOPPING_PATIENCE = 40
DEFAULT_OPTIMIZER = "adam"


def load_cr_params(cohort, split_type):
    # Split-MATCHED CR anchor (2026-08-01 fix, switched from the pooled/"cr_pooled" version
    # both PINN models used before) -- fit using ONLY this split_type's own train-assigned
    # plots (models/baselines/run_baselines.py's cr_train_df), not a random 60% plot_level
    # split unrelated to spatial_block. The old pooled version was confirmed to leak: its
    # random 60% training plots inevitably overlap with spatial_block's own test plots, since
    # the two splits were never coordinated (verified directly: 139,472 rows = exactly 60% of
    # the 232,448 filtered 4survey rows, i.e. plot_level_split's train share, not spatial_block's).
    # This file already exists on disk for every split_type PINN uses (spatial_block, temporal,
    # temporal_narrow_gap) -- built as a side effect of the CR baseline's own per-split fit, no
    # new fitting needed here. y_max here becomes the ANCHOR the terrain sub-network's
    # adjustment is added to, not the final value used directly.
    params_path = PROJECT_ROOT / "outputs" / split_type / "chapman_richards" / cohort / "params.json"
    with open(params_path) as f:
        params = json.load(f)
    return {"y_max": params["y_max"], "k": params["k"], "p": params["p"]}


def run_for_cohort(
    cohort, split_type, max_epochs, early_stopping_patience, seed, optimizer_name,
    physics_weight, trajectory_weight, batch_size, pairs_batch_size, run_name,
    feature_set_name=DEFAULT_ENV_TERRAIN_FEATURE_SET, dropout_rate=0.0,
):
    output_model_name = run_name if run_name else MODEL_NAME
    print(f"===== {cohort} ({output_model_name}, {split_type}) — FIT ONLY, no test-set evaluation =====")

    is_test_run = max_epochs < TEST_RUN_MAX_EPOCHS_THRESHOLD
    device = select_device()
    print(f"  Using device: {device}")

    feature_columns = ENV_TERRAIN_FEATURE_SETS[feature_set_name]
    print(f"  Terrain/wind feature set: {feature_set_name} = {feature_columns}")

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
        "feature_set_name": feature_set_name,
        "dropout_rate": dropout_rate,
    }
    if split_type == "temporal":
        hyperparameters["temporal_split_years"] = TEMPORAL_YEARS[cohort]

    timer = RunTimer().start()
    attempt_id = write_started_marker(
        model_name=output_model_name, cohort=cohort, split_type=split_type, run_phase="fit",
        is_test_run=is_test_run, device=str(device), hyperparameters=hyperparameters,
    )

    try:
        cr_params = load_cr_params(cohort, split_type)
        print(f"  Frozen CR anchor (split-matched, {split_type}): y_max={cr_params['y_max']:.4f}, k={cr_params['k']:.6f}, p={cr_params['p']:.6f}")

        # ----- Load and prepare the data -----
        split_df = load_split_table_with_terrain(cohort, split_type, feature_columns)
        train_df = split_df[split_df["split"] == "train"]
        val_df = split_df[split_df["split"] == "val"]

        pairs_df = load_trajectory_pairs(cohort, split_df)
        print_pre_training_diagnostic(cohort, split_df, pairs_df)

        scaler_age, scaler_other_features, scaler_height = fit_scalers(train_df)
        scaler_terrain = fit_terrain_scaler(train_df, feature_columns)
        encoded_column_names = encode_thinning_status(train_df).columns.tolist()

        # Main network's inputs: age + no-env features ONLY -- terrain/wind is NOT concatenated
        # in here (unlike dnn_env_terrain.py), it only reaches the model via the y_max
        # sub-network below. See pinn_env_terrain.py's own top-of-file note for why.
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
            dropout_rate=dropout_rate,
        )
        last_row = history_df.iloc[-1]
        best_val_loss = float(history_df["val_loss_smoothed"].min())
        print(
            f"  Trained for {len(history_df)} epochs in {timer.elapsed_seconds():.1f}s. "
            f"best_val_loss={best_val_loss:.6f}  final: data_loss={last_row['data_loss']:.6f}  "
            f"physics_loss={last_row['physics_loss']:.6f}  trajectory_loss={last_row['trajectory_loss']:.6f}"
        )

        # ----- Save everything needed to evaluate this model LATER -----
        output_dir = model_output_dir(output_model_name, cohort, split_type=split_type)
        output_dir.mkdir(parents=True, exist_ok=True)

        history_df.to_csv(output_dir / "training_history.csv", index=False)
        save_checkpoints(best_model, final_model_state, n_other_features, n_terrain_features, output_dir / "checkpoints")

        preprocessing_dir = output_dir / "preprocessing"
        preprocessing_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler_age, preprocessing_dir / "scaler_age.joblib")
        joblib.dump(scaler_other_features, preprocessing_dir / "scaler_other_features.joblib")
        joblib.dump(scaler_height, preprocessing_dir / "scaler_height.joblib")
        joblib.dump(scaler_terrain, preprocessing_dir / "scaler_terrain.joblib")
        with open(preprocessing_dir / "encoded_column_names.json", "w") as f:
            json.dump(encoded_column_names, f, indent=2)
        # Saved explicitly (not just the feature_set NAME in run_metadata.json) so
        # evaluate_pinn_env_terrain.py reads back the EXACT columns used to train this specific
        # checkpoint -- safe even if ENV_TERRAIN_FEATURE_SETS's definitions ever change later.
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
            f"  Next step: python -m models.pinn_env_terrain.evaluate_pinn_env_terrain "
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
    parser.add_argument("--optimizer", choices=["adam", "sgd_momentum"], default=DEFAULT_OPTIMIZER)
    parser.add_argument(
        "--physics-weight", type=float, default=PHYSICS_WEIGHT,
        help=f"Weight on the instantaneous-derivative physics loss term. Default {PHYSICS_WEIGHT} "
             "(untested base case) -- the no-env pipeline's low-weight finding is NOT assumed to "
             "carry over here, see pinn_env_terrain.py's own note.",
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
        help=f"Named terrain/wind feature set for the y_max sub-network (see "
             f"ENV_TERRAIN_FEATURE_SETS in torch_data.py). Default: {DEFAULT_ENV_TERRAIN_FEATURE_SET}.",
    )
    parser.add_argument(
        "--dropout-rate", type=float, default=0.0,
        help="Dropout probability in both the main network's and the y_max sub-network's hidden "
             "layers. Default 0.0 (no dropout, matching pinn_noenv's architecture) -- a real "
             "hyperparameter to sweep, not a guessed value.",
    )
    args = parser.parse_args()

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]

    results = {}
    for cohort in cohorts:
        results[cohort] = run_for_cohort(
            cohort, args.split_type, args.max_epochs, args.patience, args.seed, args.optimizer,
            args.physics_weight, args.trajectory_weight, args.batch_size, args.pairs_batch_size,
            args.run_name, feature_set_name=args.feature_set, dropout_rate=args.dropout_rate,
        )

    print("===== Summary: best validation loss reached =====")
    for cohort, best_val_loss in results.items():
        if best_val_loss is None:
            print(f"  {cohort}: FAILED, see outputs/run_logs/")
            continue
        print(f"  {cohort}: best_val_loss={best_val_loss:.6f}")


if __name__ == "__main__":
    main()

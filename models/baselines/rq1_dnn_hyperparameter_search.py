# Run as: python -m models.baselines.rq1_dnn_hyperparameter_search
#
# Tests the last remaining candidate reason for "XGBoost beats the DNN on 4survey": a narrower
# training-hyperparameter gap. XGBoost's config was grid-searched over 27 combinations
# (n_estimators, max_depth, learning_rate; see TEMP_rq1_xgb_hyperparameter_search_2026-08-16.tex).
# The DNN has never had its own training hyperparameters swept. Only architecture SIZE was
# swept (TEMP_rq1_architecture_sweep_results_2026-08-13.tex, a null result). Learning rate and
# weight decay were fixed constants throughout this project (dnn_env_terrain.py's own top-of-file
# note: "Fixed hyperparameters. Identical to dnn_noenv.py's, on purpose").
#
# Method mirrors rq1_xgb_hyperparameter_search.py's own convention exactly: grid search selected
# on VALIDATION R2 only, never test R2 (avoids tuning-on-test leakage); test metrics reported only
# for the winning config, after selection is already decided.
#
# Grid: learning_rate in {0.0001 (project default), 0.0003, 0.001} x weight_decay in
# {1e-5 (project default), 1e-4, 1e-3} = 9 configs, on the same single spatial_block split used by
# every other RQ1 mechanism check this session. Batch size and the LR-scheduler's own
# factor/patience are left at their project defaults. Varying every training knob at once would
# make a 9-point coarse screen uninterpretable; learning rate and weight decay are the two most
# standard, highest-leverage knobs to check first.

import torch

from models.common.metrics import compute_metrics
from models.common.splits import SPLIT_SEED
from models.common.torch_data import (
    ENV_TERRAIN_FEATURE_SETS,
    TARGET_COLUMN,
    build_terrain_tensor,
    build_tensors,
    encode_thinning_status,
    fit_scalers,
    fit_terrain_scaler,
    load_split_table_with_terrain,
    select_device,
)
from models.dnn_env_terrain import dnn_env_terrain as dnn_module
from models.dnn_env_terrain.dnn_env_terrain import BATCH_SIZE, fit as fit_dnn, predict as predict_dnn

COHORT = "4survey"
FEATURE_SET_NAME = "nested_set3_gated_terrain_wind_vif"
TRAINING_SEED = 42
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 40

LEARNING_RATES = [0.0001, 0.0003, 0.001]
WEIGHT_DECAYS = [1e-5, 1e-4, 1e-3]

# Already-established single-split reference points (same split, same feature set, this session's
# other checks). Printed for comparison, not recomputed here.
XGBOOST_SINGLE_SPLIT_TEST_R2 = 0.6402
DEFAULT_DNN_SINGLE_SPLIT_TEST_R2 = 0.6166  # learning_rate=0.0001, weight_decay=1e-5 (the default)


def fit_and_evaluate(train_df, val_df, test_df, feature_columns, device, learning_rate, weight_decay):
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

    # WEIGHT_DECAY is read as a module-level constant inside build_optimizer(), not exposed as a
    # fit() parameter. Temporarily override the module constant for this one config, then
    # restore it, rather than duplicating dnn_env_terrain.py's training loop just to vary one knob.
    original_weight_decay = dnn_module.WEIGHT_DECAY
    dnn_module.WEIGHT_DECAY = weight_decay
    try:
        n_other_features = other_train.shape[1]
        best_model, _final_model_state, history_df = fit_dnn(
            age_train, other_train, target_train,
            age_val, other_val, target_val,
            n_other_features, device, TRAINING_SEED,
            MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
            optimizer_name="adam", batch_size=BATCH_SIZE,
            dropout_rate=0.0, learning_rate=learning_rate, hidden_layer_sizes=None,
        )
    finally:
        dnn_module.WEIGHT_DECAY = original_weight_decay

    predicted_val_scaled = predict_dnn(best_model, age_val, other_val)
    predicted_val = scaler_height.inverse_transform(predicted_val_scaled.cpu().numpy()).flatten()
    val_r2 = compute_metrics(val_df[TARGET_COLUMN].to_numpy(), predicted_val)["r2"]

    age_test, other_test_noenv, target_test = build_tensors(
        test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
    )
    terrain_test = build_terrain_tensor(test_df, scaler_terrain, feature_columns, device)
    other_test = torch.cat([other_test_noenv, terrain_test], dim=1)
    predicted_test_scaled = predict_dnn(best_model, age_test, other_test)
    predicted_test = scaler_height.inverse_transform(predicted_test_scaled.cpu().numpy()).flatten()
    test_metrics = compute_metrics(test_df[TARGET_COLUMN].to_numpy(), predicted_test)

    return val_r2, test_metrics, len(history_df)


def main():
    device = select_device()
    feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]
    full_df = load_split_table_with_terrain(COHORT, "spatial_block", feature_columns, split_seed=SPLIT_SEED)
    train_df = full_df[full_df["split"] == "train"]
    val_df = full_df[full_df["split"] == "val"]
    test_df = full_df[full_df["split"] == "test"]
    print(f"train/val/test rows: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    results = []
    for learning_rate in LEARNING_RATES:
        for weight_decay in WEIGHT_DECAYS:
            val_r2, test_metrics, n_epochs = fit_and_evaluate(
                train_df, val_df, test_df, feature_columns, device, learning_rate, weight_decay
            )
            print(
                f"lr={learning_rate}  weight_decay={weight_decay}  epochs={n_epochs}  "
                f"val_R2={val_r2:.4f}  test_R2={test_metrics['r2']:.4f}  "
                f"test_RMSE={test_metrics['rmse']:.4f}  test_MAE={test_metrics['mae']:.4f}"
            )
            results.append({
                "learning_rate": learning_rate, "weight_decay": weight_decay,
                "n_epochs": n_epochs, "val_r2": val_r2, **test_metrics,
            })

    best = max(results, key=lambda row: row["val_r2"])
    print("\n=== Winning config (selected on validation R2 only) ===")
    print(
        f"lr={best['learning_rate']}  weight_decay={best['weight_decay']}  "
        f"val_R2={best['val_r2']:.4f}  test_R2={best['r2']:.4f}  "
        f"test_RMSE={best['rmse']:.4f}  test_MAE={best['mae']:.4f}"
    )
    print(f"\nFor comparison (same single spatial_block split, this session's other checks):")
    print(f"  Default DNN (lr=0.0001, weight_decay=1e-5) test R2: {DEFAULT_DNN_SINGLE_SPLIT_TEST_R2:.4f}")
    print(f"  XGBoost (winning config)                  test R2: {XGBOOST_SINGLE_SPLIT_TEST_R2:.4f}")
    print(f"  Winning DNN config                        test R2: {best['r2']:.4f}")
    print(f"  Gap closed: {best['r2'] - DEFAULT_DNN_SINGLE_SPLIT_TEST_R2:+.4f} "
          f"(of {XGBOOST_SINGLE_SPLIT_TEST_R2 - DEFAULT_DNN_SINGLE_SPLIT_TEST_R2:.4f} total gap to XGBoost)")


if __name__ == "__main__":
    main()

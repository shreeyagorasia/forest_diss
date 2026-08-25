# Run as: python -m models.baselines.rq1_pinn_hyperparameter_search
#
# Same check as rq1_dnn_hyperparameter_search.py, applied to PINN and PINN_k: their training
# hyperparameters (learning_rate=0.0001, weight_decay=1e-5) are the exact same fixed constants as
# the DNN's, never swept. Only architecture SIZE was swept for these two models
# (TEMP_rq1_architecture_sweep_results_2026-08-13.tex, a null result). If the DNN's own gap to
# XGBoost was mostly a training-hyperparameter artefact (confirmed,
# TEMP_rq1_dnn_hyperparameter_search_2026-08-19.tex: 86% of the gap closed), the same may be true
# for PINN/PINN_k's own gap to the DNN and to XGBoost.
#
# Grid: same as the DNN check. Learning_rate in {0.0001 default, 0.0003, 0.001} x weight_decay
# in {1e-5 default, 1e-4, 1e-3} = 9 configs, single 4survey spatial_block split, physics_weight=
# trajectory_weight=1.0 (the standard, already-reported main-sweep config. NOT the w=0
# no-physics-loss ablation, a different, already-answered question). Selected on validation R2
# only, same convention as every other hyperparameter search this session.

import importlib

import torch

from models.common.metrics import compute_metrics
from models.common.saving import load_cr_params
from models.common.splits import SPLIT_SEED
from models.common.torch_data import (
    ENV_TERRAIN_FEATURE_SETS,
    TARGET_COLUMN,
    build_pair_terrain_tensor,
    build_pair_tensors,
    build_tensors,
    build_terrain_tensor,
    encode_thinning_status,
    fit_scalers,
    fit_terrain_scaler,
    load_split_table_with_terrain,
    load_trajectory_pairs,
    select_device,
)

COHORT = "4survey"
SPLIT_TYPE = "spatial_block"
FEATURE_SET_NAME = "nested_set3_gated_terrain_wind_vif"
TRAINING_SEED = 42
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 40
PHYSICS_WEIGHT = 1.0
TRAJECTORY_WEIGHT = 1.0

LEARNING_RATES = [0.0001, 0.0003, 0.001]
WEIGHT_DECAYS = [1e-5, 1e-4, 1e-3]

# Reference points: default-hyperparameter test R2 for each model, single spatial_block split,
# 4survey (from the existing kfold-pooled numbers' single-split analogues. Recomputed fresh
# below instead of assumed, since no single-split PINN/PINN_k number for THIS exact split existed
# yet before this check).
MODELS = {
    "pinn_env_terrain": "models.pinn_env_terrain.pinn_env_terrain",
    "pinn_env_terrain_k": "models.pinn_env_terrain_k.pinn_env_terrain_k",
}


def fit_and_evaluate(pinn_module, train_df, val_df, test_df, feature_columns, cr_params, device,
                      learning_rate, weight_decay):
    scaler_age, scaler_other_features, scaler_height = fit_scalers(train_df)
    scaler_terrain = fit_terrain_scaler(train_df, feature_columns)
    encoded_column_names = encode_thinning_status(train_df).columns.tolist()

    age_train, other_train, target_train = build_tensors(
        train_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
    )
    age_val, other_val, target_val = build_tensors(
        val_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
    )
    terrain_train = build_terrain_tensor(train_df, scaler_terrain, feature_columns, device)

    pairs_df = load_trajectory_pairs(COHORT, train_df.assign(split="train"))
    pair_tensors = build_pair_tensors(pairs_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
    terrain_pairs = build_pair_terrain_tensor(pairs_df, scaler_terrain, feature_columns, device, COHORT)

    original_weight_decay = pinn_module.WEIGHT_DECAY
    pinn_module.WEIGHT_DECAY = weight_decay
    try:
        n_other_features = other_train.shape[1]
        n_terrain_features = terrain_train.shape[1]
        best_model, _final_model_state, history_df = pinn_module.fit(
            age_train, other_train, terrain_train, target_train,
            age_val, other_val, target_val,
            pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
            n_other_features, n_terrain_features, device, TRAINING_SEED,
            MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
            optimizer_name="adam",
            physics_weight=PHYSICS_WEIGHT, trajectory_weight=TRAJECTORY_WEIGHT,
            batch_size=pinn_module.BATCH_SIZE, pairs_batch_size=pinn_module.PAIRS_BATCH_SIZE,
            dropout_rate=0.0, learning_rate=learning_rate, hidden_layer_sizes=None,
        )
    finally:
        pinn_module.WEIGHT_DECAY = original_weight_decay

    predicted_val_scaled = pinn_module.predict(best_model, age_val, other_val)
    predicted_val = scaler_height.inverse_transform(predicted_val_scaled.cpu().numpy()).flatten()
    val_r2 = compute_metrics(val_df[TARGET_COLUMN].to_numpy(), predicted_val)["r2"]

    age_test, other_test, target_test = build_tensors(
        test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
    )
    predicted_test_scaled = pinn_module.predict(best_model, age_test, other_test)
    predicted_test = scaler_height.inverse_transform(predicted_test_scaled.cpu().numpy()).flatten()
    test_metrics = compute_metrics(test_df[TARGET_COLUMN].to_numpy(), predicted_test)

    return val_r2, test_metrics, len(history_df)


def run_model(model_key, module_path, feature_columns, cr_params, train_df, val_df, test_df, device):
    print(f"\n===== {model_key} =====")
    pinn_module = importlib.import_module(module_path)

    results = []
    for learning_rate in LEARNING_RATES:
        for weight_decay in WEIGHT_DECAYS:
            val_r2, test_metrics, n_epochs = fit_and_evaluate(
                pinn_module, train_df, val_df, test_df, feature_columns, cr_params, device,
                learning_rate, weight_decay,
            )
            print(
                f"  lr={learning_rate}  weight_decay={weight_decay}  epochs={n_epochs}  "
                f"val_R2={val_r2:.4f}  test_R2={test_metrics['r2']:.4f}  "
                f"test_RMSE={test_metrics['rmse']:.4f}  test_MAE={test_metrics['mae']:.4f}"
            )
            results.append({
                "learning_rate": learning_rate, "weight_decay": weight_decay,
                "n_epochs": n_epochs, "val_r2": val_r2, **test_metrics,
            })

    default_row = next(r for r in results if r["learning_rate"] == 0.0001 and r["weight_decay"] == 1e-5)
    best = max(results, key=lambda row: row["val_r2"])
    print(f"\n  Default (lr=0.0001, weight_decay=1e-5) test R2: {default_row['r2']:.4f}")
    print(f"  Winning config (by validation R2): lr={best['learning_rate']}  weight_decay={best['weight_decay']}")
    print(f"  Winning config test R2: {best['r2']:.4f}  (change from default: {best['r2'] - default_row['r2']:+.4f})")
    return default_row, best


def main():
    device = select_device()
    feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]
    cr_params = load_cr_params(COHORT, SPLIT_TYPE, split_seed=SPLIT_SEED)
    print(f"Frozen CR anchor: y_max={cr_params['y_max']:.4f}, k={cr_params['k']:.6f}, p={cr_params['p']:.6f}")

    full_df = load_split_table_with_terrain(COHORT, SPLIT_TYPE, feature_columns, split_seed=SPLIT_SEED)
    train_df = full_df[full_df["split"] == "train"]
    val_df = full_df[full_df["split"] == "val"]
    test_df = full_df[full_df["split"] == "test"]
    print(f"train/val/test rows: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    for model_key, module_path in MODELS.items():
        run_model(model_key, module_path, feature_columns, cr_params, train_df, val_df, test_df, device)


if __name__ == "__main__":
    main()

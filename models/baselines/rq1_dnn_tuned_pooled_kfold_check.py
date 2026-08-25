# Run as: python -m models.baselines.rq1_dnn_tuned_pooled_kfold_check
#
# Confirms whether the tuned-DNN gap-closing effect found on a single spatial_block split
# (TEMP_rq1_dnn_hyperparameter_search_2026-08-19.tex: learning_rate=0.001, weight_decay=1e-3
# closes 86% of the XGBoost-DNN gap on 4survey) survives pooling across the full 5-fold
# spatial_block_kfold split. The split Table 1's headline numbers actually use. A single split's
# validation set could just be an easy one; pooling across all 5 held-out folds is the more
# reliable test.
#
# Fits the SAME winning config on all 5 folds, both cohorts, pools predictions across folds (the
# same "pooled R2 over the whole population" convention used throughout this project), and
# compares against the existing Table 1 pooled numbers for DNN and XGBoost.

import numpy as np
import torch

from models.common.metrics import compute_metrics
from models.common.splits import DEFAULT_K_FOLDS, SPLIT_SEED
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

FEATURE_SET_NAME = "nested_set3_gated_terrain_wind_vif"
TRAINING_SEED = 42
MAX_EPOCHS = 500
EARLY_STOPPING_PATIENCE = 40
# Winning config from TEMP_rq1_dnn_hyperparameter_search_2026-08-19.tex.
TUNED_LEARNING_RATE = 0.001
TUNED_WEIGHT_DECAY = 1e-3

# Existing Table 1 pooled 5-fold reference points (default-hyperparameter DNN, tuned XGBoost).
TABLE1_DNN_R2 = {"4survey": 0.655, "6survey": 0.684}
TABLE1_XGB_R2 = {"4survey": 0.674, "6survey": 0.722}


def fit_one_fold(train_df, val_df, test_df, feature_columns, device):
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

    original_weight_decay = dnn_module.WEIGHT_DECAY
    dnn_module.WEIGHT_DECAY = TUNED_WEIGHT_DECAY
    try:
        n_other_features = other_train.shape[1]
        best_model, _final_model_state, history_df = fit_dnn(
            age_train, other_train, target_train,
            age_val, other_val, target_val,
            n_other_features, device, TRAINING_SEED,
            MAX_EPOCHS, EARLY_STOPPING_PATIENCE,
            optimizer_name="adam", batch_size=BATCH_SIZE,
            dropout_rate=0.0, learning_rate=TUNED_LEARNING_RATE, hidden_layer_sizes=None,
        )
    finally:
        dnn_module.WEIGHT_DECAY = original_weight_decay

    age_test, other_test_noenv, target_test = build_tensors(
        test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
    )
    terrain_test = build_terrain_tensor(test_df, scaler_terrain, feature_columns, device)
    other_test = torch.cat([other_test_noenv, terrain_test], dim=1)
    predicted_scaled = predict_dnn(best_model, age_test, other_test)
    predicted = scaler_height.inverse_transform(predicted_scaled.cpu().numpy()).flatten()
    observed = test_df[TARGET_COLUMN].to_numpy()

    return observed, predicted, len(history_df)


def run_cohort(cohort, feature_columns, device):
    print(f"\n===== {cohort} =====")
    all_observed = []
    all_predicted = []
    per_fold_r2 = []

    for held_out_fold in range(DEFAULT_K_FOLDS):
        full_df = load_split_table_with_terrain(
            cohort, "spatial_block_kfold", feature_columns, split_seed=SPLIT_SEED,
            k_folds=DEFAULT_K_FOLDS, held_out_fold=held_out_fold,
        )
        train_df = full_df[full_df["split"] == "train"]
        val_df = full_df[full_df["split"] == "val"]
        test_df = full_df[full_df["split"] == "test"]

        observed, predicted, n_epochs = fit_one_fold(train_df, val_df, test_df, feature_columns, device)
        fold_r2 = compute_metrics(observed, predicted)["r2"]
        per_fold_r2.append(fold_r2)
        all_observed.append(observed)
        all_predicted.append(predicted)
        print(f"  fold {held_out_fold}: n_test={len(test_df)}  epochs={n_epochs}  fold_R2={fold_r2:.4f}")

    pooled_observed = np.concatenate(all_observed)
    pooled_predicted = np.concatenate(all_predicted)
    pooled_metrics = compute_metrics(pooled_observed, pooled_predicted)

    print(f"  Per-fold R2: mean={np.mean(per_fold_r2):.4f}  SD={np.std(per_fold_r2):.4f}")
    print(f"  Pooled R2={pooled_metrics['r2']:.4f}  RMSE={pooled_metrics['rmse']:.4f}  MAE={pooled_metrics['mae']:.4f}")
    print(f"  Table 1 (default-hyperparameter DNN) pooled R2: {TABLE1_DNN_R2[cohort]:.4f}")
    print(f"  Table 1 (tuned XGBoost) pooled R2:               {TABLE1_XGB_R2[cohort]:.4f}")
    gap_before = TABLE1_XGB_R2[cohort] - TABLE1_DNN_R2[cohort]
    gap_after = TABLE1_XGB_R2[cohort] - pooled_metrics["r2"]
    print(f"  Gap to XGBoost: {gap_before:.4f} (default DNN) -> {gap_after:.4f} (tuned DNN)")

    return pooled_metrics


def main():
    device = select_device()
    feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]
    for cohort in ["4survey", "6survey"]:
        run_cohort(cohort, feature_columns, device)


if __name__ == "__main__":
    main()

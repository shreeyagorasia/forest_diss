# Run as: python -m models.baselines.rq1_compartment_subsample_check
#
# Tests one of the possible reasons offered in the results chapter for "XGBoost beats the DNN":
# the limited number of spatially independent compartments each fold's training set has to
# generalise from. If that mechanism is real, XGBoost's advantage over the DNN should be larger
# (not smaller) when fewer independent compartments are available.
#
# Method: take 4survey (296 compartments, plenty to subsample from), fit both models on the FULL
# compartment set (a single spatial_block split, matching this project's "coarse screen"
# convention for exploratory checks -- see the architecture-sweep TEMP note), then repeat on
# random subsamples of just 48 compartments -- the exact compartment count 6survey actually has --
# 3 different random seeds for which 48 compartments get kept, everything else held fixed
# (same XGBoost config, same DNN architecture/hyperparameters, same block-shuffle seed for the
# train/val/test assignment itself). Only the compartment count changes between conditions.
#
# Both models are fit on the EXACT SAME train/val/test split within each condition (built once,
# reused for both), for a fair paired comparison -- same principle as
# rq1_xgb_vs_dnn_paired_folds.py's paired-fold check, applied here across compartment-count
# conditions instead of across folds.

import numpy as np
import torch
import xgboost as xgb

from models.common.metrics import compute_metrics
from models.common.splits import SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, SPLIT_SEED, spatial_block_split
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
from models.dnn_env_terrain.dnn_env_terrain import (
    BATCH_SIZE,
    LEARNING_RATE,
    fit as fit_dnn,
    predict as predict_dnn,
)
from models.xgb_baseline.xgb_baseline import FEATURE_COLUMNS, prepare_features

COHORT = "4survey"
FEATURE_SET_NAME = "nested_set3_gated_terrain_wind_vif"
TARGET_N_COMPARTMENTS = 48  # 6survey's real compartment count
SUBSAMPLE_SEEDS = [101, 102, 103]
DNN_TRAINING_SEED = 42
DNN_MAX_EPOCHS = 500
DNN_EARLY_STOPPING_PATIENCE = 40
# Winning 4survey config from TEMP_rq1_xgb_hyperparameter_search_2026-08-16.tex -- held fixed
# across every condition here so only compartment count varies, not XGBoost's own tuning.
XGB_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.02, random_state=42, n_jobs=1)


def build_subsampled_split(full_df, n_compartments, subsample_seed):
    # None means "keep every compartment" -- used for the full-296 baseline condition.
    if n_compartments is None:
        working_df = full_df
    else:
        all_compartments = full_df[SPATIAL_BLOCK_COL].unique()
        rng = np.random.default_rng(subsample_seed)
        chosen_compartments = rng.choice(all_compartments, size=n_compartments, replace=False)
        working_df = full_df[full_df[SPATIAL_BLOCK_COL].isin(chosen_compartments)].copy()

    # Re-run the block-shuffle train/val/test assignment fresh on whichever compartments are
    # actually present here -- reusing the full-296 split's labels on a filtered-down set would
    # not hit the intended row-fraction targets. split_seed is always the project default
    # (SPLIT_SEED=42), so the only thing varying between conditions is which compartments exist.
    working_df["split"] = spatial_block_split(
        working_df, block_col=SPATIAL_BLOCK_COL, buffer_distance=SPATIAL_BUFFER_METRES, seed=SPLIT_SEED,
    )
    return working_df


def fit_and_score_dnn(train_df, val_df, test_df, feature_columns, device):
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

    n_other_features = other_train.shape[1]
    best_model, _final_model_state, history_df = fit_dnn(
        age_train, other_train, target_train,
        age_val, other_val, target_val,
        n_other_features, device, DNN_TRAINING_SEED,
        DNN_MAX_EPOCHS, DNN_EARLY_STOPPING_PATIENCE,
        optimizer_name="adam", batch_size=BATCH_SIZE,
        dropout_rate=0.0, learning_rate=LEARNING_RATE, hidden_layer_sizes=None,
    )

    age_test, other_test_noenv, target_test = build_tensors(
        test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
    )
    terrain_test = build_terrain_tensor(test_df, scaler_terrain, feature_columns, device)
    other_test = torch.cat([other_test_noenv, terrain_test], dim=1)

    predicted_height_test_scaled = predict_dnn(best_model, age_test, other_test)
    predicted_height_test = scaler_height.inverse_transform(
        predicted_height_test_scaled.cpu().numpy()
    ).flatten()
    observed_height_test = test_df[TARGET_COLUMN].values

    metrics = compute_metrics(observed_height_test, predicted_height_test)
    return metrics["r2"], len(history_df)


def fit_and_score_xgb(train_df, test_df, feature_columns):
    full_columns = FEATURE_COLUMNS + list(feature_columns)
    features_train = prepare_features(train_df, feature_columns=full_columns)
    features_test = prepare_features(test_df, feature_columns=full_columns, encoded_column_names=features_train.columns)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(features_train, train_df[TARGET_COLUMN])
    predictions = model.predict(features_test)

    metrics = compute_metrics(test_df[TARGET_COLUMN].values, predictions)
    return metrics["r2"]


def run_condition(full_df, feature_columns, device, n_compartments, subsample_seed, label):
    working_df = build_subsampled_split(full_df, n_compartments, subsample_seed)
    train_df = working_df[working_df["split"] == "train"]
    val_df = working_df[working_df["split"] == "val"]
    test_df = working_df[working_df["split"] == "test"]
    n_compartments_actual = working_df[SPATIAL_BLOCK_COL].nunique()

    dnn_r2, n_epochs = fit_and_score_dnn(train_df, val_df, test_df, feature_columns, device)
    xgb_r2 = fit_and_score_xgb(train_df, test_df, feature_columns)
    gap = xgb_r2 - dnn_r2

    print(
        f"{label}: n_compartments={n_compartments_actual}  "
        f"train/val/test rows={len(train_df)}/{len(val_df)}/{len(test_df)}  "
        f"DNN R2={dnn_r2:.4f} (epochs={n_epochs})  XGB R2={xgb_r2:.4f}  gap(XGB-DNN)={gap:+.4f}"
    )
    return {
        "label": label, "n_compartments": n_compartments_actual,
        "n_train": len(train_df), "n_val": len(val_df), "n_test": len(test_df),
        "dnn_r2": dnn_r2, "xgb_r2": xgb_r2, "gap": gap,
    }


def main():
    device = select_device()
    print(f"Using device: {device}")

    feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]
    # split_type/split_seed/k_folds/held_out_fold here only control what 'split' column this
    # loader assigns by default -- irrelevant, since build_subsampled_split() always overwrites
    # 'split' itself. Loaded once with "spatial_block" just to get a real, valid split_type.
    full_df = load_split_table_with_terrain(COHORT, "spatial_block", feature_columns, split_seed=SPLIT_SEED)

    results = []
    results.append(run_condition(full_df, feature_columns, device, None, None, "full compartment set"))
    for seed in SUBSAMPLE_SEEDS:
        results.append(
            run_condition(full_df, feature_columns, device, TARGET_N_COMPARTMENTS, seed, f"subsample seed={seed}")
        )

    print("\n=== Summary ===")
    full_gap = results[0]["gap"]
    subsample_gaps = [r["gap"] for r in results[1:]]
    print(f"Full-compartment gap (XGB-DNN): {full_gap:+.4f}")
    print(f"Subsampled (n={TARGET_N_COMPARTMENTS}) gaps: {[f'{g:+.4f}' for g in subsample_gaps]}")
    print(f"Subsampled mean gap: {np.mean(subsample_gaps):+.4f}  SD: {np.std(subsample_gaps):.4f}")


if __name__ == "__main__":
    main()

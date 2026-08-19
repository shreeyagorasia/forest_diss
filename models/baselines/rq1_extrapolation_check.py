# Run as: python -m models.baselines.rq1_extrapolation_check
#
# Tests a candidate reason for "XGBoost beats the DNN on 4survey": how each model handles test
# plots whose feature values fall outside the range the model actually trained on.
#
# Trees split feature space into boxes; each leaf predicts a value bounded by what training rows
# landed in that leaf. A DNN predicts with a smooth, continuous function that can extrapolate
# further from the training data, in either direction. Under spatial_block testing, held-out
# compartments can have terrain/environment combinations the model never trained on -- that is
# the whole point of the split (it is what makes this a genuine generalisation test, not
# interpolation).
#
# If this mechanism is real: the DNN's error should be worse specifically on test rows with
# out-of-training-range feature values. XGBoost's error should not show the same pattern, since
# its leaf predictions stay bounded by what training saw.

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from scipy.stats import spearmanr

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
from models.dnn_env_terrain.dnn_env_terrain import (
    BATCH_SIZE,
    LEARNING_RATE,
    fit as fit_dnn,
    predict as predict_dnn,
)
from models.xgb_baseline.xgb_baseline import FEATURE_COLUMNS, prepare_features

COHORT = "4survey"
FEATURE_SET_NAME = "nested_set3_gated_terrain_wind_vif"
DNN_TRAINING_SEED = 42
DNN_MAX_EPOCHS = 500
DNN_EARLY_STOPPING_PATIENCE = 40
XGB_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.02, random_state=42, n_jobs=1)

# Continuous columns only -- binary flags (Thin, time_since_thinning_missing, recent_thinning_5yr)
# and the categorical thinning_status have no meaningful continuous "out of training range" idea
# the same way a continuous terrain/age value does.
CONTINUOUS_NOENV_COLUMNS = ["Age", "CanopyCover", "time_since_thinning"]


def fit_dnn_and_predict(train_df, val_df, test_df, feature_columns, device):
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
    print(f"  DNN trained for {len(history_df)} epochs")

    age_test, other_test_noenv, target_test = build_tensors(
        test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
    )
    terrain_test = build_terrain_tensor(test_df, scaler_terrain, feature_columns, device)
    other_test = torch.cat([other_test_noenv, terrain_test], dim=1)

    predicted_scaled = predict_dnn(best_model, age_test, other_test)
    predicted = scaler_height.inverse_transform(predicted_scaled.cpu().numpy()).flatten()
    return predicted


def fit_xgb_and_predict(train_df, test_df, feature_columns):
    full_columns = FEATURE_COLUMNS + list(feature_columns)
    features_train = prepare_features(train_df, feature_columns=full_columns)
    features_test = prepare_features(test_df, feature_columns=full_columns, encoded_column_names=features_train.columns)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(features_train, train_df[TARGET_COLUMN])
    return model.predict(features_test)


def compute_extrapolation_score(train_df, test_df, continuous_columns):
    # For each continuous column, how far outside the TRAINING min/max does each test row sit,
    # in units of that column's own training standard deviation (so columns on different scales
    # -- e.g. elevation in metres vs. a 0-1 index -- contribute comparably). 0 if inside range.
    scores = pd.DataFrame(index=test_df.index)
    n_out_of_range = pd.Series(0, index=test_df.index)
    for column in continuous_columns:
        train_min = train_df[column].min()
        train_max = train_df[column].max()
        train_std = train_df[column].std()
        below = (train_min - test_df[column]).clip(lower=0)
        above = (test_df[column] - train_max).clip(lower=0)
        distance = (below + above) / train_std
        scores[column] = distance
        n_out_of_range += (distance > 0).astype(int)

    total_score = scores.sum(axis=1)
    return total_score, n_out_of_range


def main():
    device = select_device()
    feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]
    full_df = load_split_table_with_terrain(COHORT, "spatial_block", feature_columns, split_seed=SPLIT_SEED)

    train_df = full_df[full_df["split"] == "train"]
    val_df = full_df[full_df["split"] == "val"]
    test_df = full_df[full_df["split"] == "test"]
    print(f"train/val/test rows: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    dnn_predictions = fit_dnn_and_predict(train_df, val_df, test_df, feature_columns, device)
    xgb_predictions = fit_xgb_and_predict(train_df, test_df, feature_columns)

    observed = test_df[TARGET_COLUMN].to_numpy()
    dnn_abs_error = np.abs(observed - dnn_predictions)
    xgb_abs_error = np.abs(observed - xgb_predictions)

    print(f"\nSanity check -- DNN R2: {compute_metrics(observed, dnn_predictions)['r2']:.4f}  "
          f"XGB R2: {compute_metrics(observed, xgb_predictions)['r2']:.4f}")

    continuous_columns = CONTINUOUS_NOENV_COLUMNS + list(feature_columns)
    extrapolation_score, n_out_of_range = compute_extrapolation_score(train_df, test_df, continuous_columns)

    print(f"\n{len(continuous_columns)} continuous columns checked for out-of-training-range values.")
    print(f"Test rows with >=1 out-of-range feature: {(n_out_of_range > 0).sum()} of {len(test_df)} "
          f"({(n_out_of_range > 0).mean():.1%})")

    in_range_mask = n_out_of_range == 0
    out_of_range_mask = n_out_of_range > 0

    print("\n=== Mean absolute error, in-range vs. out-of-range test rows ===")
    for label, mask in [("in-range", in_range_mask), ("out-of-range", out_of_range_mask)]:
        n = mask.sum()
        print(f"{label} (n={n}): DNN MAE={dnn_abs_error[mask].mean():.4f}  XGB MAE={xgb_abs_error[mask].mean():.4f}")

    dnn_ratio = dnn_abs_error[out_of_range_mask].mean() / dnn_abs_error[in_range_mask].mean()
    xgb_ratio = xgb_abs_error[out_of_range_mask].mean() / xgb_abs_error[in_range_mask].mean()
    print(f"\nOut-of-range / in-range MAE ratio -- DNN: {dnn_ratio:.3f}  XGB: {xgb_ratio:.3f}")
    print("(ratio > 1 means the model does worse on out-of-range rows; if the DNN's ratio is")
    print(" clearly larger than XGBoost's, that supports the extrapolation-sensitivity guess)")

    dnn_corr, dnn_p = spearmanr(extrapolation_score, dnn_abs_error)
    xgb_corr, xgb_p = spearmanr(extrapolation_score, xgb_abs_error)
    print(f"\nSpearman correlation, extrapolation score vs. absolute error:")
    print(f"  DNN: rho={dnn_corr:.4f}  p={dnn_p:.4g}")
    print(f"  XGB: rho={xgb_corr:.4f}  p={xgb_p:.4g}")


if __name__ == "__main__":
    main()

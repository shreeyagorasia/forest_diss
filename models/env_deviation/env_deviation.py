# Decoupled deviation model -- predicts the LEFTOVER deviation from a base growth model
# (Chapman-Richards, or dnn_noenv) using terrain/wind features ONLY, with no physics loss and no
# joint training of any kind. See documentation/model_instructions/env_deviation_decoupled_
# instructions.md for the full reasoning this was built from.
#
# Two base-model variants, both real, both worth reporting (not one picked over the other):
#   - "cr": residual = observed height - Chapman-Richards-predicted height (using the SAME
#     split-matched CR anchor pinn_env_terrain already uses, models.common.saving.load_cr_params).
#     Answers "what does terrain explain that the PHYSICS anchor misses."
#   - "dnn_noenv": residual = observed height - dnn_noenv's own prediction. Answers "what does
#     terrain explain that the BEST AVAILABLE STATISTICAL model misses."
#
# This file only knows how to build residual targets and fit/predict the residual model --
# run_env_deviation.py is the one script that loads data, calls these, composes the final
# prediction, and evaluates/saves it (this repo has no separate fit-then-evaluate split for
# XGBoost-based models -- same convention as xgb_environmental.py's own run script: it fits in
# seconds, there is no cluster step to separate out).

import numpy as np
import torch

from models.chapman_richards.chapman_richards import chapman_richards
from models.common.torch_data import build_tensors
from models.dnn_noenv.dnn_noenv import load_best_model as load_dnn_noenv_model
from models.dnn_noenv.dnn_noenv import predict as predict_dnn_noenv
from models.xgb_environmental.xgb_environmental import fit_with_columns, predict_with_columns

# Same fixed config used for the 2026-08-02 scope-matched XGBoost check, which is what this
# model's design directly builds on -- not re-tuned here. models/xgb_environmental/
# xgb_environmental.py's own HYPERPARAMETER_GRID exists for a real sweep later (see the
# instructions doc's section 3); this stays the untuned base case, same "what base case means"
# convention every other model in this repo starts from.
XGB_MAX_DEPTH = 6
XGB_N_ESTIMATORS = 500
XGB_LEARNING_RATE = 0.1
# n_jobs=1: default multi-threaded XGBoost segfaults in the same process as PyTorch on this Mac
# (a known torch/xgboost OpenMP conflict, confirmed 2026-08-02 -- KMP_DUPLICATE_LIB_OK=TRUE alone
# isn't enough).
XGB_N_JOBS = 1

# Held-out fraction of the VAL split used for the residual model's own early stopping (the
# "dnn_noenv" base variant only -- see build_dnn_noenv_residual_target()'s own note for why val,
# not train, is used at all).
RESIDUAL_VAL_HOLDOUT_FRACTION = 0.2
RESIDUAL_VAL_HOLDOUT_SEED = 42


def build_cr_residual_target(split_df, cr_params, target_column):
    # Row-level (one value per plot-survey-year, not averaged per plot like the existing
    # mean_cr_residual) -- age-varying deviation is preserved. Uses the split-MATCHED anchor
    # (models.common.saving.load_cr_params), not the pooled one mean_cr_residual still reads --
    # see the instructions doc's comparison table for why that distinction matters.
    base_prediction = chapman_richards(
        split_df["Age"].values, cr_params["y_max"], cr_params["k"], cr_params["p"]
    )
    residual = split_df[target_column].values - base_prediction
    return base_prediction, residual


def build_dnn_noenv_residual_target(split_df, cohort, split_type, device, target_column):
    # Loads dnn_noenv's own saved checkpoint + scalers and predicts on EVERY row of split_df
    # (train/val/test alike), unscaled back to raw metres, then computes the residual against
    # that. Reuses models/common/saving.py::model_output_dir() + the exact preprocessing files
    # run_dnn_noenv.py already saves -- same loading pattern evaluate_dnn_noenv.py uses.
    from models.common.saving import model_output_dir

    import joblib
    import json

    output_dir = model_output_dir("dnn_noenv", cohort, split_type=split_type)
    checkpoints_dir = output_dir / "checkpoints"
    preprocessing_dir = output_dir / "preprocessing"

    with open(checkpoints_dir / "architecture.json") as f:
        architecture = json.load(f)
    n_other_features = architecture["n_other_features"]
    hidden_layer_sizes = architecture.get("hidden_layer_sizes")

    scaler_age = joblib.load(preprocessing_dir / "scaler_age.joblib")
    scaler_other_features = joblib.load(preprocessing_dir / "scaler_other_features.joblib")
    scaler_height = joblib.load(preprocessing_dir / "scaler_height.joblib")
    with open(preprocessing_dir / "encoded_column_names.json") as f:
        encoded_column_names = json.load(f)

    model = load_dnn_noenv_model(n_other_features, device, checkpoints_dir, hidden_layer_sizes=hidden_layer_sizes)

    age_tensor, other_tensor, _ = build_tensors(
        split_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device,
        include_target=False,
    )
    predicted_scaled = predict_dnn_noenv(model, age_tensor, other_tensor)
    base_prediction = scaler_height.inverse_transform(predicted_scaled.cpu().numpy()).flatten()

    residual = split_df[target_column].values - base_prediction
    return base_prediction, residual


def split_val_for_residual_early_stopping(val_df, seed=RESIDUAL_VAL_HOLDOUT_SEED, holdout_fraction=RESIDUAL_VAL_HOLDOUT_FRACTION):
    # IMPORTANT, "dnn_noenv" base variant only: the residual model must NOT be fit on dnn_noenv's
    # TRAIN-set residuals. dnn_noenv's train_loss sits well below its val_loss (it overfits, like
    # every model in this repo's training curves) -- fitting the residual model to train-set
    # residuals would fit it to an artificially small, optimistic error, not the error dnn_noenv
    # actually leaves on unseen data. Using VAL-set residuals instead avoids this: dnn_noenv's own
    # gradients never touched val rows (early stopping only used val_loss to pick WHICH checkpoint
    # to keep, not to update weights -- a much smaller, generally-accepted form of bias, not the
    # same thing as training on it directly). This function further splits val into a
    # residual-fit/residual-early-stopping pair, purely for the residual model's OWN val_df in
    # fit_with_columns() -- the actual TEST set is never touched until final composed evaluation.
    rng = np.random.default_rng(seed)
    shuffled_index = rng.permutation(val_df.index.to_numpy())
    n_holdout = int(len(shuffled_index) * holdout_fraction)
    holdout_index = shuffled_index[:n_holdout]
    fit_index = shuffled_index[n_holdout:]
    return val_df.loc[fit_index], val_df.loc[holdout_index]


def fit_residual_model(residual_fit_df, residual_early_stopping_df, feature_columns, target_col="residual", seed=42):
    return fit_with_columns(
        residual_fit_df, feature_columns, val_df=residual_early_stopping_df, target_col=target_col, seed=seed,
        max_depth=XGB_MAX_DEPTH, n_estimators=XGB_N_ESTIMATORS, learning_rate=XGB_LEARNING_RATE, n_jobs=XGB_N_JOBS,
    )


def predict_residual(residual_model, df, feature_columns):
    return predict_with_columns(df, residual_model, feature_columns)

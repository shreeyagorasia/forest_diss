import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from models.common.splits import assert_no_split_columns_in_features

FEATURE_COLUMNS = [
    "Age",
    "CanopyCover",
    "Thin",
    "time_since_thinning",
    "time_since_thinning_missing",
    "recent_thinning_5yr",
    "thinning_status",
]
# yldc removed 2026-07-28. Real ablation showed it hurts generalisation, see progress_notes.md.
# Thin/time_since_thinning/time_since_thinning_missing/recent_thinning_5yr added 2026-07-30 --
# this baseline previously got only the categorical thinning_status while rf_baseline got only
# these continuous/binary ones and DNN/PINN got both, with no reason on record; now all three
# baseline families see the same thinning information (see rf_baseline.py for the matching change).
CATEGORICAL_COLUMNS = ["thinning_status"]

assert_no_split_columns_in_features(FEATURE_COLUMNS, "linear_baseline")

# Fixed category order so drop_first (below) always drops "never_thinned" specifically,
# not whatever happens to sort first alphabetically. See encode_features().
THINNING_STATUS_CATEGORIES = [
    "never_thinned", "recent_0_5yr", "mid_6_10yr", "old_10plus_yr",
    "not_yet_thinned_at_survey", "unknown_timing",
]


def encode_features(df, feature_columns=None, encoded_column_names=None):
    # Turn thinning_status (a category like "never_thinned") into separate
    # 0/1 columns a linear model can use.
    #
    # drop_first=True matters here: without it, every category gets its own
    # column, and since every row's dummy columns always sum to exactly 1,
    # that set of columns is an exact linear combination of LinearRegression's
    # implicit intercept column. The design matrix is then rank-deficient --
    # sklearn's solver doesn't error, it silently returns a minimum-norm
    # solution, and the resulting per-category coefficients aren't uniquely
    # identified (a real problem when the coefficients themselves are meant
    # to be interpreted, not just used for prediction). Dropping one category
    # removes the redundancy; fixing it to "never_thinned" via the categorical
    # ordering above (rather than letting pandas pick whichever sorts first
    # alphabetically) makes every remaining coefficient a well-defined
    # "vs. never thinned" effect.
    #
    # If encoded_column_names is given, match this data to the exact columns
    # seen during training: a category seen only during training gets a
    # column of 0s here (fill_value=0), and a category not seen during
    # training is simply dropped, so the model never sees an unexpected
    # column at prediction time.
    #
    # feature_columns defaults to the plain FEATURE_COLUMNS (Age + management only), matching
    # every existing baseline result exactly. Pass an extended list (e.g. FEATURE_COLUMNS +
    # terrain/wind columns) to test whether adding the same environmental information the
    # neural models see closes any of their gap over this baseline. Added 2026-08-09 for that
    # specific check, see models/baselines/run_baselines_env.py.
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    encoded = df[feature_columns].copy()
    encoded["thinning_status"] = pd.Categorical(encoded["thinning_status"], categories=THINNING_STATUS_CATEGORIES)
    # time_since_thinning is NaN for plots that have never been thinned (time_since_thinning_missing
    # is True for those rows). LinearRegression can't handle NaN directly, so fill it with 0 here,
    # same as rf_baseline does; the missing flag is what actually tells the model "never thinned".
    encoded["time_since_thinning"] = encoded["time_since_thinning"].fillna(0)
    encoded = pd.get_dummies(encoded, columns=CATEGORICAL_COLUMNS, drop_first=True)

    if encoded_column_names is not None:
        encoded = encoded.reindex(columns=encoded_column_names, fill_value=0)

    return encoded


def fit(train_df, target_col="elev_percentile_95th", extra_feature_columns=None):
    feature_columns = FEATURE_COLUMNS + list(extra_feature_columns or [])
    encoded_train = encode_features(train_df, feature_columns=feature_columns)

    model = LinearRegression()
    model.fit(encoded_train, train_df[target_col])

    coefficients = {
        column_name: float(coefficient)
        for column_name, coefficient in zip(encoded_train.columns, model.coef_)
    }

    return {
        "intercept": float(model.intercept_),
        "coefficients": coefficients,
        "feature_columns": feature_columns,
        "encoded_column_names": list(encoded_train.columns),
    }


def predict(df, params):
    # Predictions are rebuilt from the plain saved numbers (intercept +
    # coefficients), not from a saved sklearn model object — this keeps a
    # linear model's saved output as simple as Chapman-Richards' params.json
    # or average-by-age's lookup.json, since linear regression has nothing
    # else worth checkpointing.
    #
    # feature_columns falls back to plain FEATURE_COLUMNS for params saved before this key
    # existed (2026-08-09). Keeps every existing saved params.json loadable unchanged.
    encoded = encode_features(
        df, feature_columns=params.get("feature_columns", FEATURE_COLUMNS),
        encoded_column_names=params["encoded_column_names"],
    )

    predicted = pd.Series(params["intercept"], index=encoded.index)
    for column_name, coefficient in params["coefficients"].items():
        predicted = predicted + coefficient * encoded[column_name]

    return predicted.values


def save_params(params, cohort, n_rows_fit, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "intercept": params["intercept"],
        "coefficients": params["coefficients"],
        "feature_columns": params.get("feature_columns", FEATURE_COLUMNS),
        "encoded_column_names": params["encoded_column_names"],
        "cohort": cohort,
        "n_rows_fit": n_rows_fit,
        "fit_date": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result

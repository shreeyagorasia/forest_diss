import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

FEATURE_COLUMNS = ["Age", "CanopyCover", "thinning_status"]
# yldc removed 2026-07-28 -- real ablation showed it hurts generalisation, see progress_notes.md.
CATEGORICAL_COLUMNS = ["thinning_status"]


def encode_features(df, encoded_column_names=None):
    # Turn thinning_status (a category like "never_thinned") into separate
    # 0/1 columns a linear model can use.
    #
    # If encoded_column_names is given, match this data to the exact columns
    # seen during training: a category seen only during training gets a
    # column of 0s here (fill_value=0), and a category not seen during
    # training is simply dropped, so the model never sees an unexpected
    # column at prediction time.
    encoded = pd.get_dummies(df[FEATURE_COLUMNS], columns=CATEGORICAL_COLUMNS)

    if encoded_column_names is not None:
        encoded = encoded.reindex(columns=encoded_column_names, fill_value=0)

    return encoded


def fit(train_df, target_col="elev_percentile_95th"):
    encoded_train = encode_features(train_df)

    model = LinearRegression()
    model.fit(encoded_train, train_df[target_col])

    coefficients = {
        column_name: float(coefficient)
        for column_name, coefficient in zip(encoded_train.columns, model.coef_)
    }

    return {
        "intercept": float(model.intercept_),
        "coefficients": coefficients,
        "encoded_column_names": list(encoded_train.columns),
    }


def predict(df, params):
    # Predictions are rebuilt from the plain saved numbers (intercept +
    # coefficients), not from a saved sklearn model object — this keeps a
    # linear model's saved output as simple as Chapman-Richards' params.json
    # or average-by-age's lookup.json, since linear regression has nothing
    # else worth checkpointing.
    encoded = encode_features(df, encoded_column_names=params["encoded_column_names"])

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
        "encoded_column_names": params["encoded_column_names"],
        "cohort": cohort,
        "n_rows_fit": n_rows_fit,
        "fit_date": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xgboost as xgb

from models.common.splits import assert_no_split_columns_in_features
from models.linear_baseline.linear_baseline import THINNING_STATUS_CATEGORIES

# Same base feature list as linear_baseline/rf_baseline, on purpose. This is a third baseline
# reference point, not a different question. Added 2026-08-09 alongside the environmental-feature
# extension (models/baselines/run_baselines_env.py) as a stronger tree-based baseline than RF,
# closer in capacity to the neural models, for the "does terrain/wind information alone let a
# simple model match pinn_env_terrain_k / dnn_env_terrain" check.
FEATURE_COLUMNS = [
    "Age",
    "CanopyCover",
    "Thin",
    "time_since_thinning",
    "time_since_thinning_missing",
    "recent_thinning_5yr",
    "thinning_status",
]
CATEGORICAL_COLUMNS = ["thinning_status"]

assert_no_split_columns_in_features(FEATURE_COLUMNS, "xgb_baseline")


def prepare_features(df, feature_columns=None, encoded_column_names=None):
    # Same encoding as rf_baseline.py's prepare_features(), on purpose. Both are tree-based
    # models with no identifiability concern the way linear_baseline's drop_first has, so this
    # is only here for consistency with the other two baselines, not a technical requirement.
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    features = df[feature_columns].copy()
    features["time_since_thinning"] = features["time_since_thinning"].fillna(0)

    features["thinning_status"] = pd.Categorical(features["thinning_status"], categories=THINNING_STATUS_CATEGORIES)
    features = pd.get_dummies(features, columns=CATEGORICAL_COLUMNS, drop_first=True)

    if encoded_column_names is not None:
        features = features.reindex(columns=encoded_column_names, fill_value=0)

    return features


def fit(train_df, target_col="elev_percentile_95th", n_estimators=100, seed=42, extra_feature_columns=None):
    # sklearn/xgboost defaults otherwise (no tuning). Same "reference point, not a tuned
    # model" philosophy as rf_baseline.py, deliberately not the more heavily tuned hyperparameters
    # used elsewhere in this project (e.g. explain_signal.py's n_estimators=500/max_depth=4/
    # learning_rate=0.04). Those are for a different attribution question, not this baseline
    # comparison.
    feature_columns = FEATURE_COLUMNS + list(extra_feature_columns or [])
    features_train = prepare_features(train_df, feature_columns=feature_columns)

    # n_jobs=1. XGBoost's default multi-threading segfaults on this machine (macOS ARM64,
    # xgboost 3.3.0), a known class of OpenMP-runtime-conflict issue on Apple Silicon, not
    # specific to this baseline. Single-threaded avoids it; fine here since these are small,
    # fast baseline fits, not a training-time bottleneck worth multi-threading for.
    model = xgb.XGBRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=1)
    model.fit(features_train, train_df[target_col])

    return model, list(features_train.columns), feature_columns


def predict(df, model, encoded_column_names, feature_columns=None):
    features = prepare_features(df, feature_columns=feature_columns, encoded_column_names=encoded_column_names)
    return model.predict(features)


def save_model(model, encoded_column_names, cohort, n_rows_fit, output_dir, feature_columns=None):
    # Same "save the real model, not just numbers" approach as rf_baseline.py. An XGBoost
    # ensemble has no small set of numbers that rebuilds it either.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.json"
    model.save_model(model_path)

    metadata = {
        "n_estimators": model.n_estimators,
        "feature_columns": feature_columns if feature_columns is not None else FEATURE_COLUMNS,
        "encoded_column_names": encoded_column_names,
        "cohort": cohort,
        "n_rows_fit": n_rows_fit,
        "fit_date": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return model_path


def load_model(output_dir):
    model = xgb.XGBRegressor()
    model.load_model(Path(output_dir) / "model.json")
    return model

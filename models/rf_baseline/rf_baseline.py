import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from models.common.splits import assert_no_split_columns_in_features
from models.linear_baseline.linear_baseline import THINNING_STATUS_CATEGORIES

FEATURE_COLUMNS = [
    "Age",
    "CanopyCover",
    "Thin",
    "time_since_thinning",
    "time_since_thinning_missing",
    "recent_thinning_5yr",
    "thinning_status",
    # yldc removed 2026-07-28 -- real ablation showed it hurts test R2 here (0.446->0.498
    # without it), see progress_notes.md. thinning_status added 2026-07-30 -- it was previously
    # given only to linear_baseline and DNN/PINN, not this baseline, with no reason on record; see
    # linear_baseline.py for why the category order (THINNING_STATUS_CATEGORIES, imported from
    # there so both baselines encode it identically) matters for drop_first.
]
CATEGORICAL_COLUMNS = ["thinning_status"]

assert_no_split_columns_in_features(FEATURE_COLUMNS, "rf_baseline")


def prepare_features(df, feature_columns=None, encoded_column_names=None):
    # time_since_thinning is NaN for plots that have never been thinned
    # (time_since_thinning_missing is True for those rows). A random forest
    # cannot handle NaN directly, so fill it with 0 here — the missing flag
    # is what actually tells the model "this plot was never thinned",
    # the filled 0 is just a placeholder value the tree splits can ignore.
    #
    # feature_columns defaults to plain FEATURE_COLUMNS (Age + management only) -- pass an
    # extended list (e.g. + terrain/wind columns) to give this baseline the same environmental
    # information the neural models see. Added 2026-08-09, see
    # models/baselines/run_baselines_env.py.
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS
    features = df[feature_columns].copy()
    features["time_since_thinning"] = features["time_since_thinning"].fillna(0)

    # thinning_status is a category (like "never_thinned"), not a number -- one-hot encode it
    # the same way linear_baseline does, so both baselines see thinning information encoded
    # identically. A random forest doesn't have linear_baseline's identifiability problem (see
    # that module's encode_features() for why drop_first matters there), but dropping one
    # category here too keeps the feature columns consistent across every baseline that uses
    # thinning_status.
    features["thinning_status"] = pd.Categorical(features["thinning_status"], categories=THINNING_STATUS_CATEGORIES)
    features = pd.get_dummies(features, columns=CATEGORICAL_COLUMNS, drop_first=True)

    if encoded_column_names is not None:
        features = features.reindex(columns=encoded_column_names, fill_value=0)

    return features


def fit(train_df, target_col="elev_percentile_95th", n_estimators=100, seed=42, extra_feature_columns=None):
    # sklearn defaults otherwise (no tuning yet) -- this is a baseline
    # reference point, not a tuned model.
    feature_columns = FEATURE_COLUMNS + list(extra_feature_columns or [])
    features_train = prepare_features(train_df, feature_columns=feature_columns)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
    model.fit(features_train, train_df[target_col])

    return model, list(features_train.columns), feature_columns


def predict(df, model, encoded_column_names, feature_columns=None):
    features = prepare_features(df, feature_columns=feature_columns, encoded_column_names=encoded_column_names)
    return model.predict(features)


def save_model(model, encoded_column_names, cohort, n_rows_fit, output_dir, feature_columns=None):
    # Unlike Chapman-Richards, average-by-age, and linear regression, a
    # fitted forest is hundreds of decision trees -- there is no small set
    # of numbers that can rebuild it, so the actual fitted model is saved
    # here (model.joblib), alongside a small metadata JSON matching the
    # other baselines' style.
    #
    # model.joblib is NOT committed to Git (see outputs/ in .gitignore) --
    # with sklearn's default unlimited tree depth, it can be hundreds of MB
    # to over 1GB depending on training set size, well past GitHub's 100MB
    # limit. Compressed joblib keeps the same load path/API while reducing
    # disk usage for newly generated artifacts. The model is deterministic and
    # fast to regenerate (fixed random_state, well under a minute), so there is
    # no real cost to leaving it out of version control -- just rerun
    # models/baselines/run_baselines.py.
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    joblib.dump(model, model_path, compress=3)

    metadata = {
        "n_estimators": model.n_estimators,
        "joblib_compress": 3,
        "feature_columns": feature_columns if feature_columns is not None else FEATURE_COLUMNS,
        # The one-hot encoded column names actually seen at fit time (thinning_status expands
        # into several columns, one dropped) -- predict() needs this exact list to align a new
        # dataset's columns, the same reason linear_baseline saves encoded_column_names.
        "encoded_column_names": encoded_column_names,
        "cohort": cohort,
        "n_rows_fit": n_rows_fit,
        "fit_date": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return model_path


def load_model(output_dir):
    return joblib.load(Path(output_dir) / "model.joblib")

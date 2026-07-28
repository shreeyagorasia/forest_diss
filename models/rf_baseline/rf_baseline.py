import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor

FEATURE_COLUMNS = [
    "Age",
    "CanopyCover",
    "Thin",
    "time_since_thinning",
    "time_since_thinning_missing",
    "recent_thinning_5yr",
    # yldc removed 2026-07-28 -- real ablation showed it hurts test R2 here (0.446->0.498
    # without it), see progress_notes.md.
]


def prepare_features(df):
    # time_since_thinning is NaN for plots that have never been thinned
    # (time_since_thinning_missing is True for those rows). A random forest
    # cannot handle NaN directly, so fill it with 0 here — the missing flag
    # is what actually tells the model "this plot was never thinned",
    # the filled 0 is just a placeholder value the tree splits can ignore.
    features = df[FEATURE_COLUMNS].copy()
    features["time_since_thinning"] = features["time_since_thinning"].fillna(0)
    return features


def fit(train_df, target_col="elev_percentile_95th", n_estimators=100, seed=42):
    # sklearn defaults otherwise (no tuning yet) -- this is a baseline
    # reference point, not a tuned model.
    features_train = prepare_features(train_df)

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=seed)
    model.fit(features_train, train_df[target_col])

    return model


def predict(df, model):
    features = prepare_features(df)
    return model.predict(features)


def save_model(model, cohort, n_rows_fit, output_dir):
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
        "feature_columns": FEATURE_COLUMNS,
        "cohort": cohort,
        "n_rows_fit": n_rows_fit,
        "fit_date": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return model_path


def load_model(output_dir):
    return joblib.load(Path(output_dir) / "model.joblib")

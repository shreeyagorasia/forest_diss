# Purpose: a real, per-target XGBoost hyperparameter search for RQ1's own task (predicting
# elev_percentile_95th, Set3, both cohorts). Replacing the config borrowed from RQ3's
# explain_signal.py (n_estimators=500, max_depth=4, learning_rate=0.04), which was never actually
# searched on RQ1's own data. Reuses the exact same data/split pipeline as
# xgb_hyperparameter_sensitivity_check.py's fit_and_score_rq1_fold().
#
# Model selection uses VALIDATION R2 only, never test R2, to avoid tuning-on-test leakage. Test R2
# is reported only for the winning config, after selection is already decided.
#
# Efficiency note: the spatial_block_kfold split (and the environmental-feature merge) does not
# depend on hyperparameters at all, so each (cohort, fold)'s data is built ONCE and reused across
# all 27 hyperparameter configs. Not recomputed per config, which would be ~27x slower for no
# benefit.
#
# Run as: python -m models.baselines.rq1_xgb_hyperparameter_search

import itertools

import numpy as np
import pandas as pd
import xgboost as xgb

from models.baselines.run_baselines import DEFAULT_K_FOLDS, MATURITY_AGE_MIN_DEFAULT, SEED, build_split_for_cohort
from models.baselines.run_baselines_env import load_full_rows_with_split, merge_environmental_features
from models.common.metrics import compute_metrics
from models.common.torch_data import ENV_TERRAIN_FEATURE_SETS
from models.xgb_baseline.xgb_baseline import prepare_features, FEATURE_COLUMNS

SET_NAME = "nested_set3_gated_terrain_wind_vif"

GRID = {
    "n_estimators": [300, 500, 800],
    "max_depth": [3, 4, 6],
    "learning_rate": [0.02, 0.04, 0.08],
}
CONFIGS = [dict(zip(GRID.keys(), values)) for values in itertools.product(*GRID.values())]

# The config RQ2b/RQ3 borrow from explain_signal.py, kept here for direct comparison.
BORROWED_CONFIG = {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.04}


def build_fold_features(cohort, held_out_fold, k_folds=DEFAULT_K_FOLDS):
    name_suffix = f"_fold{held_out_fold}"
    _, split_assignment = build_split_for_cohort(
        cohort, "spatial_block_kfold", split_seed=SEED, maturity_age_min=MATURITY_AGE_MIN_DEFAULT,
        name_suffix=name_suffix, k_folds=k_folds, held_out_fold=held_out_fold,
    )
    filtered_df = load_full_rows_with_split(cohort, split_assignment)
    env_columns = list(ENV_TERRAIN_FEATURE_SETS[SET_NAME])
    df = merge_environmental_features(filtered_df, env_columns, cohort)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    full_columns = FEATURE_COLUMNS + env_columns
    features_train = prepare_features(train_df, feature_columns=full_columns)
    features_val = prepare_features(val_df, feature_columns=full_columns, encoded_column_names=features_train.columns)
    features_test = prepare_features(test_df, feature_columns=full_columns, encoded_column_names=features_train.columns)
    return (
        features_train, train_df["elev_percentile_95th"],
        features_val, val_df["elev_percentile_95th"],
        features_test, test_df["elev_percentile_95th"],
    )


def fit_and_score(cached_fold_data, params, full_metrics=False):
    features_train, y_train, features_val, y_val, features_test, y_test = cached_fold_data
    model = xgb.XGBRegressor(random_state=SEED, n_jobs=1, early_stopping_rounds=20, **params)
    model.fit(features_train, y_train, eval_set=[(features_val, y_val)], verbose=False)
    val_pred = model.predict(features_val)
    test_pred = model.predict(features_test)
    val_metrics = compute_metrics(y_val.to_numpy(), val_pred)
    test_metrics = compute_metrics(y_test.to_numpy(), test_pred)
    if full_metrics:
        return val_metrics, test_metrics
    return val_metrics["r2"], test_metrics["r2"]


def run_grid_search():
    results = []
    for cohort in ["4survey", "6survey"]:
        fold_data = [build_fold_features(cohort, fold) for fold in range(DEFAULT_K_FOLDS)]
        for params in CONFIGS:
            val_r2s, test_r2s = [], []
            for fold in range(DEFAULT_K_FOLDS):
                val_r2, test_r2 = fit_and_score(fold_data[fold], params)
                val_r2s.append(val_r2)
                test_r2s.append(test_r2)
            results.append({
                "cohort": cohort, **params,
                "mean_val_r2": np.mean(val_r2s),
                "mean_test_r2": np.mean(test_r2s),
                "std_test_r2": np.std(test_r2s),
            })
    return pd.DataFrame(results)


def refit_winning_config_full_metrics(cohort, params):
    fold_rows = []
    for fold in range(DEFAULT_K_FOLDS):
        fold_data = build_fold_features(cohort, fold)
        _, test_metrics = fit_and_score(fold_data, params, full_metrics=True)
        fold_rows.append({"r2": test_metrics["r2"], "rmse": test_metrics["rmse"], "mae": test_metrics["mae"]})
    fold_df = pd.DataFrame(fold_rows)
    return fold_df.mean(), fold_df.std()


def main():
    results_df = run_grid_search()

    print("=== Best config per cohort, selected on VAL R2 ===")
    winning_configs = {}
    for cohort in ["4survey", "6survey"]:
        subset = results_df[results_df["cohort"] == cohort]
        best = subset.loc[subset["mean_val_r2"].idxmax()]
        winning_configs[cohort] = {k: best[k] for k in GRID}
        print(f"{cohort}: {best.to_dict()}")

    print()
    print("=== Borrowed config's result, for comparison ===")
    borrowed = results_df[
        (results_df["n_estimators"] == BORROWED_CONFIG["n_estimators"])
        & (results_df["max_depth"] == BORROWED_CONFIG["max_depth"])
        & (results_df["learning_rate"] == BORROWED_CONFIG["learning_rate"])
    ]
    print(borrowed[["cohort", "mean_val_r2", "mean_test_r2", "std_test_r2"]].to_string(index=False))

    print()
    print("=== Winning configs' full test-set metrics (R2/RMSE/MAE) ===")
    for cohort, params in winning_configs.items():
        params = {k: int(v) if k != "learning_rate" else float(v) for k, v in params.items()}
        mean_metrics, std_metrics = refit_winning_config_full_metrics(cohort, params)
        print(f"{cohort} {params}: r2={mean_metrics['r2']:.4f}+-{std_metrics['r2']:.4f} "
              f"rmse={mean_metrics['rmse']:.4f}+-{std_metrics['rmse']:.4f} "
              f"mae={mean_metrics['mae']:.4f}+-{std_metrics['mae']:.4f}")


if __name__ == "__main__":
    main()

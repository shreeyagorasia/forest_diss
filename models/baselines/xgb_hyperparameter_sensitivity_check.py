# Purpose: check whether RQ1's and RQ2b's XGBoost results are sensitive to using raw XGBoost
# defaults (n_estimators=100, max_depth=6, learning_rate=0.3, no early stopping -- confirmed via
# a real fit, not assumed) versus RQ3's own fixed, hand-picked config (n_estimators=500,
# max_depth=4, learning_rate=0.04, early stopping against a real validation fold). RQ1/RQ2 never
# had anyone choose settings for their own data; RQ3 did, for a different, older reason (fixing a
# documented overfitting failure on 6survey's small n -- see spatial_cv_check.py's own comment).
# This does NOT change any existing pipeline behaviour -- it is a one-off, local-only comparison
# script, reusing each RQ's own existing split logic so the comparison is apples-to-apples.
#
# Run as: python -m models.baselines.xgb_hyperparameter_sensitivity_check

import numpy as np
import pandas as pd
import xgboost as xgb

from models.baselines.run_baselines import DEFAULT_K_FOLDS, MATURITY_AGE_MIN_DEFAULT, SEED, build_split_for_cohort
from models.baselines.run_baselines_env import load_full_rows_with_split, merge_environmental_features
from models.common.metrics import compute_metrics
from models.common.torch_data import ENV_TERRAIN_FEATURE_SETS
from models.xgb_baseline.xgb_baseline import prepare_features
from models.xgb_environmental.data import load_plots_for_cohort

FIXED_PARAMS = {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.04}


def fit_and_score_rq1_fold(cohort, set_name, held_out_fold, k_folds=DEFAULT_K_FOLDS):
    name_suffix = f"_fold{held_out_fold}"
    _, split_assignment = build_split_for_cohort(
        cohort, "spatial_block_kfold", split_seed=SEED, maturity_age_min=MATURITY_AGE_MIN_DEFAULT,
        name_suffix=name_suffix, k_folds=k_folds, held_out_fold=held_out_fold,
    )
    filtered_df = load_full_rows_with_split(cohort, split_assignment)
    env_columns = list(ENV_TERRAIN_FEATURE_SETS[set_name])
    df = merge_environmental_features(filtered_df, env_columns, cohort)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    # prepare_features needs the SAME extra columns appended -- replicate xgb_baseline.fit()'s own
    # "FEATURE_COLUMNS + extra_feature_columns" construction directly here since fit() itself has
    # no hook for custom xgb_params/early stopping.
    from models.xgb_baseline.xgb_baseline import FEATURE_COLUMNS
    full_columns = FEATURE_COLUMNS + env_columns
    features_train = prepare_features(train_df, feature_columns=full_columns)
    features_val = prepare_features(val_df, feature_columns=full_columns, encoded_column_names=features_train.columns)
    features_test = prepare_features(test_df, feature_columns=full_columns, encoded_column_names=features_train.columns)

    model = xgb.XGBRegressor(random_state=SEED, n_jobs=1, early_stopping_rounds=20, **FIXED_PARAMS)
    model.fit(
        features_train, train_df["elev_percentile_95th"],
        eval_set=[(features_val, val_df["elev_percentile_95th"])], verbose=False,
    )
    predictions = model.predict(features_test)
    return compute_metrics(test_df["elev_percentile_95th"].to_numpy(), predictions)["r2"]


def fit_and_score_rq2_fold(cohort, set_name, held_out_fold, k_folds=DEFAULT_K_FOLDS):
    from models.common.splits import SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, spatial_kfold_split
    from models.elasticnet_environmental.elasticnet_environmental import drop_rows_with_missing_features
    from models.xgb_environmental.feature_set_builder import load_feature_set

    feature_columns = load_feature_set("RSQ2", set_name)
    plots_df = load_plots_for_cohort(cohort).copy()
    plots_df["split"] = spatial_kfold_split(
        plots_df, block_col=SPATIAL_BLOCK_COL, k=k_folds, held_out_fold=held_out_fold,
        buffer_distance=SPATIAL_BUFFER_METRES, seed=SEED,
    )

    train_df = drop_rows_with_missing_features(plots_df[plots_df["split"] == "train"], feature_columns)
    val_df = drop_rows_with_missing_features(plots_df[plots_df["split"] == "val"], feature_columns)
    test_df = drop_rows_with_missing_features(plots_df[plots_df["split"] == "test"], feature_columns)

    model = xgb.XGBRegressor(random_state=SEED, n_jobs=1, early_stopping_rounds=20, **FIXED_PARAMS)
    model.fit(
        train_df[feature_columns], train_df["mean_cr_residual"],
        eval_set=[(val_df[feature_columns], val_df["mean_cr_residual"])], verbose=False,
    )
    predictions = model.predict(test_df[feature_columns])
    return compute_metrics(test_df["mean_cr_residual"].to_numpy(), predictions)["r2"], test_df, predictions


def main():
    print("===== RQ1 Set3, fixed-hyperparameter XGBoost, 5-fold =====")
    for cohort in ["4survey", "6survey"]:
        fold_r2 = [fit_and_score_rq1_fold(cohort, "nested_set3_gated_terrain_wind_vif", fold) for fold in range(DEFAULT_K_FOLDS)]
        print(f"  {cohort}: per-fold R2={[round(x,4) for x in fold_r2]}  mean={np.mean(fold_r2):.4f}  sd={np.std(fold_r2):.4f}")

    print("\n===== RQ2b, fixed-hyperparameter XGBoost, 5-fold, 4survey only =====")
    for set_name in ["nested_set2_top10", "nested_set3_gated_terrain_wind_vif", "nested_set4_gated_all_vif"]:
        fold_results = [fit_and_score_rq2_fold("4survey", set_name, fold) for fold in range(DEFAULT_K_FOLDS)]
        fold_r2 = [r[0] for r in fold_results]
        pooled_test = pd.concat([r[1] for r in fold_results], ignore_index=True)
        pooled_pred = np.concatenate([r[2] for r in fold_results])
        pooled_r2 = compute_metrics(pooled_test["mean_cr_residual"].to_numpy(), pooled_pred)["r2"]
        print(f"  {set_name}: per-fold R2={[round(x,4) for x in fold_r2]}  mean={np.mean(fold_r2):.4f}  sd={np.std(fold_r2):.4f}  pooled={pooled_r2:.4f}")


if __name__ == "__main__":
    main()

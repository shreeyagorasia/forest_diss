# Purpose: XGBoost's own version of RQ2a's residual-reduction question -- "does giving a model
# environmental information shrink its departure from the shared Chapman-Richards curve, and does
# that shrinkage concentrate on the plots CR already does worst on?" RQ2a's existing table only
# ever answers this for DNN/PINN/PINN_k. Since the XGBoost hyperparameter correction
# (xgb_hyperparameter_sensitivity_check.py) found a fairly-configured XGBoost beats DNN on raw
# RQ1 accuracy, this checks whether XGBoost's OWN residual shrinks the same clean, quartile-
# concentrated way PINN's does -- if it doesn't, that's real evidence for PINN's specific
# attribution value, not just an accuracy-contest rerun.
#
# Fits TWO XGBoost arms per fold (env-conditioned Set3, and a no-env control), both using the
# corrected fixed config (n_estimators=500, max_depth=4, learning_rate=0.04, early stopping --
# see xgb_hyperparameter_sensitivity_check.py for why raw defaults would be unfair here), saves
# predictions.csv in the exact schema rq2_residual_reduction.py already expects, then reuses that
# script's own compute_residual_reduction() directly -- no reimplementation of the quartile logic.
#
# Run as: python -m models.spatial_attribution.rq2a_xgb_check

import numpy as np
import pandas as pd
import xgboost as xgb

from models.baselines.run_baselines import DEFAULT_K_FOLDS, MATURITY_AGE_MIN_DEFAULT, SEED, build_split_for_cohort
from models.baselines.run_baselines_env import load_full_rows_with_split, merge_environmental_features
from models.common.saving import model_output_dir
from models.common.torch_data import ENV_TERRAIN_FEATURE_SETS
from models.spatial_attribution.rq2_residual_reduction import compute_residual_reduction, load_predictions
from models.xgb_baseline.xgb_baseline import FEATURE_COLUMNS, prepare_features

FIXED_PARAMS = {"n_estimators": 500, "max_depth": 4, "learning_rate": 0.04}
SET_NAME = "nested_set3_gated_terrain_wind_vif"
TARGET = "elev_percentile_95th"


def fit_one_arm(train_df, val_df, test_df, feature_columns):
    features_train = prepare_features(train_df, feature_columns=feature_columns)
    features_val = prepare_features(val_df, feature_columns=feature_columns, encoded_column_names=features_train.columns)
    features_test = prepare_features(test_df, feature_columns=feature_columns, encoded_column_names=features_train.columns)

    model = xgb.XGBRegressor(random_state=SEED, n_jobs=1, early_stopping_rounds=20, **FIXED_PARAMS)
    model.fit(
        features_train, train_df[TARGET],
        eval_set=[(features_val, val_df[TARGET])], verbose=False,
    )
    predicted = model.predict(features_test)

    # Same schema as every other predictions.csv in this project (chapman_richards, dnn_env_terrain,
    # etc.) -- rq2_residual_reduction.py's load_predictions()/compute_residual_reduction() expect
    # exactly these columns, joined on (identification, LiDAR_year).
    out = test_df[["identification", "blk", "cpmt", "LiDAR_year", "Age"]].copy()
    out["observed_top_height"] = test_df[TARGET].to_numpy()
    out["predicted_top_height"] = predicted
    out["residual"] = out["observed_top_height"] - out["predicted_top_height"]
    out["split"] = "test"
    return out


def fit_and_save_fold(cohort, held_out_fold, k_folds=DEFAULT_K_FOLDS):
    name_suffix = f"_fold{held_out_fold}"
    _, split_assignment = build_split_for_cohort(
        cohort, "spatial_block_kfold", split_seed=SEED, maturity_age_min=MATURITY_AGE_MIN_DEFAULT,
        name_suffix=name_suffix, k_folds=k_folds, held_out_fold=held_out_fold,
    )
    filtered_df = load_full_rows_with_split(cohort, split_assignment)
    env_columns = list(ENV_TERRAIN_FEATURE_SETS[SET_NAME])
    env_df = merge_environmental_features(filtered_df, env_columns, cohort)

    for label, df, feature_columns in [
        ("xgb_env_terrain_fixed_" + SET_NAME, env_df, FEATURE_COLUMNS + env_columns),
        ("xgb_noenv_fixed", filtered_df, FEATURE_COLUMNS),
    ]:
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]
        predictions = fit_one_arm(train_df, val_df, test_df, feature_columns)

        output_dir = model_output_dir(label, cohort, f"fold_{held_out_fold}", split_type="spatial_block_kfold")
        output_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_dir / "predictions.csv", index=False)
        print(f"  Saved {label} / {cohort} / fold {held_out_fold} -> {output_dir / 'predictions.csv'} ({len(predictions)} test rows)")


def run_reduction_check(run_name, cohort, k_folds=DEFAULT_K_FOLDS):
    overall_rows = []
    quartile_tables = []
    for fold in range(k_folds):
        cr_predictions = load_predictions(f"chapman_richards_fold{fold}", cohort, "spatial_block_kfold")
        model_predictions = load_predictions(run_name, cohort, "spatial_block_kfold", held_out_fold=fold)
        overall, by_quartile, _ = compute_residual_reduction(cr_predictions, model_predictions)
        overall["fold"] = fold
        overall_rows.append(overall)
        quartile_tables.append(by_quartile.assign(fold=fold))

    overall_df = pd.DataFrame(overall_rows)
    pooled_quartiles = pd.concat(quartile_tables).groupby("cr_residual_quartile", observed=True).agg(
        mean_reduction=("mean_reduction", "mean"),
    )
    return overall_df, pooled_quartiles


def main():
    for cohort in ["4survey", "6survey"]:
        for fold in range(DEFAULT_K_FOLDS):
            fit_and_save_fold(cohort, fold)

    print("\n===== XGBoost's own RQ2a: residual reduction vs Chapman-Richards =====")
    for label in [f"xgb_env_terrain_fixed_{SET_NAME}", "xgb_noenv_fixed"]:
        for cohort in ["4survey", "6survey"]:
            overall_df, pooled_quartiles = run_reduction_check(label, cohort)
            print(f"\n{label} / {cohort}")
            print(f"  Mean reduction across folds: {overall_df['mean_reduction'].mean():.4f} (SD {overall_df['mean_reduction'].std():.4f})")
            print(pooled_quartiles.to_string())


if __name__ == "__main__":
    main()

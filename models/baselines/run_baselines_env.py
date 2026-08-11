# Run as: python -m models.baselines.run_baselines_env --cohort 4survey --split-type plot_level --feature-set stage2_terrain_wind
#
# Standalone script, deliberately separate from run_baselines.py -- added 2026-08-09 to check
# whether linear/RF/XGBoost baselines given the SAME terrain/wind/environmental features the
# neural models see (dnn_env_terrain, pinn_env_terrain, pinn_env_terrain_k) can match them, or
# whether the neural models' added complexity/physics-conditioning earns its keep. Kept separate
# from run_baselines.py so this never touches the already-established, already-cited baseline
# numbers in that shared script -- this only ADDS a new comparison, it doesn't replace anything.
#
# Reuses run_baselines.py's own split-building logic directly (imported, not copied) so the
# train/val/test rows are identical to the ones the plain baselines and every neural model use.
# Environmental features are merged in from the same one-row-per-plot table
# models/xgb_environmental/data.py already reads (data/processed/environmental/
# plot_environmental_features.parquet), joined on `identification`.

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.baselines.run_baselines import MATURITY_AGE_MIN_DEFAULT, SEED, build_split_for_cohort
from models.common.data import filter_data, load_model_table
from models.common.saving import model_output_dir as output_dir
from models.common.torch_data import ENV_TERRAIN_FEATURE_SETS
from models.linear_baseline.linear_baseline import fit as fit_linear_baseline
from models.linear_baseline.linear_baseline import predict as predict_linear_baseline
from models.rf_baseline.rf_baseline import fit as fit_rf_baseline
from models.rf_baseline.rf_baseline import predict as predict_rf_baseline
from models.xgb_baseline.xgb_baseline import fit as fit_xgb_baseline
from models.xgb_baseline.xgb_baseline import predict as predict_xgb_baseline
from models.xgb_environmental.data import COHORT_SPECIFIC_COLUMNS, load_environmental_features

TARGET_COL = "elev_percentile_95th"


def merge_environmental_features(filtered_df, feature_columns, cohort):
    # model_table.parquet can already carry a stray column of the same name as an environmental
    # feature (found: 'whcl', unclear provenance, possibly unrelated to the canonical value) --
    # drop any such pre-existing column first so the merge always uses
    # plot_environmental_features.parquet as the single source of truth for these columns,
    # matching how Avenue 1/Avenue 2/the env-conditioned neural models already treat that file.
    already_present = [c for c in feature_columns if c in filtered_df.columns]
    if already_present:
        print(f"  Dropping pre-existing column(s) {already_present} from the base table before merge "
              f"(using plot_environmental_features.parquet as the canonical source instead).")
        filtered_df = filtered_df.drop(columns=already_present)

    # A handful of features (tas_mean, groundfrost_mean -- COHORT_SPECIFIC_COLUMNS, see
    # xgb_environmental/data.py) are stored per-cohort in the raw export (tas_mean_4survey /
    # tas_mean_6survey), not as one shared column -- rename this cohort's version to the plain
    # name before selecting, same pattern xgb_environmental's load_plots_for_cohort() uses
    # (not reused directly here since that function also does an unrelated mean_cr_residual
    # dropna this script has no use for).
    raw_env_df = load_environmental_features()
    rename_map = {f"{base}_{cohort}": base for base in COHORT_SPECIFIC_COLUMNS if f"{base}_{cohort}" in raw_env_df.columns}
    raw_env_df = raw_env_df.rename(columns=rename_map)

    env_df = raw_env_df[["identification"] + feature_columns].copy()
    n_before = len(filtered_df)
    merged = filtered_df.merge(env_df, on="identification", how="left")
    n_missing = int(merged[feature_columns].isna().any(axis=1).sum())
    if n_missing:
        print(
            f"  Warning: {n_missing}/{n_before} rows have no environmental data for one or more "
            f"of {feature_columns} -- dropping these rows (not imputing)."
        )
        merged = merged.dropna(subset=feature_columns)
    return merged


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "n": int(len(y_true)),
    }


def run_one_model(model_name, fit_fn, predict_fn, train_df, test_df, extra_feature_columns, seed):
    if model_name == "linear_baseline_env":
        params = fit_fn(train_df, target_col=TARGET_COL, extra_feature_columns=extra_feature_columns)
        predicted = predict_fn(test_df, params)
    else:
        model, encoded_column_names, feature_columns = fit_fn(
            train_df, target_col=TARGET_COL, seed=seed, extra_feature_columns=extra_feature_columns,
        )
        predicted = predict_fn(test_df, model, encoded_column_names, feature_columns=feature_columns)
    return evaluate(test_df[TARGET_COL], predicted)


def load_full_rows_with_split(cohort, split_assignment):
    # build_split_for_cohort()'s own returned table is the minimal split-building table (no
    # management columns) -- the real baselines pull the full feature set from model_table.parquet
    # separately (see run_baselines.py's load_train_rows(), which this mirrors but keeps both
    # train AND test rows, not just train).
    table = load_model_table(cohort)
    filtered_table = filter_data(table, maturity_age_min=MATURITY_AGE_MIN_DEFAULT)
    merged = filtered_table.merge(split_assignment, on=["identification", "LiDAR_year"], how="inner")
    assert len(merged) == len(filtered_table), (
        "Row count changed after merging with the split assignment -- source tables disagree "
        "on which rows survive filtering."
    )
    return merged


def run_for_cohort(cohort, split_type, feature_set, split_seed=SEED, k_folds=5, held_out_fold=0):
    feature_columns = ENV_TERRAIN_FEATURE_SETS[feature_set]
    name_suffix = f"_fold{held_out_fold}" if split_type == "spatial_block_kfold" else ""

    _, split_assignment = build_split_for_cohort(
        cohort, split_type, split_seed=split_seed, maturity_age_min=MATURITY_AGE_MIN_DEFAULT,
        name_suffix=name_suffix, k_folds=k_folds, held_out_fold=held_out_fold,
    )
    filtered_df = load_full_rows_with_split(cohort, split_assignment)
    filtered_df = merge_environmental_features(filtered_df, feature_columns, cohort)

    train_df = filtered_df[filtered_df["split"] == "train"]
    test_df = filtered_df[filtered_df["split"] == "test"]
    print(f"{cohort} {split_type} {feature_set}: train={len(train_df)} test={len(test_df)}")

    results = {}
    for model_name, fit_fn, predict_fn in [
        ("linear_baseline_env", fit_linear_baseline, predict_linear_baseline),
        ("rf_baseline_env", fit_rf_baseline, predict_rf_baseline),
        ("xgb_baseline_env", fit_xgb_baseline, predict_xgb_baseline),
    ]:
        metrics = run_one_model(model_name, fit_fn, predict_fn, train_df, test_df, feature_columns, split_seed)
        results[model_name] = metrics
        print(f"  {model_name}: r2={metrics['r2']:.4f} mae={metrics['mae']:.4f} n={metrics['n']}")

        out_dir = output_dir(f"{model_name}_{feature_set}{name_suffix}", cohort, split_type=split_type)
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["4survey", "6survey"], required=True)
    parser.add_argument(
        "--split-type", choices=["plot_level", "spatial_block_kfold"], required=True,
        help="Only these two are supported here -- plot_level as the cheap first-pass check, "
             "spatial_block_kfold to match E6_stage_sweep's own rigor level directly.",
    )
    parser.add_argument(
        "--feature-set",
        choices=[
            "stage1_terrain", "stage2_terrain_wind", "stage4_all_environmental",
            # Added 2026-08-10: the new rank-aggregate, VIF-screened RQ1 tiers (see
            # documentation/methodlogy_env_setpick.md) -- lets this script's baseline-vs-neural
            # comparison run on the SAME feature membership as the current RQ1 DNN/PINN sweep,
            # not just the older stage1-4 tiers. Both left in choices -- old stage-tier numbers
            # already cited (TEMP_baseline_env_results_2026-08-09.tex) stay reproducible.
            "nested_set2_top10", "nested_set3_gated_terrain_wind_vif", "nested_set4_gated_all_vif",
        ],
        required=True, help="Matches E6_stage_sweep's own three tiers exactly (stage3 excluded there "
                             "too -- identical column list to stage4), or the new nested_set* tiers "
                             "for a controlled comparison against the current RQ1 sweep.",
    )
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    args = parser.parse_args()

    run_for_cohort(
        args.cohort, args.split_type, args.feature_set,
        split_seed=args.split_seed, k_folds=args.n_folds, held_out_fold=args.fold_index,
    )


if __name__ == "__main__":
    main()

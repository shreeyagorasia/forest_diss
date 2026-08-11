# Run as: python -m models.growth_curve_attribution.compute_rq3_shap --cohort 4survey --set-name nested_set4_gated_all_vif
#
# Per-plot, per-fold SHAP values for RQ3's XGBoost model on local_y_max_difference -- the
# capability nothing else in RQ3's toolkit gives: EN coefficients and XGBoost gain importance are
# both GLOBAL (one number per variable, averaged over every plot); this is PER-PLOT, so it can
# actually answer "why is this specific plot's prediction what it is" -- the outlier-diagnosis
# question RQ3 cares about, not just "which variable matters on average."
#
# REFITS the 5 per-fold XGBoost models (via run_columns(), same call already used for the
# headline R2/coefficient numbers) rather than loading a saved checkpoint -- run_columns()'s
# underlying run_spatial_cv() never persisted per-fold XGBoost models to disk in the first place
# (confirmed 2026-08-11 by listing the actual output folder: no xgboost_model.json anywhere,
# only the already-extracted coefficients/importances). Refitting is deterministic given the same
# seed, so this reproduces the EXACT same models that already produced the coefficient/importance
# tables -- not a new experiment, just recomputing something that was fit once and discarded.
# Cheap (XGBoost fits in seconds) and entirely local -- no cluster time, no new rsync.

import argparse

import pandas as pd

from models.common.saving import model_output_dir
from models.growth_curve_attribution.broad_environmental_check import run_columns
from models.xgb_environmental.feature_set_builder import load_feature_set
from models.xgb_environmental.xgb_environmental import compute_shap_values_for_columns

MODEL_NAME = "rq3_en_xgb"


def run_one_set(cohort, set_name, k_folds=5, seed=42):
    print(f"===== RQ3 SHAP: {cohort} / {set_name} =====")
    raw_columns = load_feature_set("RSQ3", set_name)
    bare_columns = list(dict.fromkeys(column.split("=")[0] for column in raw_columns))
    continuous_requested = [column for column in raw_columns if "=" not in column]

    # run_columns() itself resolves the exact model_columns list (continuous + present/zero-filled
    # dummies) internally -- rather than duplicate that resolution here, results_df/predictions
    # are thrown away (already have them from the original run) and only fold_models is used.
    _, _, _, fold_models = run_columns(cohort, raw_columns, k=k_folds, seed=seed)

    shap_rows = []
    for fold_entry in fold_models:
        fold = fold_entry["fold"]
        xgb_model = fold_entry["xgboost_model"]
        held_out_features = fold_entry["held_out_features"]
        feature_columns = [column for column in held_out_features.columns if column != "identification"]

        shap_df = compute_shap_values_for_columns(xgb_model, held_out_features, feature_columns)
        shap_df.insert(1, "fold", fold)
        shap_rows.append(shap_df)
        print(f"  fold {fold}: SHAP computed for {len(shap_df):,} held-out plots, {len(feature_columns)} features")

    all_shap = pd.concat(shap_rows, ignore_index=True)
    output_dir = model_output_dir(f"{MODEL_NAME}_{set_name}", cohort, split_type="spatial_block_kfold")
    output_path = output_dir / "xgboost_shap_values.csv"
    all_shap.to_csv(output_path, index=False)
    print(f"  Saved -> {output_path}")
    return all_shap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="4survey", choices=["4survey", "6survey"])
    parser.add_argument(
        "--set-name", default="nested_set4_gated_all_vif",
        choices=["nested_set2_top10", "nested_set3_gated_terrain_wind_vif", "nested_set4_gated_all_vif", "nested_set5_all_ungated_vif"],
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_one_set(args.cohort, args.set_name, k_folds=args.k_folds, seed=args.seed)


if __name__ == "__main__":
    main()

# Run as: python -m models.spatial_attribution.compute_rq2_shap --cohort 4survey --set-name nested_set2_top10 --fold-index 0
#
# Per-plot SHAP values for RQ2b's XGBoost model on mean_cr_residual. Cheaper than RQ3's version
# of this: RQ2's fit step (run_rq2_attribution.py) already saves xgboost_model.json per fold, so
# this is a pure reload + compute, no refitting at all (unlike RQ3, whose run_spatial_cv() never
# persisted per-fold models to disk in the first place). Reuses the already-generic
# compute_shap_values_for_columns() built for RQ3. No new SHAP code needed, only this wiring.
#
# Mirrors evaluate_rq2_attribution.py's own split-re-derivation exactly (same cohort/split_seed/
# fold-dependent logic) so the SHAP rows are computed against the identical test population the
# headline R2 numbers already use.

import argparse
import json

import xgboost as xgb

from models.common.saving import model_output_dir
from models.common.splits import (
    DEFAULT_K_FOLDS,
    SPATIAL_BLOCK_COL,
    SPATIAL_BUFFER_METRES,
    SPLIT_SEED,
    spatial_block_split,
    spatial_kfold_split,
)
from models.elasticnet_environmental.elasticnet_environmental import drop_rows_with_missing_features
from models.xgb_environmental.data import load_plots_for_cohort
from models.xgb_environmental.xgb_environmental import compute_shap_values_for_columns

MODEL_NAME = "rq2_attribution"


def compute_one_set(
    cohort, set_name, split_type="spatial_block_kfold", split_seed=SPLIT_SEED,
    held_out_fold=0, k_folds=DEFAULT_K_FOLDS,
):
    fold_suffix = f", fold={held_out_fold}/{k_folds}" if split_type == "spatial_block_kfold" else ""
    print(f"===== RQ2 SHAP: {cohort} / {set_name} ({split_type}{fold_suffix}) =====")

    run_name = f"{MODEL_NAME}_{set_name}"
    if split_type == "spatial_block_kfold":
        output_dir = model_output_dir(run_name, cohort, f"fold_{held_out_fold}", split_type=split_type)
    else:
        output_dir = model_output_dir(run_name, cohort, split_type=split_type)

    with open(output_dir / "feature_columns.json") as f:
        feature_columns = json.load(f)

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(str(output_dir / "xgboost_model.json"))

    plots_df = load_plots_for_cohort(cohort).copy()
    if split_type == "spatial_block_kfold":
        plots_df["split"] = spatial_kfold_split(
            plots_df, block_col=SPATIAL_BLOCK_COL, k=k_folds, held_out_fold=held_out_fold,
            buffer_distance=SPATIAL_BUFFER_METRES, seed=split_seed,
        )
    else:
        plots_df["split"] = spatial_block_split(
            plots_df, block_col=SPATIAL_BLOCK_COL, buffer_distance=SPATIAL_BUFFER_METRES, seed=split_seed,
        )
    test_df = plots_df[plots_df["split"] == "test"]
    test_df = drop_rows_with_missing_features(test_df, feature_columns)
    print(f"  test rows: {len(test_df):,}  features: {len(feature_columns)}")

    shap_df = compute_shap_values_for_columns(xgb_model, test_df, feature_columns)
    if split_type == "spatial_block_kfold":
        shap_df.insert(1, "fold", held_out_fold)
    output_path = output_dir / "xgboost_shap_values.csv"
    shap_df.to_csv(output_path, index=False)
    print(f"  Saved -> {output_path}")
    return shap_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="4survey", choices=["4survey", "6survey"])
    parser.add_argument(
        "--set-name", default="nested_set2_top10",
        choices=["nested_set2_top10", "nested_set3_gated_terrain_wind_vif", "nested_set4_gated_all_vif", "nested_set5_all_ungated_vif"],
    )
    parser.add_argument("--split-type", default="spatial_block_kfold", choices=["spatial_block", "spatial_block_kfold"])
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_K_FOLDS)
    args = parser.parse_args()

    compute_one_set(
        args.cohort, args.set_name, split_type=args.split_type, split_seed=args.split_seed,
        held_out_fold=args.fold_index, k_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()

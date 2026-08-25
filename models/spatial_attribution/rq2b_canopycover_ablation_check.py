# Run as: PYTHONPATH=. .venv/bin/python -m models.spatial_attribution.rq2b_canopycover_ablation_check
# Ad-hoc re-verification of the CanopyCover-dropped ablation (TEMP_rq2_attribution_results_2026-08-11.tex),
# reusing the exact production fit/predict/SHAP functions. Only the driving loop (drop CanopyCover,
# loop over 5 folds) is new. Matches this project's own "ad-hoc TEMP check" convention.
import json
import numpy as np
import pandas as pd

from models.common.metrics import compute_metrics
from models.common.splits import DEFAULT_K_FOLDS, SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, SPLIT_SEED, spatial_kfold_split
from models.elasticnet_environmental.elasticnet_environmental import drop_rows_with_missing_features
from models.elasticnet_environmental.elasticnet_environmental import fit_with_columns as en_fit_with_columns
from models.elasticnet_environmental.elasticnet_environmental import predict_with_columns as en_predict_with_columns
from models.xgb_environmental.data import load_plots_for_cohort
from models.xgb_environmental.feature_set_builder import load_feature_set
from models.xgb_environmental.xgb_environmental import fit_with_columns as xgb_fit_with_columns
from models.xgb_environmental.xgb_environmental import predict_with_columns as xgb_predict_with_columns
from models.xgb_environmental.xgb_environmental import compute_shap_values_for_columns

TARGET_COLUMN = "mean_cr_residual"
COHORT = "4survey"

feature_columns_full = load_feature_set("RSQ2", "nested_set4_gated_all_vif")
feature_columns_ablated = [c for c in feature_columns_full if c != "CanopyCover"]
print(f"Full Set4: {len(feature_columns_full)} cols. Ablated (no CanopyCover): {len(feature_columns_ablated)} cols.")

plots_df_base = load_plots_for_cohort(COHORT)

en_r2_per_fold, xgb_r2_per_fold = [], []
all_test_shap = []

for fold in range(DEFAULT_K_FOLDS):
    plots_df = plots_df_base.copy()
    plots_df["split"] = spatial_kfold_split(
        plots_df, block_col=SPATIAL_BLOCK_COL, k=DEFAULT_K_FOLDS, held_out_fold=fold,
        buffer_distance=SPATIAL_BUFFER_METRES, seed=SPLIT_SEED,
    )
    train_df = drop_rows_with_missing_features(plots_df[plots_df["split"] == "train"], feature_columns_ablated)
    val_df = drop_rows_with_missing_features(plots_df[plots_df["split"] == "val"], feature_columns_ablated)
    test_df = drop_rows_with_missing_features(plots_df[plots_df["split"] == "test"], feature_columns_ablated)

    en_fitted = en_fit_with_columns(train_df, feature_columns_ablated, target_col=TARGET_COLUMN)
    xgb_model = xgb_fit_with_columns(
        train_df, feature_columns_ablated, val_df=val_df, target_col=TARGET_COLUMN,
        n_jobs=1, n_estimators=500, max_depth=4, learning_rate=0.04,
    )

    en_pred = en_predict_with_columns(test_df, en_fitted, feature_columns_ablated)
    xgb_pred = xgb_predict_with_columns(test_df, xgb_model, feature_columns_ablated)

    en_r2 = compute_metrics(test_df[TARGET_COLUMN], en_pred)["r2"]
    xgb_r2 = compute_metrics(test_df[TARGET_COLUMN], xgb_pred)["r2"]
    en_r2_per_fold.append(en_r2)
    xgb_r2_per_fold.append(xgb_r2)
    print(f"  fold {fold}: EN R2={en_r2:.4f}  XGB R2={xgb_r2:.4f}  (train={len(train_df):,} test={len(test_df):,})")

    shap_df = compute_shap_values_for_columns(xgb_model, test_df, feature_columns_ablated)
    all_test_shap.append(shap_df)

en_r2_per_fold, xgb_r2_per_fold = np.array(en_r2_per_fold), np.array(xgb_r2_per_fold)
print(f"\nWITHOUT CanopyCover. Elastic Net: {en_r2_per_fold.mean():.3f}+/-{en_r2_per_fold.std():.3f}")
print(f"WITHOUT CanopyCover. XGBoost:     {xgb_r2_per_fold.mean():.3f}+/-{xgb_r2_per_fold.std():.3f}")
print("(compare to TEMP note: EN 0.350->0.231, XGBoost 0.388->0.249)")

all_shap = pd.concat(all_test_shap, ignore_index=True)
mean_abs_shap = all_shap[feature_columns_ablated].abs().mean().sort_values(ascending=False)
print("\nMean |SHAP|, without CanopyCover, top 6:")
print(mean_abs_shap.head(6))

results = {
    "en_r2_without_canopy_per_fold": en_r2_per_fold.tolist(),
    "xgb_r2_without_canopy_per_fold": xgb_r2_per_fold.tolist(),
    "mean_abs_shap_without_canopy": mean_abs_shap.to_dict(),
}
with open("outputs/spatial_block_kfold/rq2_attribution_nested_set4_no_canopycover/4survey/ablation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved outputs/spatial_block_kfold/rq2_attribution_nested_set4_no_canopycover/4survey/ablation_results.json")

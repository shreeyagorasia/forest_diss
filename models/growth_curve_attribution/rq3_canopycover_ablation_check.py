# Run as: PYTHONPATH=. .venv/bin/python -m models.growth_curve_attribution.rq3_canopycover_ablation_check
#
# Re-verification of the CanopyCover-dropped ablation for Q2's OWN target (local_y_max_difference,
# Set4, 4survey) -- the equivalent check already exists for Q1's target (mean_cr_residual, see
# models/spatial_attribution/rq2b_canopycover_ablation_check.py) but had never been run for Q2's
# own model. Reuses the exact production fit/SHAP functions (run_columns, compute_shap_values_for_columns)
# with CanopyCover removed from the Set4 feature list -- not a new experiment, just the same
# already-used pipeline with one column dropped.

import json

import numpy as np
import pandas as pd

from models.growth_curve_attribution.broad_environmental_check import run_columns
from models.xgb_environmental.feature_set_builder import load_feature_set
from models.xgb_environmental.xgb_environmental import compute_shap_values_for_columns

COHORT = "4survey"
SET_NAME = "nested_set4_gated_all_vif"

raw_columns = load_feature_set("RSQ3", SET_NAME)
ablated_columns = [c for c in raw_columns if c != "CanopyCover"]
print(f"Full Set4: {len(raw_columns)} cols. Ablated (no CanopyCover): {len(ablated_columns)} cols.")

results_df, predictions, fold_counts, fold_models = run_columns(COHORT, ablated_columns, k=5, seed=42)
print("\nWITHOUT CanopyCover -- pooled results:")
print(results_df[["method", "pooled_r2", "per_fold_r2_mean", "per_fold_r2_std"]].to_string(index=False))

shap_rows = []
for fold_entry in fold_models:
    fold = fold_entry["fold"]
    xgb_model = fold_entry["xgboost_model"]
    held_out_features = fold_entry["held_out_features"]
    feature_columns = [c for c in held_out_features.columns if c != "identification"]
    shap_df = compute_shap_values_for_columns(xgb_model, held_out_features, feature_columns)
    shap_df.insert(1, "fold", fold)
    shap_rows.append(shap_df)

all_shap = pd.concat(shap_rows, ignore_index=True)
shap_feature_cols = [c for c in all_shap.columns if c not in ("identification", "fold")]
mean_abs_shap = all_shap[shap_feature_cols].abs().mean().sort_values(ascending=False)
print("\nTop 6 mean |SHAP|, WITHOUT CanopyCover (Q2's own target):")
print(mean_abs_shap.head(6))

output = {
    "results_without_canopy": results_df.to_dict(orient="records"),
    "mean_abs_shap_without_canopy": mean_abs_shap.to_dict(),
}
with open("outputs/spatial_block_kfold/rq3_en_xgb_nested_set4_no_canopycover_ablation.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print("\nSaved outputs/spatial_block_kfold/rq3_en_xgb_nested_set4_no_canopycover_ablation.json")

# Run as: PYTHONPATH=. .venv/bin/python -m models.growth_curve_attribution.rq3_no_canopy_shap_dependence_check
#
# Quick sanity check for the "is GNNWR's local-linear assumption hiding non-linear environmental
# signal" question: plots each plot's SHAP value against its own raw feature value, for the three
# variables whose SHAP rose most once CanopyCover was dropped (windward_topex, slope_degrees,
# tas_mean. See rq3_canopycover_ablation_check.py's own saved ablation JSON). A flat/noisy cloud
# argues against a real relationship, linear or not; a clear curve/threshold shape would argue for
# real (if non-linear) global signal that a local-LINEAR model like GNNWR still could not fit
# correctly even though it is spatially aware. Global check only (not spatial). Same re-run as
# rq3_canopycover_ablation_check.py, just keeping the per-plot SHAP+feature table instead of only
# the aggregated mean.

import matplotlib.pyplot as plt
import pandas as pd

from models.growth_curve_attribution.broad_environmental_check import run_columns
from models.xgb_environmental.feature_set_builder import load_feature_set
from models.xgb_environmental.xgb_environmental import compute_shap_values_for_columns

COHORT = "4survey"
SET_NAME = "nested_set4_gated_all_vif"
CHECK_VARS = ["windward_topex", "slope_degrees", "tas_mean"]

raw_columns = load_feature_set("RSQ3", SET_NAME)
ablated_columns = [c for c in raw_columns if c != "CanopyCover"]

print("Refitting WITHOUT CanopyCover (same as rq3_canopycover_ablation_check.py) to get per-plot SHAP+feature pairs...")
results_df, predictions, fold_counts, fold_models = run_columns(COHORT, ablated_columns, k=5, seed=42)

shap_rows, feature_rows = [], []
for fold_entry in fold_models:
    fold = fold_entry["fold"]
    xgb_model = fold_entry["xgboost_model"]
    held_out_features = fold_entry["held_out_features"]
    feature_columns = [c for c in held_out_features.columns if c != "identification"]
    shap_df = compute_shap_values_for_columns(xgb_model, held_out_features, feature_columns)
    shap_df.insert(1, "fold", fold)
    shap_rows.append(shap_df)
    feature_rows.append(held_out_features.assign(fold=fold))

all_shap = pd.concat(shap_rows, ignore_index=True)
all_features = pd.concat(feature_rows, ignore_index=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, var in zip(axes, CHECK_VARS):
    feature_values = all_features.set_index(["identification", "fold"])[var]
    shap_values = all_shap.set_index(["identification", "fold"])[var]
    merged = pd.concat([feature_values.rename("feature"), shap_values.rename("shap")], axis=1).dropna()
    ax.scatter(merged["feature"], merged["shap"], s=4, alpha=0.15, color="#E07856")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel(var)
    ax.set_ylabel("SHAP value (m)")
    ax.set_title(f"{var}\n(WITHOUT CanopyCover, Set4, 4survey, pooled 5 folds)", fontsize=9.5)

fig.suptitle("SHAP dependence. Is there visible non-linear structure, or just noise?", fontsize=12)
plt.tight_layout()
out_path = "figures/fig_results/q2_no_canopy_shap_dependence_check.png"
fig.savefig(out_path, dpi=150)
print(f"\nSaved {out_path}")

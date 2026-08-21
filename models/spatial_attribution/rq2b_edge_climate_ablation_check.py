# Run as: PYTHONPATH=. .venv/bin/python -m models.spatial_attribution.rq2b_edge_climate_ablation_check
#
# Tests a % THEORY, UNTESTED note in results_q1_21_08_0754pm.tex: XGBoost's residual spatial
# autocorrelation (Moran's I) drops a lot at Set2, partially rebounds at Set3, then drops again at
# Set4. Set2 and Set4 both carry dist_to_road/chelsa_bio12_precip_mm/tas_mean; Set3 (terrain/wind
# only) does not. If XGBoost is using those three variables to remove spatial clustering, dropping
# them from Set4 should push its residual Moran's I back up towards Set3's level (0.077), not stay
# near Set4's own level (0.054).
#
# Same pattern as rq2b_canopycover_ablation_check.py -- 5-fold spatial-block refit, only the
# dropped-column list changes -- but this one also needs residual Moran's I (not just R2), so it
# pools test-set predictions with plot coordinates the same way the Q1 figure notebook does.

import numpy as np
import pandas as pd

from models.common.geo import load_plot_coordinates
from models.common.splits import DEFAULT_K_FOLDS, SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, SPLIT_SEED, spatial_kfold_split
from models.growth_curve_attribution.residual_spatial_autocorrelation_check import compute_residual_morans_i
from models.xgb_environmental.data import load_plots_for_cohort
from models.xgb_environmental.feature_set_builder import load_feature_set
from models.xgb_environmental.xgb_environmental import fit_with_columns as xgb_fit_with_columns
from models.xgb_environmental.xgb_environmental import predict_with_columns as xgb_predict_with_columns

TARGET_COLUMN = "mean_cr_residual"
COHORT = "4survey"
DROPPED_COLUMNS = ["dist_to_road", "chelsa_bio12_precip_mm", "tas_mean"]

feature_columns_full = load_feature_set("RSQ2", "nested_set4_gated_all_vif")
feature_columns_ablated = [c for c in feature_columns_full if c not in DROPPED_COLUMNS]
print(f"Full Set4: {len(feature_columns_full)} cols. Ablated (no {DROPPED_COLUMNS}): {len(feature_columns_ablated)} cols.")

plots_df_base = load_plots_for_cohort(COHORT)
coordinates = load_plot_coordinates()

xgb_r2_per_fold = []
test_frames = []

for fold in range(DEFAULT_K_FOLDS):
    plots_df = plots_df_base.copy()
    plots_df["split"] = spatial_kfold_split(
        plots_df, block_col=SPATIAL_BLOCK_COL, k=DEFAULT_K_FOLDS, held_out_fold=fold,
        buffer_distance=SPATIAL_BUFFER_METRES, seed=SPLIT_SEED,
    )
    train_df = plots_df[plots_df["split"] == "train"].dropna(subset=feature_columns_ablated + [TARGET_COLUMN])
    val_df = plots_df[plots_df["split"] == "val"].dropna(subset=feature_columns_ablated + [TARGET_COLUMN])
    test_df = plots_df[plots_df["split"] == "test"].dropna(subset=feature_columns_ablated + [TARGET_COLUMN])

    xgb_model = xgb_fit_with_columns(
        train_df, feature_columns_ablated, val_df=val_df, target_col=TARGET_COLUMN,
        n_jobs=1, n_estimators=500, max_depth=4, learning_rate=0.04,
    )
    xgb_pred = xgb_predict_with_columns(test_df, xgb_model, feature_columns_ablated)

    test_df = test_df.copy()
    test_df["xgboost_predicted"] = xgb_pred
    test_frames.append(test_df[["identification", TARGET_COLUMN, "xgboost_predicted"]])

    fold_r2 = 1 - np.sum((test_df[TARGET_COLUMN] - xgb_pred) ** 2) / np.sum(
        (test_df[TARGET_COLUMN] - test_df[TARGET_COLUMN].mean()) ** 2
    )
    xgb_r2_per_fold.append(fold_r2)
    print(f"  fold {fold}: XGB R2={fold_r2:.4f}  (train={len(train_df):,} test={len(test_df):,})")

xgb_r2_per_fold = np.array(xgb_r2_per_fold)
print(f"\nWITHOUT dist_to_road/precip/tas_mean -- XGBoost R2: {xgb_r2_per_fold.mean():.3f}+/-{xgb_r2_per_fold.std():.3f}")

pooled = pd.concat(test_frames, ignore_index=True).merge(coordinates, on="identification", how="left")
pooled["xgb_residual"] = pooled[TARGET_COLUMN] - pooled["xgboost_predicted"]

xgb_i, xgb_p, xgb_n, xgb_range_m, xgb_status = compute_residual_morans_i(pooled["x"], pooled["y"], pooled["xgb_residual"])
print(f"\nXGBoost residual Moran's I WITHOUT dist_to_road/precip/tas_mean: {xgb_i:.3f} (p={xgb_p:.3f}, n={xgb_n}, range={xgb_range_m:.0f}m, status={xgb_status})")
print("Compare: Set4 (with these vars) = 0.054, Set3 (terrain/wind only, never had these vars) = 0.077")

import json
import os

os.makedirs("outputs/spatial_block_kfold/rq2_attribution_nested_set4_no_edge_climate/4survey", exist_ok=True)
with open("outputs/spatial_block_kfold/rq2_attribution_nested_set4_no_edge_climate/4survey/ablation_results.json", "w") as f:
    json.dump({
        "dropped_columns": DROPPED_COLUMNS,
        "xgb_r2_per_fold": xgb_r2_per_fold.tolist(),
        "xgb_morans_i": xgb_i,
        "xgb_morans_i_p": xgb_p,
        "xgb_morans_i_range_m": xgb_range_m,
        "compare_set4_with_vars_morans_i": 0.054,
        "compare_set3_terrain_wind_only_morans_i": 0.077,
    }, f, indent=2)
print("\nSaved outputs/spatial_block_kfold/rq2_attribution_nested_set4_no_edge_climate/4survey/ablation_results.json")

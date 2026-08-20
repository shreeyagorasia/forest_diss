# Quicktest: does the fixed forward pass move PINN's R2 toward DNN's, or not?
# Reuses the EXACT data-loading pipeline models/pinn_env_terrain/run_pinn_env_terrain.py uses
# (Mistake #7, PLAN.md -- don't let a hand-rolled copy silently diverge from what Table 1
# actually used) via direct import, not reimplementation.
#
# Runs OLD (unfixed, original models/pinn_env_terrain/pinn_env_terrain.py) and NEW (fixed,
# pinn_env_terrain_fix.py) at IDENTICAL matched settings (Mistake #3 -- the real old-PINN number
# is a 500-epoch production run, not a fair comparison against a 40-epoch new one).
#
# 4survey only (6survey skipped for time, per PLAN.md's own quicktest scope), single
# spatial_block split (not 5-fold -- this is step 8 of PLAN.md's order, before deciding on the
# real 5-fold run), Set3 (nested_set3_gated_terrain_wind_vif, matching the actual headline
# table's feature set, confirmed via outputs/run_logs/ -- NOT the module's own different
# default). max_epochs=40, patience=10 (PLAN.md's quicktest scope).
#
# Run as: PYTHONPATH=. .venv/bin/python temp_results_pinn/pinn_env_terrain_fix/run_quicktest.py

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common.metrics import compute_metrics
from models.common.saving import load_cr_params
from models.common.splits import SPLIT_SEED
from models.common.torch_data import (
    ENV_TERRAIN_FEATURE_SETS,
    build_pair_terrain_tensor,
    build_pair_tensors,
    build_tensors,
    build_terrain_tensor,
    encode_thinning_status,
    fit_scalers,
    fit_terrain_scaler,
    load_split_table_with_terrain,
    load_trajectory_pairs,
    select_device,
)

COHORT = "4survey"
SPLIT_TYPE = "spatial_block"
FEATURE_SET_NAME = "nested_set3_gated_terrain_wind_vif"  # confirmed match to the headline table
MAX_EPOCHS = 40
PATIENCE = 10
SEED = 42

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "quicktest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {select_device()}")
device = select_device()

feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]
print(f"Feature set ({FEATURE_SET_NAME}): {feature_columns}")

cr_params = load_cr_params(COHORT, SPLIT_TYPE, split_seed=SPLIT_SEED)
print(f"Frozen CR anchor: y_max={cr_params['y_max']:.4f}, k={cr_params['k']:.6f}, p={cr_params['p']:.6f}")

split_df = load_split_table_with_terrain(COHORT, SPLIT_TYPE, feature_columns, split_seed=SPLIT_SEED)
train_df = split_df[split_df["split"] == "train"]
val_df = split_df[split_df["split"] == "val"]
test_df = split_df[split_df["split"] == "test"]
print(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

pairs_df = load_trajectory_pairs(COHORT, split_df)
print(f"trajectory pairs: {len(pairs_df):,}")

scaler_age, scaler_other_features, scaler_height = fit_scalers(train_df)
scaler_terrain = fit_terrain_scaler(train_df, feature_columns)
encoded_column_names = encode_thinning_status(train_df).columns.tolist()

age_train, other_train, target_train = build_tensors(
    train_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
)
age_val, other_val, target_val = build_tensors(
    val_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
)
age_test, other_test, target_test = build_tensors(
    test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
)
terrain_train = build_terrain_tensor(train_df, scaler_terrain, feature_columns, device)
terrain_val = build_terrain_tensor(val_df, scaler_terrain, feature_columns, device)
terrain_test = build_terrain_tensor(test_df, scaler_terrain, feature_columns, device)

pair_tensors = build_pair_tensors(
    pairs_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device
)
terrain_pairs = build_pair_terrain_tensor(pairs_df, scaler_terrain, feature_columns, device, COHORT)

n_other_features = other_train.shape[1]
n_terrain_features = terrain_train.shape[1]
print(f"n_other_features={n_other_features}  n_terrain_features={n_terrain_features}")


def unscale_height(scaled_tensor):
    return scaled_tensor.cpu().numpy().flatten() * scaler_height.scale_[0] + scaler_height.mean_[0]


results = {}

# ----- OLD (unfixed, original file) -----
print("\n" + "=" * 70)
print("OLD (unfixed): terrain never reaches the prediction")
print("=" * 70)
t0 = time.time()
from models.pinn_env_terrain.pinn_env_terrain import build_model as old_build_model
from models.pinn_env_terrain.pinn_env_terrain import fit as old_fit
from models.pinn_env_terrain.pinn_env_terrain import predict as old_predict

old_best_model, _, old_history = old_fit(
    age_train, other_train, terrain_train, target_train,
    age_val, other_val, target_val,
    pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
    n_other_features, n_terrain_features, device, SEED,
    max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
)
old_preds_scaled = old_predict(old_best_model, age_test, other_test)
old_preds = unscale_height(old_preds_scaled)
old_target = unscale_height(target_test)
old_metrics = compute_metrics(old_target, old_preds)
old_elapsed = time.time() - t0
print(f"OLD test R2={old_metrics['r2']:.4f}  RMSE={old_metrics['rmse']:.4f}  "
      f"({len(old_history)} epochs, {old_elapsed:.1f}s)")
results["old_unfixed"] = {"r2": old_metrics["r2"], "rmse": old_metrics["rmse"],
                           "n_epochs": len(old_history), "elapsed_seconds": old_elapsed}
old_history.to_csv(OUTPUT_DIR / "old_unfixed_history.csv", index=False)

# ----- NEW (fixed) -----
print("\n" + "=" * 70)
print("NEW (fixed): terrain reaches the prediction via the CR term")
print("=" * 70)
t0 = time.time()
from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_fix import fit as new_fit
from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_fix import predict as new_predict

new_best_model, _, new_history = new_fit(
    age_train, other_train, terrain_train, target_train,
    age_val, other_val, terrain_val, target_val,
    pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
    n_other_features, n_terrain_features, device, SEED,
    max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
)
new_preds_scaled = new_predict(new_best_model, age_test, other_test, terrain_test)
new_preds = unscale_height(new_preds_scaled)
new_target = unscale_height(target_test)
new_metrics = compute_metrics(new_target, new_preds)
new_elapsed = time.time() - t0
print(f"NEW test R2={new_metrics['r2']:.4f}  RMSE={new_metrics['rmse']:.4f}  "
      f"({len(new_history)} epochs, {new_elapsed:.1f}s)")
results["new_fixed"] = {"r2": new_metrics["r2"], "rmse": new_metrics["rmse"],
                         "n_epochs": len(new_history), "elapsed_seconds": new_elapsed}
new_history.to_csv(OUTPUT_DIR / "new_fixed_history.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"OLD (unfixed) test R2: {results['old_unfixed']['r2']:.4f}")
print(f"NEW (fixed)   test R2: {results['new_fixed']['r2']:.4f}")
print(f"Movement: {results['new_fixed']['r2'] - results['old_unfixed']['r2']:+.4f}")
print("(Reference only, not rerun here: production DNN R2 at 500 epochs = 0.655 on 4survey, "
      "Table tab:results-rq1. This quicktest's OLD number is NOT that production number -- "
      "it's a matched 40-epoch rerun for a fair OLD-vs-NEW comparison, per PLAN.md Mistake #3.)")

import json
with open(OUTPUT_DIR / "quicktest_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUTPUT_DIR}")

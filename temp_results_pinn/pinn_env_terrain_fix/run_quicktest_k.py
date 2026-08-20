# Quicktest, k version -- same structure and scope as run_quicktest.py (Set3, 4survey,
# spatial_block, max_epochs=40, patience=10, matched OLD-vs-NEW comparison).
# Run as: PYTHONPATH=. .venv/bin/python temp_results_pinn/pinn_env_terrain_fix/run_quicktest_k.py

import json
import sys
import time
from pathlib import Path

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
FEATURE_SET_NAME = "nested_set3_gated_terrain_wind_vif"
MAX_EPOCHS = 40
PATIENCE = 10
SEED = 42

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "quicktest"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = select_device()
print(f"Device: {device}")

feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]
cr_params = load_cr_params(COHORT, SPLIT_TYPE, split_seed=SPLIT_SEED)
print(f"Frozen CR anchor: y_max={cr_params['y_max']:.4f}, k={cr_params['k']:.6f}, p={cr_params['p']:.6f}")

split_df = load_split_table_with_terrain(COHORT, SPLIT_TYPE, feature_columns, split_seed=SPLIT_SEED)
train_df = split_df[split_df["split"] == "train"]
val_df = split_df[split_df["split"] == "val"]
test_df = split_df[split_df["split"] == "test"]
print(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

pairs_df = load_trajectory_pairs(COHORT, split_df)

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


def unscale_height(scaled_tensor):
    return scaled_tensor.cpu().numpy().flatten() * scaler_height.scale_[0] + scaler_height.mean_[0]


results = {}

print("\n" + "=" * 70)
print("OLD-k (unfixed): terrain never reaches the prediction")
print("=" * 70)
t0 = time.time()
from models.pinn_env_terrain_k.pinn_env_terrain_k import fit as old_fit
from models.pinn_env_terrain_k.pinn_env_terrain_k import predict as old_predict

old_best_model, _, old_history = old_fit(
    age_train, other_train, terrain_train, target_train,
    age_val, other_val, target_val,
    pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
    n_other_features, n_terrain_features, device, SEED,
    max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
)
old_preds = unscale_height(old_predict(old_best_model, age_test, other_test))
old_target = unscale_height(target_test)
old_metrics = compute_metrics(old_target, old_preds)
old_elapsed = time.time() - t0
print(f"OLD-k test R2={old_metrics['r2']:.4f}  RMSE={old_metrics['rmse']:.4f}  "
      f"({len(old_history)} epochs, {old_elapsed:.1f}s)")
results["old_k_unfixed"] = {"r2": old_metrics["r2"], "rmse": old_metrics["rmse"], "n_epochs": len(old_history)}
old_history.to_csv(OUTPUT_DIR / "old_k_unfixed_history.csv", index=False)

print("\n" + "=" * 70)
print("NEW-k (fixed): terrain reaches the prediction via the CR term (y_max AND k)")
print("=" * 70)
t0 = time.time()
from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_k_fix import fit as new_fit
from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_k_fix import predict as new_predict

new_best_model, _, new_history = new_fit(
    age_train, other_train, terrain_train, target_train,
    age_val, other_val, terrain_val, target_val,
    pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
    n_other_features, n_terrain_features, device, SEED,
    max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
)
new_preds = unscale_height(new_predict(new_best_model, age_test, other_test, terrain_test))
new_target = unscale_height(target_test)
new_metrics = compute_metrics(new_target, new_preds)
new_elapsed = time.time() - t0
print(f"NEW-k test R2={new_metrics['r2']:.4f}  RMSE={new_metrics['rmse']:.4f}  "
      f"({len(new_history)} epochs, {new_elapsed:.1f}s)")
results["new_k_fixed"] = {"r2": new_metrics["r2"], "rmse": new_metrics["rmse"], "n_epochs": len(new_history)}
new_history.to_csv(OUTPUT_DIR / "new_k_fixed_history.csv", index=False)

print("\n" + "=" * 70)
print("SUMMARY (k version)")
print("=" * 70)
print(f"OLD-k (unfixed) test R2: {results['old_k_unfixed']['r2']:.4f}")
print(f"NEW-k (fixed)   test R2: {results['new_k_fixed']['r2']:.4f}")
print(f"Movement: {results['new_k_fixed']['r2'] - results['old_k_unfixed']['r2']:+.4f}")
print("(Reference: production DNN R2 = 0.655 (4survey, 500 epochs). Production PINN-k R2 = "
      "0.575 (Table tab:results-rq1). OLD-k here is a matched 40-epoch rerun, not that number.)")

with open(OUTPUT_DIR / "quicktest_k_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {OUTPUT_DIR}")

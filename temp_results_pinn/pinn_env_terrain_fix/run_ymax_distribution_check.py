# One-off script: train the CORRECTED (fixed-forward-pass) plain PINN (y_max only, k fixed at
# population value) on fold 0, same settings as the already-verified PINN-k run in
# run_example_plot_curve.py, and save every test-set plot's y_max_pred.
#
# Purpose: the earlier plain-PINN y_max check used outputs/pinn_env_terrain/4survey/predictions.csv,
# which was traced back to the PRE-FIX pinn_env_terrain.py (its own forward() comment confirms the
# y_max sub-network is "never as part of this forward pass" -- exactly the bug Table 3's footnote
# describes). That earlier finding (100% of plots inflated, mean +7m) is therefore not valid. This
# reruns with the corrected pinn_env_terrain_fix.py to get a real answer.
#
# A corrected fold-0 run already happened once (temp_results_pinn/outputs/full_rerun/fold_0/,
# R2=0.674, 687s) but only saved aggregate metrics, no per-plot predictions, and no checkpoint --
# hence this rerun, not a from-scratch new experiment.
#
# Run as: PYTHONPATH=. .venv/bin/python temp_results_pinn/pinn_env_terrain_fix/run_ymax_distribution_check.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

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
from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_fix import fit, predict_y_max

COHORT = "4survey"
SPLIT_TYPE = "spatial_block_kfold"
FOLD_INDEX = 0
N_FOLDS = 5
FEATURE_SET_NAME = "nested_set3_gated_terrain_wind_vif"
MAX_EPOCHS = 500
PATIENCE = 40
SEED = 42

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "example_curve"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = select_device()
print(f"Device: {device}")

feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]
cr_params = load_cr_params(COHORT, SPLIT_TYPE, split_seed=SPLIT_SEED, held_out_fold=FOLD_INDEX)
print(f"Population CR anchor: y_max={cr_params['y_max']:.4f}  k={cr_params['k']:.6f}  p={cr_params['p']:.6f}")

split_df = load_split_table_with_terrain(
    COHORT, SPLIT_TYPE, feature_columns, split_seed=SPLIT_SEED, k_folds=N_FOLDS, held_out_fold=FOLD_INDEX,
)
train_df = split_df[split_df["split"] == "train"]
val_df = split_df[split_df["split"] == "val"]
test_df = split_df[split_df["split"] == "test"]
print(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

pairs_df = load_trajectory_pairs(COHORT, split_df)

scaler_age, scaler_other_features, scaler_height = fit_scalers(train_df)
scaler_terrain = fit_terrain_scaler(train_df, feature_columns)
encoded_column_names = encode_thinning_status(train_df).columns.tolist()

age_train, other_train, target_train = build_tensors(
    train_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
age_val, other_val, target_val = build_tensors(
    val_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
terrain_train = build_terrain_tensor(train_df, scaler_terrain, feature_columns, device)
terrain_val = build_terrain_tensor(val_df, scaler_terrain, feature_columns, device)
terrain_test = build_terrain_tensor(test_df, scaler_terrain, feature_columns, device)

pair_tensors = build_pair_tensors(
    pairs_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
terrain_pairs = build_pair_terrain_tensor(pairs_df, scaler_terrain, feature_columns, device, COHORT)

n_other_features = other_train.shape[1]
n_terrain_features = terrain_train.shape[1]

print("\nTraining fixed plain PINN (y_max only, k fixed), fold 0, production settings...")
model, _, history = fit(
    age_train, other_train, terrain_train, target_train,
    age_val, other_val, terrain_val, target_val,
    pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
    n_other_features, n_terrain_features, device, SEED,
    max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
)
print(f"Trained {len(history)} epochs (early-stopped).")

y_max_per_row = predict_y_max(model, terrain_test, cr_params["y_max"]).cpu().numpy().flatten()
test_df = test_df.copy()
test_df["y_max_pred"] = y_max_per_row

out_path = OUTPUT_DIR / "plain_pinn_fixed_test_set_predictions.csv"
test_df[["identification", "Age", "y_max_pred"]].to_csv(out_path, index=False)
print(f"Saved -> {out_path}")

plot_level = test_df.groupby("identification")["y_max_pred"].first()
diff = plot_level - cr_params["y_max"]
print(f"\nn plots: {len(plot_level)}")
print(diff.describe())
print(f"\nImplausible (<5m or >70m): {((plot_level<5)|(plot_level>70)).sum()} / {len(plot_level)}")
print(f"Plots with y_max_pred ABOVE population: {(diff>0).mean()*100:.1f}%")
print(f"Plots with y_max_pred BELOW population: {(diff<0).mean()*100:.1f}%")

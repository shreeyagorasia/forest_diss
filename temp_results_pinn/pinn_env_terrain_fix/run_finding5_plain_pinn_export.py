# Finding 5 (errors by age/height, DNN vs. PINN vs. PINN-k) needs full per-row predictions for
# all three models. DNN and PINN-k already have this saved; plain PINN's existing export
# (run_ymax_distribution_check.py -> plain_pinn_fixed_test_set_predictions.csv) only has
# identification/Age/y_max_pred, not the actual predicted height or residual -- this script adds
# that, matching DNN's predictions.csv schema (identification, Age, observed_top_height,
# predicted_top_height, residual) so the three files can be compared directly.
#
# Fold 0 only, same trusted config as every other Set3 PINN run this session (lr=0.0001,
# weight_decay=1e-5, batch_size=256, physics_weight=1.0). No checkpoint exists for the trusted
# run, so this retrains -- same reasoning as every other diagnostic script this session.
#
# Run: PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_finding5_plain_pinn_export.py

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
from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_fix import fit, predict

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


def unscale(scaled_tensor, scaler):
    return scaled_tensor.cpu().numpy().flatten() * scaler.scale_[0] + scaler.mean_[0]


def main():
    device = select_device()
    print(f"Device: {device}")

    feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]
    cr_params = load_cr_params(COHORT, SPLIT_TYPE, split_seed=SPLIT_SEED, held_out_fold=FOLD_INDEX)

    split_df = load_split_table_with_terrain(
        COHORT, SPLIT_TYPE, feature_columns, split_seed=SPLIT_SEED, k_folds=N_FOLDS, held_out_fold=FOLD_INDEX,
    )
    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"].reset_index(drop=True)
    print(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    pairs_df = load_trajectory_pairs(COHORT, split_df)

    scaler_age, scaler_other_features, scaler_height = fit_scalers(train_df)
    scaler_terrain = fit_terrain_scaler(train_df, feature_columns)
    encoded_column_names = encode_thinning_status(train_df).columns.tolist()

    age_train, other_train, target_train = build_tensors(
        train_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
    age_val, other_val, target_val = build_tensors(
        val_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
    age_test, other_test, target_test = build_tensors(
        test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
    terrain_train = build_terrain_tensor(train_df, scaler_terrain, feature_columns, device)
    terrain_val = build_terrain_tensor(val_df, scaler_terrain, feature_columns, device)
    terrain_test = build_terrain_tensor(test_df, scaler_terrain, feature_columns, device)

    pair_tensors = build_pair_tensors(
        pairs_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
    terrain_pairs = build_pair_terrain_tensor(pairs_df, scaler_terrain, feature_columns, device, COHORT)

    n_other_features = other_train.shape[1]
    n_terrain_features = terrain_train.shape[1]

    print("\nTraining plain PINN (y_max-only fix), fold 0, trusted Set3 config...")
    model, _, history = fit(
        age_train, other_train, terrain_train, target_train,
        age_val, other_val, terrain_val, target_val,
        pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
        n_other_features, n_terrain_features, device, SEED,
        max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
    )
    print(f"Trained {len(history)} epochs (early-stopped).")

    predicted_scaled = predict(model, age_test, other_test, terrain_test)
    predicted_top_height = unscale(predicted_scaled, scaler_height)
    observed_top_height = unscale(target_test, scaler_height)
    residual = predicted_top_height - observed_top_height

    out_df = test_df[["identification", "blk", "cpmt", "Age"]].copy()
    out_df["observed_top_height"] = observed_top_height
    out_df["predicted_top_height"] = predicted_top_height
    out_df["residual"] = residual

    out_path = OUTPUT_DIR / "plain_pinn_fixed_full_predictions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    from models.common.metrics import compute_metrics
    metrics = compute_metrics(observed_top_height, predicted_top_height)
    print(f"\nSanity check -- fold 0 test R2: {metrics['r2']:.4f} (should be close to the trusted 0.631)")


if __name__ == "__main__":
    main()

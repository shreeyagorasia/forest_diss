# Generalizes run_ymax_distribution_check.py (fold-0-only) to any fold, and saves blk/cpmt
# alongside identification/Age/y_max_pred -- needed to pool all 5 folds' test-set predictions
# into one full-forest map (Q3 redraft, Block D main-text figure). Each compartment is the
# test set in exactly one of the 5 folds under spatial_block_kfold, so pooling all 5 folds'
# TEST predictions gives every plot in the forest exactly one genuine held-out prediction --
# same pattern Q1's own maps already use (pooling test predictions across folds), not a new
# or riskier approach than what's already established.
#
# Fold 0 already exists (temp_results_pinn/outputs/example_curve/plain_pinn_fixed_test_set_predictions.csv,
# no blk/cpmt saved) -- this script covers folds 1-4, and could re-do fold 0 too if needed.
#
# Run (any fold): PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_ymax_population_check_allfolds.py --fold-index 1
#
# Saves to
# temp_results_pinn/outputs/CORRECTED_2026-08-23_mechanism_checks/ymax_population_check/fold_<i>/predictions.csv

import argparse
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
N_FOLDS = 5
FEATURE_SET_NAME = "nested_set3_gated_terrain_wind_vif"
MAX_EPOCHS = 500
PATIENCE = 40
SEED = 42

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "CORRECTED_2026-08-23_mechanism_checks" / "ymax_population_check"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-index", type=int, required=True, help="0-4, which spatial_block_kfold fold to hold out as test.")
    args = parser.parse_args()
    fold_index = args.fold_index

    fold_dir = OUTPUT_DIR / f"fold_{fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    out_path = fold_dir / "predictions.csv"
    if out_path.exists():
        print(f"{out_path} already exists -- delete it first to redo this fold. Exiting without retraining.")
        return

    device = select_device()
    print(f"Device: {device}  Fold: {fold_index}")

    feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]
    cr_params = load_cr_params(COHORT, SPLIT_TYPE, split_seed=SPLIT_SEED, held_out_fold=fold_index)
    print(f"Population CR anchor: y_max={cr_params['y_max']:.4f}  k={cr_params['k']:.6f}  p={cr_params['p']:.6f}")

    split_df = load_split_table_with_terrain(
        COHORT, SPLIT_TYPE, feature_columns, split_seed=SPLIT_SEED, k_folds=N_FOLDS, held_out_fold=fold_index,
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

    print("\nTraining fixed plain PINN (y_max only, k fixed), production settings...")
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
    test_df["fold_index"] = fold_index
    test_df["population_y_max"] = cr_params["y_max"]

    test_df[["identification", "blk", "cpmt", "Age", "y_max_pred", "population_y_max", "fold_index"]].to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()

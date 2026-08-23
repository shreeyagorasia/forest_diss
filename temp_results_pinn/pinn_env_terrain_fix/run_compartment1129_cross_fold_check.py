# Cross-fold check for the compartment-1129 y_max inflation finding
# (temp_results_pinn/RESULTS_TABLE.md, section 9). Compartment 1129 (blk=21) was found to be
# the source of 31/40 of the most-inflated y_max predictions, discovered on fold 0's HELD-OUT
# test set. Question this answers: is that a genuine site effect (the model predicts an
# inflated y_max for 1129 even when it HAS been trained on 1129's own data), or a held-out
# generalization artifact (only inflated when 1129's own labels were never seen)?
#
# spatial_block_kfold assigns each compartment to the test set in exactly one of the 5 folds --
# so compartment 1129 is test-only in fold 0 and part of the TRAINING data in folds 1-4. This
# script trains plain PINN (y_max-only fix) on each fold in turn (0-4, so fold 0 is included as
# the already-known "held-out" reference point, derived the same way as the others for a clean
# apples-to-apples comparison) at the trusted config, then evaluates y_max_pred specifically on
# compartment 1129's rows regardless of which split they fall into for that fold -- reports both
# the split membership (train/val/test) and the resulting y_max_pred distribution.
#
# Run (fold 0, local): PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_compartment1129_cross_fold_check.py
# Run (any fold): PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_compartment1129_cross_fold_check.py --fold-index 1
#
# Saves a summary JSON per fold to
# temp_results_pinn/outputs/CORRECTED_2026-08-23_mechanism_checks/compartment1129_cross_fold/fold_<i>/summary.json

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

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

FEATURE_SET = "nested_set3_gated_terrain_wind_vif"
COHORT = "4survey"
SPLIT_TYPE = "spatial_block_kfold"
N_FOLDS = 5
SEED = 42
TARGET_BLK = 21
TARGET_CPMT = 1129
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "CORRECTED_2026-08-23_mechanism_checks" / "compartment1129_cross_fold"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-index", type=int, default=0, help="0-4, which spatial_block_kfold fold to hold out as test.")
    args = parser.parse_args()
    fold_index = args.fold_index

    device = select_device()
    print(f"Device: {device}  Fold: {fold_index}  Feature set: {FEATURE_SET}")

    fold_dir = OUTPUT_DIR / f"fold_{fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_path = fold_dir / "summary.json"
    if summary_path.exists():
        print(f"{summary_path} already exists -- delete it first to redo this fold. Exiting without retraining.")
        return

    feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET]
    cr_params = load_cr_params(COHORT, SPLIT_TYPE, split_seed=SPLIT_SEED, held_out_fold=fold_index)
    split_df = load_split_table_with_terrain(
        COHORT, SPLIT_TYPE, feature_columns, split_seed=SPLIT_SEED,
        k_folds=N_FOLDS, held_out_fold=fold_index,
    )

    # Which split does compartment 1129 fall into for this fold, before we do anything else?
    cpmt1129_mask_full = (split_df["blk"] == TARGET_BLK) & (split_df["cpmt"] == TARGET_CPMT)
    cpmt1129_rows = split_df[cpmt1129_mask_full]
    split_membership = cpmt1129_rows["split"].value_counts().to_dict()
    print(f"Compartment 1129 rows in this fold's split_df: {len(cpmt1129_rows)}  membership: {split_membership}")
    if len(cpmt1129_rows) == 0:
        print("WARNING: compartment 1129 has zero rows in this fold's split_df (buffer-excluded or "
              "otherwise not present) -- cannot evaluate. Saving a summary noting this and exiting.")
        with open(summary_path, "w") as f:
            json.dump({"fold_index": fold_index, "cpmt1129_n_rows": 0, "note": "compartment 1129 absent from this fold's split_df"}, f, indent=2)
        return

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

    pair_tensors = build_pair_tensors(
        pairs_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
    terrain_pairs = build_pair_terrain_tensor(pairs_df, scaler_terrain, feature_columns, device, COHORT)

    n_other_features = other_train.shape[1]
    n_terrain_features = terrain_train.shape[1]

    print("\nTraining plain PINN (y_max-only), trusted Set3 config...")
    t0 = time.time()
    model, _, history = fit(
        age_train, other_train, terrain_train, target_train,
        age_val, other_val, terrain_val, target_val,
        pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
        n_other_features, n_terrain_features, device, SEED,
        max_epochs=500, early_stopping_patience=40,
    )
    print(f"Trained {len(history)} epochs in {time.time()-t0:.0f}s")

    # Now evaluate compartment 1129's rows specifically, regardless of which split they're in.
    # Use the SAME scalers fit on this fold's training data (consistent with how the model
    # itself was trained -- not refitting scalers on the compartment subset).
    terrain_cpmt1129 = build_terrain_tensor(cpmt1129_rows, scaler_terrain, feature_columns, device)
    y_max_pred_cpmt1129 = predict_y_max(model, terrain_cpmt1129, cr_params["y_max"]).cpu().numpy().flatten()

    unique_plots = cpmt1129_rows["identification"].nunique()
    print(f"\nCompartment 1129: {unique_plots} unique plots, {len(cpmt1129_rows)} rows, split membership: {split_membership}")
    print(f"y_max_pred for compartment 1129: mean={y_max_pred_cpmt1129.mean():.3f}  std={y_max_pred_cpmt1129.std():.3f}  "
          f"min={y_max_pred_cpmt1129.min():.3f}  max={y_max_pred_cpmt1129.max():.3f}")
    print(f"Population y_max (reference): {cr_params['y_max']:.3f}")
    n_over_70 = int((y_max_pred_cpmt1129 > 70).sum())
    print(f"Rows with y_max_pred > 70m: {n_over_70} / {len(y_max_pred_cpmt1129)}")

    with open(summary_path, "w") as f:
        json.dump({
            "fold_index": fold_index,
            "cpmt1129_n_rows": len(cpmt1129_rows),
            "cpmt1129_n_unique_plots": int(unique_plots),
            "split_membership": {str(k): int(v) for k, v in split_membership.items()},
            "y_max_pred_mean": float(y_max_pred_cpmt1129.mean()),
            "y_max_pred_std": float(y_max_pred_cpmt1129.std()),
            "y_max_pred_min": float(y_max_pred_cpmt1129.min()),
            "y_max_pred_max": float(y_max_pred_cpmt1129.max()),
            "population_y_max": float(cr_params["y_max"]),
            "n_rows_over_70m": n_over_70,
        }, f, indent=2)
    print(f"\nSaved -> {summary_path}")


if __name__ == "__main__":
    main()

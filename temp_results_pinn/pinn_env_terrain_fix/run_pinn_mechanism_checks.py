# Two "why" checks for claims already being made about the CORRECTED PINN/PINN-k, both cheap
# (no cluster time, single fold, reuses existing methodology where possible):
#
# Check 1 -- why does plain PINN beat PINN-k on accuracy? Compares how much "compensating work"
# each variant's trunk (main_network) residual is doing. If PINN-k's trunk residual is
# systematically larger than plain PINN's, that's evidence the k sub-network adds noise the
# trunk has to correct for, rather than useful structure.
#
# Check 2 -- why do specific plots get an implausible/most-inflated y_max? Reuses the exact
# out-of-training-range methodology already used for the DNN-vs-XGBoost terrain-extrapolation
# check (models/baselines/rq1_extrapolation_check.py::compute_extrapolation_score), applied to
# plain PINN's y_max output instead. Tests whether the most-inflated/implausible plots sit at
# unusual terrain values.
#
# Both checks retrain fresh (fold 0, Set3, trusted config: lr=0.0001, weight_decay=1e-5,
# batch_size=256, physics_weight=1.0) since no checkpoint was saved for the trusted Table 3 run
# -- same reasoning as run_terrain_permutation_importance.py. No new output directory needed,
# this is a diagnostic, not a Table-3 number.
#
# Run (fold 0, local): PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_pinn_mechanism_checks.py
# Run (any fold): PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_pinn_mechanism_checks.py --fold-index 1
#
# Saves a summary JSON per fold to
# temp_results_pinn/outputs/CORRECTED_2026-08-23_mechanism_checks/trunk_and_terrain/fold_<i>/summary.json
# -- added 2026-08-23 so this can be run on the cluster across multiple folds and rsynced back.

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.baselines.rq1_extrapolation_check import compute_extrapolation_score
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

FEATURE_SET = "nested_set3_gated_terrain_wind_vif"
COHORT = "4survey"
SPLIT_TYPE = "spatial_block_kfold"
N_FOLDS = 5
SEED = 42
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "CORRECTED_2026-08-23_mechanism_checks" / "trunk_and_terrain"


def train_variant(variant, age_train, other_train, terrain_train, target_train,
                   age_val, other_val, terrain_val, target_val,
                   pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
                   n_other_features, n_terrain_features, device):
    if variant == "ymax":
        from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_fix import fit
    else:
        from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_k_fix import fit

    print(f"\nTraining {variant}, trusted Set3 config...")
    t0 = time.time()
    model, _, history = fit(
        age_train, other_train, terrain_train, target_train,
        age_val, other_val, terrain_val, target_val,
        pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
        n_other_features, n_terrain_features, device, SEED,
        max_epochs=500, early_stopping_patience=40,
    )
    print(f"  Trained {len(history)} epochs in {time.time()-t0:.0f}s")
    return model


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

    common_args = (age_train, other_train, terrain_train, target_train,
                   age_val, other_val, terrain_val, target_val,
                   pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
                   n_other_features, n_terrain_features, device)

    model_ymax = train_variant("ymax", *common_args)
    model_k = train_variant("k", *common_args)

    # ---------------------------------------------------------------------------
    # Check 1: trunk-residual magnitude, plain PINN vs. PINN-k
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CHECK 1: trunk-residual magnitude (plain PINN vs. PINN-k)")
    print("=" * 70)
    model_ymax.eval()
    model_k.eval()
    import torch
    with torch.no_grad():
        trunk_ymax = model_ymax.main_network(other_test, age_test).cpu().numpy().flatten()
        trunk_k = model_k.main_network(other_test, age_test).cpu().numpy().flatten()

    print(f"Plain PINN trunk residual (scaled): mean|.|={np.abs(trunk_ymax).mean():.4f}  std={trunk_ymax.std():.4f}")
    print(f"PINN-k trunk residual (scaled):     mean|.|={np.abs(trunk_k).mean():.4f}  std={trunk_k.std():.4f}")
    ratio = np.abs(trunk_k).mean() / np.abs(trunk_ymax).mean()
    print(f"Ratio (PINN-k / plain PINN): {ratio:.3f}")
    print("(ratio > 1 means PINN-k's trunk is doing MORE compensating work -- evidence the k")
    print(" sub-network adds noise the trunk has to correct for, rather than useful structure)")

    # ---------------------------------------------------------------------------
    # Check 2: do implausible/most-inflated y_max plots sit at unusual terrain values?
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CHECK 2: implausible/inflated y_max vs. terrain extrapolation")
    print("=" * 70)
    from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_fix import predict_y_max
    y_max_pred = predict_y_max(model_ymax, terrain_test, cr_params["y_max"]).cpu().numpy().flatten()
    y_max_deviation = y_max_pred - cr_params["y_max"]

    implausible_mask = (y_max_pred < 5) | (y_max_pred > 70)
    top_decile_threshold = np.quantile(y_max_deviation, 0.90)
    most_inflated_mask = y_max_deviation >= top_decile_threshold

    print(f"n test plots: {len(test_df)}")
    print(f"Implausible (<5m or >70m): {implausible_mask.sum()} ({implausible_mask.mean():.2%})")
    print(f"Most-inflated (top 10% by deviation, >= {top_decile_threshold:.2f}m above population): {most_inflated_mask.sum()}")

    continuous_columns = ["Age", "CanopyCover", "time_since_thinning"] + list(feature_columns)
    extrapolation_score, n_out_of_range = compute_extrapolation_score(train_df, test_df, continuous_columns)
    extrapolation_score = extrapolation_score.to_numpy()
    n_out_of_range = n_out_of_range.to_numpy()

    print(f"\nMean extrapolation score (0 = fully in-range) by group:")
    print(f"  All test plots:        {extrapolation_score.mean():.4f}  (n={len(test_df)})")
    print(f"  Implausible plots:     {extrapolation_score[implausible_mask].mean():.4f}  (n={implausible_mask.sum()})" if implausible_mask.sum() > 0 else "  Implausible plots:     n=0, skipped")
    print(f"  Most-inflated plots:   {extrapolation_score[most_inflated_mask].mean():.4f}  (n={most_inflated_mask.sum()})")
    print(f"  Everyone else:         {extrapolation_score[~most_inflated_mask].mean():.4f}  (n={(~most_inflated_mask).sum()})")

    print(f"\nMean # out-of-range terrain features by group:")
    print(f"  All test plots:        {n_out_of_range.mean():.3f}")
    print(f"  Most-inflated plots:   {n_out_of_range[most_inflated_mask].mean():.3f}")
    print(f"  Everyone else:         {n_out_of_range[~most_inflated_mask].mean():.3f}")

    with open(summary_path, "w") as f:
        json.dump({
            "fold_index": fold_index, "feature_set": FEATURE_SET,
            "check1_trunk_residual": {
                "plain_pinn_mean_abs": float(np.abs(trunk_ymax).mean()), "plain_pinn_std": float(trunk_ymax.std()),
                "pinn_k_mean_abs": float(np.abs(trunk_k).mean()), "pinn_k_std": float(trunk_k.std()),
                "ratio_pinnk_over_plain": ratio,
            },
            "check2_terrain_extrapolation": {
                "n_test": len(test_df),
                "n_implausible": int(implausible_mask.sum()), "pct_implausible": float(implausible_mask.mean()),
                "n_most_inflated": int(most_inflated_mask.sum()), "top_decile_threshold_m": float(top_decile_threshold),
                "extrapolation_score_all": float(extrapolation_score.mean()),
                "extrapolation_score_implausible": float(extrapolation_score[implausible_mask].mean()) if implausible_mask.sum() > 0 else None,
                "extrapolation_score_most_inflated": float(extrapolation_score[most_inflated_mask].mean()),
                "extrapolation_score_everyone_else": float(extrapolation_score[~most_inflated_mask].mean()),
                "n_out_of_range_all": float(n_out_of_range.mean()),
                "n_out_of_range_most_inflated": float(n_out_of_range[most_inflated_mask].mean()),
                "n_out_of_range_everyone_else": float(n_out_of_range[~most_inflated_mask].mean()),
            },
        }, f, indent=2)
    print(f"\nSaved -> {summary_path}")


if __name__ == "__main__":
    main()

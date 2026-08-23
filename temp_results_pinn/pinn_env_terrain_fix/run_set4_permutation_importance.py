# Permutation importance for PINN-k's terrain sub-network, trained on Set4 instead of Set3 --
# answers "why doesn't the broader feature set (Set4) help over the curated Set3?"
# (temp_results_pinn/RESULTS_TABLE.md, section 5: Set4 R2=0.6196 vs. Set3's 0.6180 for PINN-k,
# statistically flat, but Set4 measurably hurts plain PINN, 0.618 vs. 0.631).
#
# Note Set4 is NOT simply Set3 plus extra features -- it DROPS elevation (present in Set3) and
# ADDS four new features: dist_to_scpt_boundary, tas_mean, chelsa_gdd5_degc,
# cpmt_compactness_ratio. This script flags each feature as "shared with Set3" or "new in Set4"
# so the analysis reads correctly -- checking whether the NEW features specifically get near-zero
# importance (harmless but useless, explaining the flat/negative result) or real-but-noisy
# importance (actively unhelpful).
#
# Same method as run_terrain_permutation_importance.py (Set3 version): shuffle one terrain
# feature at a time, measure how much y_max_pred/k_pred moves. No checkpoint exists for Set4
# (never saved), so this retrains fresh and keeps the model in memory.
#
# Run (fold 0, local): PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_set4_permutation_importance.py
# Run (any fold): PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_set4_permutation_importance.py --fold-index 1
#
# Saves a summary JSON per fold to
# temp_results_pinn/outputs/CORRECTED_2026-08-23_mechanism_checks/set4_permutation_importance/fold_<i>/summary.json

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
from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_k_fix import fit, predict_y_max, predict_k

FEATURE_SET = "nested_set4_gated_all_vif"
SET3_FEATURE_SET = "nested_set3_gated_terrain_wind_vif"  # for the shared-vs-new flag only
COHORT = "4survey"
SPLIT_TYPE = "spatial_block_kfold"
N_FOLDS = 5
SEED = 42
N_PERMUTATION_REPEATS = 10
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "CORRECTED_2026-08-23_mechanism_checks" / "set4_permutation_importance"


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
    set3_columns = set(ENV_TERRAIN_FEATURE_SETS[SET3_FEATURE_SET])
    print(f"Terrain features ({len(feature_columns)}): {feature_columns}")
    new_in_set4 = [f for f in feature_columns if f not in set3_columns]
    print(f"New in Set4 (not in Set3): {new_in_set4}")

    cr_params = load_cr_params(COHORT, SPLIT_TYPE, split_seed=SPLIT_SEED, held_out_fold=fold_index)
    split_df = load_split_table_with_terrain(
        COHORT, SPLIT_TYPE, feature_columns, split_seed=SPLIT_SEED,
        k_folds=N_FOLDS, held_out_fold=fold_index,
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

    print("\nTraining PINN-k, Set4, trusted config (lr=0.0001, weight_decay=1e-5, "
          "batch_size=256, physics_weight=1.0)...")
    t0 = time.time()
    model, _, history = fit(
        age_train, other_train, terrain_train, target_train,
        age_val, other_val, terrain_val, target_val,
        pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
        n_other_features, n_terrain_features, device, SEED,
        max_epochs=500, early_stopping_patience=40,
    )
    print(f"Trained {len(history)} epochs in {time.time()-t0:.0f}s")

    baseline_y_max = predict_y_max(model, terrain_test, cr_params["y_max"]).cpu().numpy().flatten()
    baseline_k = predict_k(model, terrain_test, cr_params["k"]).cpu().numpy().flatten()

    print(f"\nBaseline y_max_pred: mean={baseline_y_max.mean():.3f}  std={baseline_y_max.std():.3f}")
    print(f"Baseline k_pred: mean={baseline_k.mean():.5f}  std={baseline_k.std():.5f}")

    rng = np.random.default_rng(SEED)
    n_rows = terrain_test.shape[0]

    results = []
    for feature_idx, feature_name in enumerate(feature_columns):
        y_max_shifts = []
        k_shifts = []
        for _ in range(N_PERMUTATION_REPEATS):
            permuted_terrain = terrain_test.clone()
            shuffled_row_order = rng.permutation(n_rows)
            permuted_terrain[:, feature_idx] = terrain_test[shuffled_row_order, feature_idx]

            permuted_y_max = predict_y_max(model, permuted_terrain, cr_params["y_max"]).cpu().numpy().flatten()
            permuted_k = predict_k(model, permuted_terrain, cr_params["k"]).cpu().numpy().flatten()

            y_max_shifts.append(np.mean(np.abs(permuted_y_max - baseline_y_max)))
            k_shifts.append(np.mean(np.abs(permuted_k - baseline_k)))

        results.append({
            "feature": feature_name,
            "new_in_set4": feature_name in new_in_set4,
            "y_max_importance": float(np.mean(y_max_shifts)),
            "k_importance": float(np.mean(k_shifts)),
        })

    total_y_max_importance = sum(r["y_max_importance"] for r in results)
    total_k_importance = sum(r["k_importance"] for r in results)

    results_sorted_ymax = sorted(results, key=lambda r: -r["y_max_importance"])
    print(f"\n{'Feature':>25} | {'New?':>5} | {'y_max importance':>18} | {'% of total':>10} | {'k importance':>14} | {'% of total':>10}")
    print("-" * 100)
    for r in results_sorted_ymax:
        y_max_pct = 100 * r["y_max_importance"] / total_y_max_importance if total_y_max_importance > 0 else 0
        k_pct = 100 * r["k_importance"] / total_k_importance if total_k_importance > 0 else 0
        new_flag = "NEW" if r["new_in_set4"] else ""
        print(f"{r['feature']:>25} | {new_flag:>5} | {r['y_max_importance']:>18.4f} | {y_max_pct:>9.1f}% | {r['k_importance']:>14.6f} | {k_pct:>9.1f}%")

    new_feature_total_share = sum(100 * r["y_max_importance"] / total_y_max_importance for r in results if r["new_in_set4"])
    print(f"\nThe {len(new_in_set4)} features new to Set4 collectively carry {new_feature_total_share:.1f}% of total y_max importance")
    print(f"(if this is roughly proportional to their share of all {len(feature_columns)} features, "
          f"{100*len(new_in_set4)/len(feature_columns):.1f}%, they're not doing anything special either way)")

    with open(summary_path, "w") as f:
        json.dump({
            "fold_index": fold_index, "feature_set": FEATURE_SET,
            "new_in_set4": new_in_set4,
            "baseline_y_max_mean": float(baseline_y_max.mean()), "baseline_y_max_std": float(baseline_y_max.std()),
            "baseline_k_mean": float(baseline_k.mean()), "baseline_k_std": float(baseline_k.std()),
            "new_feature_total_ymax_share_pct": new_feature_total_share,
            "per_feature_importance": results,
        }, f, indent=2)
    print(f"\nSaved -> {summary_path}")


if __name__ == "__main__":
    main()

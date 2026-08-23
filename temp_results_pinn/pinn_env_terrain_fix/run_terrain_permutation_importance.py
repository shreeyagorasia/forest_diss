# Permutation importance for PINN-k's terrain sub-network -- answers "is the personalized
# y_max (and k) dominated by one terrain feature, the way Q2 found GNNWR's whole advantage
# collapsed down to CanopyCover alone?" Same underlying question, same technique (permute one
# feature, measure how much the model's output moves), applied to a different model.
#
# No checkpoint exists for the trusted Set3 PINN-k run (temp_results_pinn/outputs/full_rerun*/
# only saved metrics/history, not the model) -- so this retrains fold 0 at the exact trusted
# config (lr=0.0001, weight_decay=1e-5, batch_size=256, physics_weight=1.0, Set3) and keeps the
# model in memory, then runs the importance analysis directly against it. No new output
# directory needed -- this doesn't produce a number that goes in Table 3, just a diagnostic.
#
# Method: for each terrain feature, shuffle that column's values across test-set rows (breaking
# its link to every other feature/row, keeping everything else fixed), recompute y_max_pred and
# k_pred, and measure the mean absolute change from the unpermuted baseline. A feature whose
# shuffle barely moves the prediction isn't doing much; a feature whose shuffle moves it a lot is
# doing most of the work. Repeated 10x per feature (different random shuffle each time) and
# averaged, to reduce noise from any one unlucky shuffle.
#
# Run (fold 0, local): PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_terrain_permutation_importance.py
# Run (any fold): PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_terrain_permutation_importance.py --fold-index 1
#
# Saves a summary JSON per fold to
# temp_results_pinn/outputs/CORRECTED_2026-08-23_mechanism_checks/permutation_importance/fold_<i>/summary.json
# -- added 2026-08-23 so this can be run on the cluster across multiple folds and rsynced back,
# not just read off a single local stdout capture.

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

FEATURE_SET = "nested_set3_gated_terrain_wind_vif"
COHORT = "4survey"
SPLIT_TYPE = "spatial_block_kfold"
N_FOLDS = 5
SEED = 42
N_PERMUTATION_REPEATS = 10
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "CORRECTED_2026-08-23_mechanism_checks" / "permutation_importance"


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
    print(f"Terrain features ({len(feature_columns)}): {feature_columns}")

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

    print("\nTraining PINN-k, fold 0, trusted Set3 config (lr=0.0001, weight_decay=1e-5, "
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

    # Baseline predictions (unpermuted terrain).
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
            "y_max_importance": float(np.mean(y_max_shifts)),
            "k_importance": float(np.mean(k_shifts)),
        })

    total_y_max_importance = sum(r["y_max_importance"] for r in results)
    total_k_importance = sum(r["k_importance"] for r in results)

    results_sorted_ymax = sorted(results, key=lambda r: -r["y_max_importance"])
    print(f"\n{'Feature':>25} | {'y_max importance':>18} | {'% of total':>10} | {'k importance':>14} | {'% of total':>10}")
    print("-" * 90)
    for r in results_sorted_ymax:
        y_max_pct = 100 * r["y_max_importance"] / total_y_max_importance if total_y_max_importance > 0 else 0
        k_pct = 100 * r["k_importance"] / total_k_importance if total_k_importance > 0 else 0
        print(f"{r['feature']:>25} | {r['y_max_importance']:>18.4f} | {y_max_pct:>9.1f}% | {r['k_importance']:>14.6f} | {k_pct:>9.1f}%")

    top_feature = results_sorted_ymax[0]
    top_pct = 100 * top_feature["y_max_importance"] / total_y_max_importance
    print(f"\nTop feature for y_max: {top_feature['feature']} ({top_pct:.1f}% of total importance)")
    print("(GNNWR/Q2 comparison point: CanopyCover carried ~80-90% of GNNWR's total signal)")

    with open(summary_path, "w") as f:
        json.dump({
            "fold_index": fold_index, "feature_set": FEATURE_SET,
            "baseline_y_max_mean": float(baseline_y_max.mean()), "baseline_y_max_std": float(baseline_y_max.std()),
            "baseline_k_mean": float(baseline_k.mean()), "baseline_k_std": float(baseline_k.std()),
            "top_feature_ymax": top_feature["feature"], "top_feature_ymax_pct": top_pct,
            "per_feature_importance": results,
        }, f, indent=2)
    print(f"Saved -> {summary_path}")


if __name__ == "__main__":
    main()

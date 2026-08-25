# MIGRATED 2026-08-25 from temp_results_pinn/pinn_env_terrain_fix/run_full_rerun.py into models/
# alongside the fixed model files. Two changes from the original, both non-behavioural:
#   1. The two model imports now point at models.pinn_env_terrain_fix.* (this package) instead
#      of temp_results_pinn.pinn_env_terrain_fix.* -- same code, new home.
#   2. OUTPUT_DIR is now an explicit path back to temp_results_pinn/outputs/full_rerun instead
#      of being derived from this file's own location. The original computed it as
#      `Path(__file__).resolve().parents[1] / "outputs" / "full_rerun"`, i.e. "my grandparent
#      folder's outputs/" -- that resolved to temp_results_pinn/outputs/full_rerun when the file
#      lived at temp_results_pinn/pinn_env_terrain_fix/run_full_rerun.py, but would have silently
#      resolved to a new, empty models/outputs/full_rerun/ if copied here unchanged. Hardcoded
#      instead, so this script still sees (and still skips) the 5 already-completed folds whose
#      results are cited in the dissertation, rather than treating them as not-yet-run.
#
# Everything else -- the fit/predict calls, the fold loop, the CR anchor loading, the printed
# summary -- is unchanged from the original.
#
# Full-epoch rerun of the FIXED PINN/PINN-k architectures only (OLD/unfixed numbers already
# exist in the dissertation's own Table 1 -- no need to redo those). Matches production settings
# exactly: spatial_block_kfold, 5 folds, Set3, max_epochs=500, patience=40 (models/pinn_env_terrain
# /run_pinn_env_terrain.py's own DEFAULT_MAX_EPOCHS/DEFAULT_EARLY_STOPPING_PATIENCE).
#
# Loops fold 0 first (both models), then folds 1-4 (both models) if reached -- designed to run
# unattended in the background for a long time; saves each fold's result to disk incrementally
# so partial progress survives even if the whole loop doesn't finish.
#
# Run as: PYTHONPATH=. .venv/bin/python models/pinn_env_terrain_fix/run_full_rerun.py

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
SPLIT_TYPE = "spatial_block_kfold"
N_FOLDS = 5
FEATURE_SET_NAME = "nested_set3_gated_terrain_wind_vif"
MAX_EPOCHS = 500       # matches production DEFAULT_MAX_EPOCHS
PATIENCE = 40          # matches production DEFAULT_EARLY_STOPPING_PATIENCE
SEED = 42

# Fixed to the same physical location the original script used, so the 5 folds already run
# there (and cited in the dissertation) are found and skipped, not silently redone or duplicated
# under a new path. See migration note at the top of this file.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "temp_results_pinn" / "outputs" / "full_rerun"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = select_device()
print(f"Device: {device}", flush=True)

feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]


def unscale(scaled_tensor, scaler):
    return scaled_tensor.cpu().numpy().flatten() * scaler.scale_[0] + scaler.mean_[0]


def run_one_fold(fold_index):
    print(f"\n{'#' * 70}\n# FOLD {fold_index} / {N_FOLDS}\n{'#' * 70}", flush=True)
    fold_dir = OUTPUT_DIR / f"fold_{fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_path = fold_dir / "summary.json"
    if summary_path.exists():
        print(f"  fold {fold_index} already complete, skipping (delete {summary_path} to redo)", flush=True)
        return json.loads(summary_path.read_text())

    cr_params = load_cr_params(COHORT, SPLIT_TYPE, split_seed=SPLIT_SEED, held_out_fold=fold_index)
    split_df = load_split_table_with_terrain(
        COHORT, SPLIT_TYPE, feature_columns, split_seed=SPLIT_SEED, k_folds=N_FOLDS, held_out_fold=fold_index,
    )
    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}", flush=True)

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

    fold_results = {}

    # ----- y_max-only fixed -----
    print(f"\n  --- fold {fold_index}: PINN (y_max only), fixed, full epochs ---", flush=True)
    t0 = time.time()
    from models.pinn_env_terrain_fix.pinn_env_terrain_fix import fit as fit_ymax
    from models.pinn_env_terrain_fix.pinn_env_terrain_fix import predict as predict_ymax

    model_ymax, _, history_ymax = fit_ymax(
        age_train, other_train, terrain_train, target_train,
        age_val, other_val, terrain_val, target_val,
        pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
        n_other_features, n_terrain_features, device, SEED,
        max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
    )
    preds_ymax = unscale(predict_ymax(model_ymax, age_test, other_test, terrain_test), scaler_height)
    target_unscaled = unscale(target_test, scaler_height)
    metrics_ymax = compute_metrics(target_unscaled, preds_ymax)
    elapsed_ymax = time.time() - t0
    print(f"  fold {fold_index} PINN(y_max)-fixed: R2={metrics_ymax['r2']:.4f} RMSE={metrics_ymax['rmse']:.4f} "
          f"MAE={metrics_ymax['mae']:.4f} ({len(history_ymax)} epochs, {elapsed_ymax:.1f}s)", flush=True)
    history_ymax.to_csv(fold_dir / "pinn_ymax_fixed_history.csv", index=False)
    fold_results["pinn_ymax_fixed"] = {**metrics_ymax, "n_epochs": len(history_ymax), "elapsed_seconds": elapsed_ymax}

    # ----- y_max + k fixed -----
    print(f"\n  --- fold {fold_index}: PINN-k (y_max + k), fixed, full epochs ---", flush=True)
    t0 = time.time()
    from models.pinn_env_terrain_fix.pinn_env_terrain_k_fix import fit as fit_k
    from models.pinn_env_terrain_fix.pinn_env_terrain_k_fix import predict as predict_k

    model_k, _, history_k = fit_k(
        age_train, other_train, terrain_train, target_train,
        age_val, other_val, terrain_val, target_val,
        pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
        n_other_features, n_terrain_features, device, SEED,
        max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
    )
    preds_k = unscale(predict_k(model_k, age_test, other_test, terrain_test), scaler_height)
    metrics_k = compute_metrics(target_unscaled, preds_k)
    elapsed_k = time.time() - t0
    print(f"  fold {fold_index} PINN-k-fixed: R2={metrics_k['r2']:.4f} RMSE={metrics_k['rmse']:.4f} "
          f"MAE={metrics_k['mae']:.4f} ({len(history_k)} epochs, {elapsed_k:.1f}s)", flush=True)
    history_k.to_csv(fold_dir / "pinn_k_fixed_history.csv", index=False)
    fold_results["pinn_k_fixed"] = {**metrics_k, "n_epochs": len(history_k), "elapsed_seconds": elapsed_k}

    with open(summary_path, "w") as f:
        json.dump(fold_results, f, indent=2)
    print(f"  fold {fold_index} complete, saved -> {fold_dir}", flush=True)
    return fold_results


all_results = {}
for fold_index in range(N_FOLDS):
    all_results[fold_index] = run_one_fold(fold_index)
    # Write a running pooled summary after every fold, so progress is inspectable at any time
    # without waiting for all 5 folds to finish.
    with open(OUTPUT_DIR / "all_folds_summary_so_far.json", "w") as f:
        json.dump(all_results, f, indent=2)

    r2_ymax = [r["pinn_ymax_fixed"]["r2"] for r in all_results.values()]
    r2_k = [r["pinn_k_fixed"]["r2"] for r in all_results.values()]
    print(f"\n{'=' * 70}\nRUNNING SUMMARY after fold {fold_index} ({len(all_results)}/{N_FOLDS} folds done)\n{'=' * 70}", flush=True)
    print(f"PINN(y_max)-fixed R2 so far: {[f'{v:.4f}' for v in r2_ymax]}  mean={sum(r2_ymax)/len(r2_ymax):.4f}", flush=True)
    print(f"PINN-k-fixed      R2 so far: {[f'{v:.4f}' for v in r2_k]}  mean={sum(r2_k)/len(r2_k):.4f}", flush=True)

print("\nALL FOLDS COMPLETE")

# Set2/Set4 feature-set sweep for the CORRECTED (fixed-forward-pass) PINN/PINN-k, run on the
# cluster -- a light copy of run_cluster_fold.py (the already-proven script that produced the
# real Table 3 numbers for Set3), with two changes:
#   1. Output path includes the feature-set name, so this can NEVER collide with or overwrite
#      the existing Set3 results in temp_results_pinn/outputs/full_rerun_cluster/ -- writes to
#      a separate, clearly-named directory instead (CORRECTED_2026-08-22_pinn_set_sweep/<set>/fold_<i>/).
#   2. --feature-set has no default -- must be passed explicitly (nested_set2_top10 or
#      nested_set4_gated_all_vif; Set3 already exists, don't rerun it here).
#
# Purpose: the earlier Set2/Set3/Set4 sweep (Aug 11) used the pre-fix architecture -- all three
# sets landed at ~0.577-0.579, indistinguishable from no-environment, the known bug signature.
# This reruns Set2 and Set4 with the real, corrected code, to actually test whether feature-set
# choice (and environment more broadly) affects PINN/PINN-k, which is the core argument for
# environmental attribution mattering across this whole dissertation.
#
# Run directly (cluster login node, tiny settings, to smoke-test before a real sbatch submit):
#   PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_set_sweep.py \
#     --fold-index 0 --variant ymax --feature-set nested_set2_top10 --max-epochs 3 --patience 2
#
# Normally launched via temp_results_pinn/jobs/run_pinn_fix_set_sweep_cluster.sh (sbatch).

import argparse
import json
import sys
import time
from pathlib import Path

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

# Deliberately a DIFFERENT directory from run_cluster_fold.py's OUTPUT_DIR (full_rerun_cluster/)
# -- this is the whole point, see module docstring above.
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "CORRECTED_2026-08-22_pinn_set_sweep"

VALID_SETS = {"nested_set2_top10", "nested_set4_gated_all_vif", "no_environment_ablation"}


def unscale(scaled_tensor, scaler):
    return scaled_tensor.cpu().numpy().flatten() * scaler.scale_[0] + scaler.mean_[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-index", type=int, required=True, help="0-4, which spatial_block_kfold fold to hold out as test.")
    parser.add_argument("--variant", choices=["ymax", "k"], required=True, help="'ymax' = y_max-only fix, 'k' = y_max+k fix.")
    parser.add_argument("--feature-set", required=True, choices=sorted(VALID_SETS), help="Set3 already has real corrected results -- only Set2/Set4 need rerunning here.")
    parser.add_argument("--cohort", default="4survey")
    parser.add_argument("--split-type", default="spatial_block_kfold")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=500, help="Matches production DEFAULT_MAX_EPOCHS -- same as the Set3 run this is meant to be comparable to.")
    parser.add_argument("--patience", type=int, default=40, help="Matches production DEFAULT_EARLY_STOPPING_PATIENCE.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Project default -- override with the winning value from run_cluster_fold_lr_check.py once known.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Project default -- override with the winning value from run_cluster_fold_lr_check.py once known.")
    parser.add_argument("--physics-weight", type=float, default=1.0, help="Project default -- override with the winning lambda from run_cluster_fold_lambda_ablation.py once known.")
    parser.add_argument("--batch-size", type=int, default=256, help="Matches PINN's own default -- set explicitly (not relying on the module default) so it's recorded in the summary JSON.")
    args = parser.parse_args()

    if not (0 <= args.fold_index < args.n_folds):
        raise ValueError(f"--fold-index must be in [0, {args.n_folds}), got {args.fold_index}")

    fold_dir = OUTPUT_DIR / args.feature_set / f"fold_{args.fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_path = fold_dir / f"pinn_{args.variant}_fixed_summary.json"
    if summary_path.exists():
        print(f"{summary_path} already exists -- delete it first to redo this fold/variant. Exiting without retraining.")
        return

    device = select_device()
    print(f"Device: {device}")
    print(f"Fold {args.fold_index}/{args.n_folds}  variant={args.variant}  feature_set={args.feature_set}  cohort={args.cohort}")

    feature_columns = ENV_TERRAIN_FEATURE_SETS[args.feature_set]

    cr_params = load_cr_params(args.cohort, args.split_type, split_seed=SPLIT_SEED, held_out_fold=args.fold_index)
    split_df = load_split_table_with_terrain(
        args.cohort, args.split_type, feature_columns, split_seed=SPLIT_SEED,
        k_folds=args.n_folds, held_out_fold=args.fold_index,
    )
    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]
    print(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    pairs_df = load_trajectory_pairs(args.cohort, split_df)

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
    terrain_pairs = build_pair_terrain_tensor(pairs_df, scaler_terrain, feature_columns, device, args.cohort)

    n_other_features = other_train.shape[1]
    n_terrain_features = terrain_train.shape[1]

    if args.variant == "ymax":
        from temp_results_pinn.pinn_env_terrain_fix import pinn_env_terrain_fix as pinn_module
        from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_fix import fit, predict
    else:
        from temp_results_pinn.pinn_env_terrain_fix import pinn_env_terrain_k_fix as pinn_module
        from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_k_fix import fit, predict

    # weight_decay is a module-level constant inside build_optimizer, not a fit() kwarg -- same
    # monkeypatch pattern used in run_cluster_fold_lr_check.py, restored after training.
    original_weight_decay = pinn_module.WEIGHT_DECAY
    pinn_module.WEIGHT_DECAY = args.weight_decay
    try:
        t0 = time.time()
        model, _, history = fit(
            age_train, other_train, terrain_train, target_train,
            age_val, other_val, terrain_val, target_val,
            pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
            n_other_features, n_terrain_features, device, args.seed,
            max_epochs=args.max_epochs, early_stopping_patience=args.patience,
            physics_weight=args.physics_weight,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
    finally:
        pinn_module.WEIGHT_DECAY = original_weight_decay

    preds = unscale(predict(model, age_test, other_test, terrain_test), scaler_height)
    target_unscaled = unscale(target_test, scaler_height)
    metrics = compute_metrics(target_unscaled, preds)
    elapsed = time.time() - t0

    print(f"fold {args.fold_index} variant={args.variant} set={args.feature_set}: R2={metrics['r2']:.4f} "
          f"RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} ({len(history)} epochs, {elapsed:.1f}s)")

    history.to_csv(fold_dir / f"pinn_{args.variant}_fixed_history.csv", index=False)
    with open(summary_path, "w") as f:
        json.dump({**metrics, "n_epochs": len(history), "elapsed_seconds": elapsed,
                   "fold_index": args.fold_index, "variant": args.variant, "feature_set": args.feature_set,
                   "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                   "physics_weight": args.physics_weight, "batch_size": args.batch_size}, f, indent=2)
    print(f"Saved -> {summary_path}")


if __name__ == "__main__":
    main()

# Tests learning_rate/weight_decay/batch_size overrides for the CORRECTED (fixed-forward-pass)
# PINN/PINN-k against the trusted Set3 baseline. Fixed to Set3
# (nested_set3_gated_terrain_wind_vif), the set with the already-trusted Table 3 numbers, so
# results are directly comparable. Physics_weight (lambda) stays at the current default (1.0)
# here -- the lambda ablation is a separate, later step.
#
# History: originally tested the DNN's Aug-19 hyperparameter finding (learning_rate=0.001,
# weight_decay=1e-3 -- TEMP_results/TEMP_rq1_dnn_hyperparameter_search_2026-08-19.tex) for
# transfer to PINN. Result (2026-08-22, batch_size=256): flat-to-worse for both variants (see
# temp_results_pinn/RESULTS_TABLE.md #3) -- the Aug-19 win turned out to be a batch_size=512
# artefact specific to that sweep, not a real improvement. Defaults below reverted to the
# project's original values accordingly. Use --output-dir-name to point at a fresh directory
# whenever hyperparameters differ from a previous run (e.g. testing batch_size=512 next).
#
# Isolation: writes to outputs/<output-dir-name>/ -- cannot collide with the existing Set3
# result (full_rerun_cluster/), the lambda ablation (CORRECTED_2026-08-22_lambda_ablation/), or
# the Set2/Set4 sweep (CORRECTED_2026-08-22_pinn_set_sweep/).
#
# Run directly (cluster login node, tiny settings, to smoke-test before a real sbatch submit):
#   PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_lr_check.py \
#     --fold-index 0 --variant k --max-epochs 3 --patience 2
#
# Normally launched via temp_results_pinn/jobs/run_pinn_fix_lr_check_cluster.sh (sbatch).

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

OUTPUT_DIR_ROOT = Path(__file__).resolve().parents[1] / "outputs"
FEATURE_SET = "nested_set3_gated_terrain_wind_vif"  # fixed -- directly comparable to Table 3's Set3 number


def unscale(scaled_tensor, scaler):
    return scaled_tensor.cpu().numpy().flatten() * scaler.scale_[0] + scaler.mean_[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-index", type=int, required=True, help="0-4, which spatial_block_kfold fold to hold out as test.")
    parser.add_argument("--variant", choices=["ymax", "k"], required=True, help="'ymax' = y_max-only fix, 'k' = y_max+k fix.")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Project default. The Aug-19 DNN-sweep-derived value (0.001) was tested 2026-08-22 and found flat-to-worse -- see temp_results_pinn/RESULTS_TABLE.md #3.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Project default. The Aug-19 DNN-sweep-derived value (1e-3) was tested 2026-08-22 and found flat-to-worse -- see temp_results_pinn/RESULTS_TABLE.md #3.")
    parser.add_argument("--cohort", default="4survey")
    parser.add_argument("--split-type", default="spatial_block_kfold")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256, help="PINN's own default -- set explicitly (not relying on the module default) so it's recorded in the summary JSON.")
    parser.add_argument("--output-dir-name", default="CORRECTED_2026-08-22_lr_check", help="Which subdirectory of outputs/ to write to -- change this whenever batch_size/learning_rate/weight_decay differ from a previous run, so results never collide.")
    args = parser.parse_args()

    if not (0 <= args.fold_index < args.n_folds):
        raise ValueError(f"--fold-index must be in [0, {args.n_folds}), got {args.fold_index}")

    fold_dir = OUTPUT_DIR_ROOT / args.output_dir_name / args.variant / f"fold_{args.fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_path = fold_dir / f"pinn_{args.variant}_lr_check_summary.json"
    if summary_path.exists():
        print(f"{summary_path} already exists -- delete it first to redo this fold/variant. Exiting without retraining.")
        return

    device = select_device()
    print(f"Device: {device}  Fold: {args.fold_index}  variant={args.variant}  "
          f"learning_rate={args.learning_rate}  weight_decay={args.weight_decay}")

    feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET]
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
    # monkeypatch pattern already used and validated by the DNN's Aug-19 sweep script, restored
    # after training.
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
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
    finally:
        pinn_module.WEIGHT_DECAY = original_weight_decay

    preds = unscale(predict(model, age_test, other_test, terrain_test), scaler_height)
    target_unscaled = unscale(target_test, scaler_height)
    metrics = compute_metrics(target_unscaled, preds)
    elapsed = time.time() - t0

    print(f"fold {args.fold_index} variant={args.variant}: R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f} "
          f"MAE={metrics['mae']:.4f} ({len(history)} epochs, {elapsed:.1f}s)")

    history.to_csv(fold_dir / f"pinn_{args.variant}_lr_check_history.csv", index=False)
    with open(summary_path, "w") as f:
        json.dump({**metrics, "n_epochs": len(history), "elapsed_seconds": elapsed,
                   "fold_index": args.fold_index, "variant": args.variant,
                   "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                   "batch_size": args.batch_size, "feature_set": FEATURE_SET}, f, indent=2)
    print(f"Saved -> {summary_path}")


if __name__ == "__main__":
    main()

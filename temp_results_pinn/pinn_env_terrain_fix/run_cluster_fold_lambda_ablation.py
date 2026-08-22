# Lambda (physics-loss weight) ablation for the CORRECTED (fixed-forward-pass) PINN/PINN-k -- a
# light copy of run_cluster_fold.py, varying --physics-weight instead of --feature-set, fixed to
# Set3 (nested_set3_gated_terrain_wind_vif, the set with the already-trusted, corrected Table 3
# numbers at physics_weight=1.0) so this ablation is directly comparable to the existing result.
#
# Writes to a separate, clearly-named directory (CORRECTED_2026-08-22_lambda_ablation/) -- cannot
# collide with the existing Set3 result (full_rerun_cluster/) or the Set2/Set4 sweep
# (CORRECTED_2026-08-22_pinn_set_sweep/).
#
# Purpose: the OLD lambda ablation already in the main draft ("Testing lambda_phys in {0,1,2}...")
# used the pre-fix architecture and is explicitly disclosed as such -- not valid evidence for the
# corrected model. This reruns the same idea with the real, corrected code, so the result can
# actually be trusted and then used as the physics_weight for the Set2/Set4 sweep. Originally
# PINN-k only (the interpretability-story variant); extended 2026-08-22 to cover plain PINN
# (y_max only) too, for a complete picture across both Table 3 rows. Note: plain PINN has no k
# sub-network (k is a fixed global constant for that variant), so param_correlation_ymax_k is
# only computed/meaningful for the k variant.
#
# Run directly (cluster login node, tiny settings, to smoke-test before a real sbatch submit):
#   PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/run_cluster_fold_lambda_ablation.py \
#     --fold-index 0 --variant k --physics-weight 0.5 --max-epochs 3 --patience 2
#
# Normally launched via temp_results_pinn/jobs/run_pinn_fix_lambda_ablation_cluster.sh (sbatch).

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
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "CORRECTED_2026-08-22_lambda_ablation"
FEATURE_SET = "nested_set3_gated_terrain_wind_vif"  # fixed -- directly comparable to the trusted lambda=1.0 Table 3 number


def unscale(scaled_tensor, scaler):
    return scaled_tensor.cpu().numpy().flatten() * scaler.scale_[0] + scaler.mean_[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-index", type=int, required=True, help="0-4, which spatial_block_kfold fold to hold out as test.")
    parser.add_argument("--variant", choices=["ymax", "k"], required=True, help="'ymax' = y_max-only fix (no k sub-network, no param_correlation), 'k' = y_max+k fix.")
    parser.add_argument("--physics-weight", type=float, required=True, help="lambda_phys to test, e.g. 0, 0.5, 1.0, 2.0.")
    parser.add_argument("--cohort", default="4survey")
    parser.add_argument("--split-type", default="spatial_block_kfold")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Project default -- override once the LR-transfer check (run_cluster_fold_lr_check.py) finds a better value.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Project default -- override once the LR-transfer check finds a better value.")
    parser.add_argument("--batch-size", type=int, default=256, help="Matches PINN's own default -- set explicitly (not relying on the module default) so it's recorded in the summary JSON.")
    args = parser.parse_args()

    if not (0 <= args.fold_index < args.n_folds):
        raise ValueError(f"--fold-index must be in [0, {args.n_folds}), got {args.fold_index}")

    if args.variant == "ymax":
        from temp_results_pinn.pinn_env_terrain_fix import pinn_env_terrain_fix as pinn_module
        from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_fix import fit, predict
    else:
        from temp_results_pinn.pinn_env_terrain_fix import pinn_env_terrain_k_fix as pinn_module
        from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_k_fix import fit, predict, predict_y_max, predict_k

    lambda_label = f"lambda{args.physics_weight}".replace(".", "p")
    fold_dir = OUTPUT_DIR / lambda_label / args.variant / f"fold_{args.fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_path = fold_dir / f"pinn_{args.variant}_fixed_summary.json"
    if summary_path.exists():
        print(f"{summary_path} already exists -- delete it first to redo this fold/lambda. Exiting without retraining.")
        return

    device = select_device()
    print(f"Device: {device}  Fold: {args.fold_index}  variant={args.variant}  physics_weight (lambda): {args.physics_weight}  Feature set: {FEATURE_SET}")

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
            physics_weight=args.physics_weight,  # <-- the thing being ablated; trajectory_weight stays at its default
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
    finally:
        pinn_module.WEIGHT_DECAY = original_weight_decay
    preds = unscale(predict(model, age_test, other_test, terrain_test), scaler_height)
    target_unscaled = unscale(target_test, scaler_height)
    metrics = compute_metrics(target_unscaled, preds)
    elapsed = time.time() - t0

    # Parameter separability (y_max_i vs k_i correlation) -- a real physics-loss ablation should
    # check whether raising lambda actually improves the thing physics loss is meant to improve
    # (trajectory consistency / sensible separated parameters), not just R2. Only meaningful for
    # the k variant -- plain PINN has no per-plot k (it's a fixed global constant there).
    extra_fields = {}
    param_corr_str = "n/a (ymax variant, no k sub-network)"
    if args.variant == "k":
        y_max_per_row = predict_y_max(model, terrain_test, cr_params["y_max"]).cpu().numpy().flatten()
        k_per_row = predict_k(model, terrain_test, cr_params["k"]).cpu().numpy().flatten()
        import numpy as np
        param_correlation = float(np.corrcoef(y_max_per_row, k_per_row)[0, 1])
        extra_fields["param_correlation_ymax_k"] = param_correlation
        param_corr_str = f"{param_correlation:.4f}"

    print(f"fold {args.fold_index} variant={args.variant} lambda={args.physics_weight}: R2={metrics['r2']:.4f} "
          f"RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} param_corr(y_max,k)={param_corr_str} "
          f"({len(history)} epochs, {elapsed:.1f}s)")

    history.to_csv(fold_dir / f"pinn_{args.variant}_fixed_history.csv", index=False)
    with open(summary_path, "w") as f:
        json.dump({**metrics, "n_epochs": len(history), "elapsed_seconds": elapsed,
                   "fold_index": args.fold_index, "variant": args.variant, "physics_weight": args.physics_weight,
                   **extra_fields,
                   "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                   "batch_size": args.batch_size}, f, indent=2)
    print(f"Saved -> {summary_path}")


if __name__ == "__main__":
    main()

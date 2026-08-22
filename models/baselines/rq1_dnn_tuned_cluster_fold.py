# Re-runs the DNN on all 5 spatial_block_kfold folds (Set3, the Table 3 feature set), one
# process per fold so the cluster can run all 5 in parallel, with learning_rate/weight_decay/
# batch_size all overridable and recorded in the summary JSON.
#
# History: originally written to test the Aug-19 single-split sweep's winning config
# (learning_rate=0.001, weight_decay=1e-3 -- see
# TEMP_results/TEMP_rq1_dnn_hyperparameter_search_2026-08-19.tex). Result (2026-08-22, proper
# 5-fold CV, batch_size=256): flat-to-worse (0.6426 vs the trusted 0.6550) -- the Aug-19 win was
# a batch_size=512 artefact, not a real improvement at Table 3's actual batch_size=256. See
# temp_results_pinn/RESULTS_TABLE.md #3 for the full writeup. Defaults below reverted to the
# project's original values accordingly; use --output-dir-name to point at a fresh directory
# whenever hyperparameters differ from a previous run.
#
# Isolation: writes to a new, separate output directory (outputs/<output-dir-name>/) -- CANNOT
# collide with or overwrite the existing Table 3 DNN results
# (outputs/spatial_block_kfold/rq1_dnn_env_terrain_..._seed42/).
#
# Run directly (cluster login node, tiny settings, to smoke-test before a real sbatch submit):
#   PYTHONPATH=. python models/baselines/rq1_dnn_tuned_cluster_fold.py \
#     --fold-index 0 --max-epochs 3 --patience 2
#
# Normally launched via temp_results_pinn/jobs/run_dnn_tuned_cluster.sh (sbatch).

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common.metrics import compute_metrics
from models.common.torch_data import (
    ENV_TERRAIN_FEATURE_SETS,
    build_terrain_tensor,
    build_tensors,
    encode_thinning_status,
    fit_scalers,
    fit_terrain_scaler,
    load_split_table_with_terrain,
    select_device,
)
from models.common.splits import SPLIT_SEED
from models.dnn_env_terrain import dnn_env_terrain as dnn_module
from models.dnn_env_terrain.dnn_env_terrain import fit as fit_dnn, predict as predict_dnn

OUTPUT_DIR_ROOT = Path(__file__).resolve().parents[2] / "outputs"
FEATURE_SET = "nested_set3_gated_terrain_wind_vif"  # Table 3's feature set, so this is directly comparable


def unscale(scaled_tensor, scaler):
    return scaled_tensor.cpu().numpy().flatten() * scaler.scale_[0] + scaler.mean_[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-index", type=int, required=True, help="0-4, which spatial_block_kfold fold to hold out as test.")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Project default. The Aug-19 'tuned' value (0.001) was tested 2026-08-22 and found to be flat-to-worse at batch_size=256 -- see temp_results_pinn/RESULTS_TABLE.md #3.")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Project default. The Aug-19 'tuned' value (1e-3) was tested 2026-08-22 and found to be flat-to-worse at batch_size=256 -- see temp_results_pinn/RESULTS_TABLE.md #3.")
    parser.add_argument("--batch-size", type=int, default=256, help="DNN's module default (changed from 512 on 2026-08-22 to match PINN/PINN-k's default for a fair comparison).")
    parser.add_argument("--output-dir-name", default="CORRECTED_2026-08-22_dnn_tuned_cluster", help="Which subdirectory of outputs/ to write to -- change this whenever batch_size/learning_rate/weight_decay differ from a previous run, so results never collide.")
    parser.add_argument("--cohort", default="4survey")
    parser.add_argument("--split-type", default="spatial_block_kfold")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0 <= args.fold_index < args.n_folds):
        raise ValueError(f"--fold-index must be in [0, {args.n_folds}), got {args.fold_index}")

    fold_dir = OUTPUT_DIR_ROOT / args.output_dir_name / f"fold_{args.fold_index}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    summary_path = fold_dir / "dnn_tuned_summary.json"
    if summary_path.exists():
        print(f"{summary_path} already exists -- delete it first to redo this fold. Exiting without retraining.")
        return

    device = select_device()
    print(f"Device: {device}  Fold: {args.fold_index}  learning_rate={args.learning_rate}  "
          f"weight_decay={args.weight_decay}  batch_size={args.batch_size}")

    feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET]
    split_df = load_split_table_with_terrain(
        args.cohort, args.split_type, feature_columns, split_seed=SPLIT_SEED,
        k_folds=args.n_folds, held_out_fold=args.fold_index,
    )
    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "val"]
    test_df = split_df[split_df["split"] == "test"]
    print(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    scaler_age, scaler_other_features, scaler_height = fit_scalers(train_df)
    scaler_terrain = fit_terrain_scaler(train_df, feature_columns)
    encoded_column_names = encode_thinning_status(train_df).columns.tolist()

    age_train, other_train_noenv, target_train = build_tensors(
        train_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
    age_val, other_val_noenv, target_val = build_tensors(
        val_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
    age_test, other_test_noenv, target_test = build_tensors(
        test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
    terrain_train = build_terrain_tensor(train_df, scaler_terrain, feature_columns, device)
    terrain_val = build_terrain_tensor(val_df, scaler_terrain, feature_columns, device)
    terrain_test = build_terrain_tensor(test_df, scaler_terrain, feature_columns, device)

    import torch
    other_train = torch.cat([other_train_noenv, terrain_train], dim=1)
    other_val = torch.cat([other_val_noenv, terrain_val], dim=1)
    other_test = torch.cat([other_test_noenv, terrain_test], dim=1)
    n_other_features = other_train.shape[1]

    # weight_decay is a module-level constant inside dnn_env_terrain.py's build_optimizer, not a
    # fit() kwarg -- same monkeypatch pattern already used and validated by the Aug-19 sweep
    # script (models/baselines/rq1_dnn_hyperparameter_search.py), restored after training.
    original_weight_decay = dnn_module.WEIGHT_DECAY
    dnn_module.WEIGHT_DECAY = args.weight_decay
    try:
        t0 = time.time()
        model, _, history = fit_dnn(
            age_train, other_train, target_train,
            age_val, other_val, target_val,
            n_other_features, device, args.seed,
            max_epochs=args.max_epochs, early_stopping_patience=args.patience,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
        )
    finally:
        dnn_module.WEIGHT_DECAY = original_weight_decay

    preds = unscale(predict_dnn(model, age_test, other_test), scaler_height)
    target_unscaled = unscale(target_test, scaler_height)
    metrics = compute_metrics(target_unscaled, preds)
    elapsed = time.time() - t0

    print(f"fold {args.fold_index}: R2={metrics['r2']:.4f} RMSE={metrics['rmse']:.4f} "
          f"MAE={metrics['mae']:.4f} ({len(history)} epochs, {elapsed:.1f}s)")

    history.to_csv(fold_dir / "dnn_tuned_history.csv", index=False)
    with open(summary_path, "w") as f:
        json.dump({**metrics, "n_epochs": len(history), "elapsed_seconds": elapsed,
                   "fold_index": args.fold_index, "learning_rate": args.learning_rate,
                   "weight_decay": args.weight_decay, "batch_size": args.batch_size,
                   "feature_set": FEATURE_SET}, f, indent=2)
    print(f"Saved -> {summary_path}")


if __name__ == "__main__":
    main()

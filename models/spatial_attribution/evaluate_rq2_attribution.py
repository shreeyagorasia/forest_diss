# Run as: python -m models.spatial_attribution.evaluate_rq2_attribution --cohort 4survey --set-name nested_set2_top5
#     or: python -m models.spatial_attribution.evaluate_rq2_attribution --cohort 4survey --set-name nested_set2_top5 --split-type spatial_block_kfold --fold-index 0
#
# RQ2's EVALUATE step -- always local, never the cluster (this project's standing convention).
# Loads the Elastic Net/XGBoost checkpoints run_rq2_attribution.py saved, re-derives the IDENTICAL
# test split (same cohort/split_seed/fold -- deterministic given the same inputs, so the split
# itself was never saved to disk), and scores it. NLME has nothing to do here -- see
# run_rq2_attribution.py's own comment for why its diagnostics are already complete at fit time.

import argparse
import json

import joblib
import pandas as pd

from models.common.metrics import compute_metrics
from models.common.run_logging import RunTimer, format_error, write_run_log, write_started_marker
from models.common.saving import model_output_dir
from models.common.splits import (
    DEFAULT_K_FOLDS,
    SPATIAL_BLOCK_COL,
    SPATIAL_BUFFER_METRES,
    SPLIT_SEED,
    spatial_block_split,
    spatial_kfold_split,
)
from models.elasticnet_environmental.elasticnet_environmental import predict_with_columns as en_predict_with_columns
from models.xgb_environmental.data import load_plots_for_cohort
from models.xgb_environmental.xgb_environmental import predict_with_columns as xgb_predict_with_columns

TARGET_COLUMN = "mean_cr_residual"
MODEL_NAME = "rq2_attribution"


def evaluate_one_set(
    cohort, set_name, split_type="spatial_block", split_seed=SPLIT_SEED,
    held_out_fold=0, k_folds=DEFAULT_K_FOLDS,
):
    fold_suffix = f", fold={held_out_fold}/{k_folds}" if split_type == "spatial_block_kfold" else ""
    print(f"===== RQ2 attribution EVALUATE: {cohort} / {set_name} ({split_type}{fold_suffix}) =====")
    timer = RunTimer().start()
    run_name = f"{MODEL_NAME}_{set_name}"
    hyperparameters = {"set_name": set_name, "split_seed": split_seed}
    if split_type == "spatial_block_kfold":
        hyperparameters.update({"held_out_fold": held_out_fold, "k_folds": k_folds})
    attempt_id = write_started_marker(
        model_name=run_name, cohort=cohort, split_type=split_type,
        run_phase="evaluate", is_test_run=False, device="cpu", hyperparameters=hyperparameters,
    )

    try:
        if split_type == "spatial_block_kfold":
            output_dir = model_output_dir(run_name, cohort, f"fold_{held_out_fold}", split_type=split_type)
        else:
            output_dir = model_output_dir(run_name, cohort, split_type=split_type)

        with open(output_dir / "feature_columns.json") as f:
            feature_columns = json.load(f)
        en_fitted = joblib.load(output_dir / "elastic_net_model.joblib")
        xgb_model = joblib.load(output_dir / "xgboost_model.joblib")

        plots_df = load_plots_for_cohort(cohort).copy()
        if split_type == "spatial_block_kfold":
            plots_df["split"] = spatial_kfold_split(
                plots_df, block_col=SPATIAL_BLOCK_COL, k=k_folds, held_out_fold=held_out_fold,
                buffer_distance=SPATIAL_BUFFER_METRES, seed=split_seed,
            )
        else:
            plots_df["split"] = spatial_block_split(
                plots_df, block_col=SPATIAL_BLOCK_COL, buffer_distance=SPATIAL_BUFFER_METRES, seed=split_seed,
            )
        test_df = plots_df[plots_df["split"] == "test"]
        print(f"  test rows: {len(test_df):,}  features: {len(feature_columns)}")

        en_predictions = en_predict_with_columns(test_df, en_fitted, feature_columns)
        xgb_predictions = xgb_predict_with_columns(test_df, xgb_model, feature_columns)

        metrics = {
            "elastic_net": compute_metrics(test_df[TARGET_COLUMN], en_predictions),
            "xgboost": compute_metrics(test_df[TARGET_COLUMN], xgb_predictions),
        }
        print(f"  Elastic Net R2={metrics['elastic_net']['r2']:.4f}  XGBoost R2={metrics['xgboost']['r2']:.4f}")

        predictions_df = pd.DataFrame({
            "identification": test_df["identification"].values,
            "split": "test",
            "observed": test_df[TARGET_COLUMN].values,
            "elastic_net_predicted": en_predictions,
            "xgboost_predicted": xgb_predictions,
        })
        predictions_df.to_csv(output_dir / "predictions.csv", index=False)
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Saved predictions + metrics -> {output_dir}")

        write_run_log(
            attempt_id=attempt_id, model_name=run_name, cohort=cohort, split_type=split_type,
            run_phase="evaluate", status="success", is_test_run=False, device="cpu",
            hyperparameters=hyperparameters,
            metrics={"elastic_net_r2": metrics["elastic_net"]["r2"], "xgboost_r2": metrics["xgboost"]["r2"]},
            error=None, output_dir=str(output_dir), runtime_seconds=timer.elapsed_seconds(),
        )
        return metrics

    except Exception as error:
        write_run_log(
            attempt_id=attempt_id, model_name=run_name, cohort=cohort, split_type=split_type,
            run_phase="evaluate", status="failed", is_test_run=False, device="cpu",
            hyperparameters=hyperparameters, metrics=None, error=format_error(error),
            output_dir=None, runtime_seconds=timer.elapsed_seconds(),
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="4survey", choices=["4survey", "6survey"])
    parser.add_argument(
        "--set-name", default="nested_set2_top5",
        choices=["nested_set2_top5", "nested_set3_gated_terrain_wind_vif", "nested_set4_gated_all_vif", "nested_set5_all_ungated_vif"],
    )
    parser.add_argument("--split-type", default="spatial_block", choices=["spatial_block", "spatial_block_kfold"])
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_K_FOLDS)
    args = parser.parse_args()

    evaluate_one_set(
        args.cohort, args.set_name, split_type=args.split_type, split_seed=args.split_seed,
        held_out_fold=args.fold_index, k_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()

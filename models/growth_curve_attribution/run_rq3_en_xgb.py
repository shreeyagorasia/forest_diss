# Run as: python -m models.growth_curve_attribution.run_rq3_en_xgb --cohort 4survey --set-name nested_set2_top5
#
# RQ3's Elastic Net + XGBoost driver for the new rank-aggregate environmental-feature methodology
# -- calls broad_environmental_check.py's run_columns() (the new raw-column-list sibling of
# run_scope(), see that file) with a Set loaded from documentation/env_feature_sets_manifest.csv,
# instead of a named SCOPE_GROUPS entry. run_columns() already runs the full 5-fold spatial CV
# internally and returns pooled results in one call -- unlike DNN/PINN, there's no "one job per
# fold" here, one job per (cohort, set_name) already IS the pooled 5-fold number.
#
# Elastic Net for RQ3 is deliberately NOT VIF-screened (a decision, not an oversight -- see
# documentation/experiment_log.md's 2026-08-10 entry): read its coefficient-direction/stability
# results with the "not VIF-screened" caveat the plan requires, distinct from RQ2's NLME/EN,
# which ARE VIF-screened.

import argparse
import json

from models.common.run_logging import RunTimer, format_error, write_run_log, write_started_marker
from models.common.saving import model_output_dir
from models.growth_curve_attribution.broad_environmental_check import run_columns
from models.xgb_environmental.feature_set_builder import load_feature_set

MODEL_NAME = "rq3_en_xgb"


def run_one_set(cohort, set_name, k_folds=5, seed=42):
    print(f"===== RQ3 EN/XGBoost: {cohort} / {set_name} =====")
    timer = RunTimer().start()
    run_name = f"{MODEL_NAME}_{set_name}"
    hyperparameters = {"set_name": set_name, "k_folds": k_folds, "seed": seed}
    attempt_id = write_started_marker(
        model_name=run_name, cohort=cohort, split_type="spatial_block_kfold",
        run_phase="fit_and_evaluate", is_test_run=False, device="cpu", hyperparameters=hyperparameters,
    )

    try:
        raw_columns = load_feature_set("RSQ3", set_name)
        results_df, predictions, fold_counts = run_columns(cohort, raw_columns, k=k_folds, seed=seed)
        print(results_df[["method", "pooled_r2", "per_fold_r2_mean", "per_fold_r2_std"]].to_string(index=False))

        output_dir = model_output_dir(run_name, cohort, split_type="spatial_block_kfold")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_dir / "metrics.csv", index=False)
        predictions.to_csv(output_dir / "predictions.csv", index=False)
        with open(output_dir / "fold_counts.json", "w") as f:
            json.dump(fold_counts, f, indent=2, default=str)
        print(f"  Saved metrics + predictions -> {output_dir}")

        pooled_r2_by_method = dict(zip(results_df["method"], results_df["pooled_r2"]))
        write_run_log(
            attempt_id=attempt_id, model_name=run_name, cohort=cohort,
            split_type="spatial_block_kfold", run_phase="fit_and_evaluate", status="success",
            is_test_run=False, device="cpu", hyperparameters=hyperparameters,
            metrics=pooled_r2_by_method, error=None, output_dir=str(output_dir),
            runtime_seconds=timer.elapsed_seconds(),
        )
        return results_df

    except Exception as error:
        write_run_log(
            attempt_id=attempt_id, model_name=run_name, cohort=cohort,
            split_type="spatial_block_kfold", run_phase="fit_and_evaluate", status="failed",
            is_test_run=False, device="cpu", hyperparameters=hyperparameters,
            metrics=None, error=format_error(error), output_dir=None,
            runtime_seconds=timer.elapsed_seconds(),
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="4survey", choices=["4survey", "6survey"])
    parser.add_argument(
        "--set-name", default="nested_set2_top5",
        choices=["nested_set2_top5", "nested_set3_gated_terrain_wind", "nested_set4_gated_all", "nested_set5_all_ungated"],
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_one_set(args.cohort, args.set_name, k_folds=args.k_folds, seed=args.seed)


if __name__ == "__main__":
    main()

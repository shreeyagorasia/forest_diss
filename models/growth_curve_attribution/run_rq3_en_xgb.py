# Run as: python -m models.growth_curve_attribution.run_rq3_en_xgb --cohort 4survey --set-name nested_set2_top10
#
# RQ3's Elastic Net + XGBoost driver for the new rank-aggregate environmental-feature methodology
#. Calls broad_environmental_check.py's run_columns() (the new raw-column-list sibling of
# run_scope(), see that file) with a Set loaded from documentation/env_feature_sets_manifest.csv,
# instead of a named SCOPE_GROUPS entry. run_columns() already runs the full 5-fold spatial CV
# internally and returns pooled results in one call. Unlike DNN/PINN, there's no "one job per
# fold" here, one job per (cohort, set_name) already IS the pooled 5-fold number.
#
# UPDATED 2026-08-10 (documentation/methodlogy_env_setpick.md): RQ3's Sets are now VIF-screened
# too (Set2 with skip-and-backfill, Set3/4/5 with iterative reduction, threshold 5.0). This
# reverses the earlier "deliberately NOT VIF-screened" decision recorded in
# documentation/experiment_log.md's 2026-08-10 entry. That entry's caveat is now STALE, not
# current guidance. RQ3's Elastic Net coefficients from `nested_set{2,3,4,5}_..._vif` no longer
# need the "read direction cautiously, not VIF-screened" hedge in the write-up. Reference-level
# dropping was also added for RQ3's three one-hot categoricals (one level per category dropped
# before screening) to fix a real `VIF = inf` structural-singularity bug the old, un-VIF'd sets
# never hit.

import argparse
import json

import pandas as pd

from models.common.run_logging import RunTimer, format_error, write_run_log, write_started_marker
from models.common.saving import model_output_dir
from models.elasticnet_environmental.elasticnet_environmental import get_coefficients_table
from models.growth_curve_attribution.broad_environmental_check import run_columns
from models.xgb_environmental.feature_set_builder import load_feature_set

MODEL_NAME = "rq3_en_xgb"


def save_fold_attribution(fold_models, output_dir):
    # Both models are cheap enough (seconds to fit) that keeping their attribution numbers around
    # per fold costs nothing. This is the "which variable matters" table the RQ3 chapter
    # section actually needs, matching what RQ2b's fit step already saves for Elastic Net
    # (see elastic_net_coefficients.csv there) and what GNNWR gets for free from its own
    # geographically-weighted architecture (per-row local coefficients baked into result_data).
    en_rows = []
    xgb_rows = []
    for fold_entry in fold_models:
        fold = fold_entry["fold"]

        coefficients = get_coefficients_table(fold_entry["elastic_net_model"])
        en_rows.append(pd.DataFrame({"fold": fold, "feature": coefficients.index, "coefficient": coefficients.values}))

        # XGBoost's feature_importances_ (gain-based) is already computed as a byproduct of
        # fitting. No extra pass over the data needed, so there's no cost reason to leave it
        # unsaved. Per-row SHAP values would need the held-out X matrix, which run_spatial_cv()
        # doesn't return. Left for later if a per-row/local attribution map turns out to be
        # wanted, since that's a bigger change than this cheap fix.
        xgb_model = fold_entry["xgboost_model"]
        importances = pd.Series(xgb_model.feature_importances_, index=xgb_model.feature_names_in_)
        importances = importances.reindex(importances.sort_values(ascending=False).index)
        xgb_rows.append(pd.DataFrame({"fold": fold, "feature": importances.index, "gain_importance": importances.values}))

    pd.concat(en_rows, ignore_index=True).to_csv(output_dir / "elastic_net_coefficients_by_fold.csv", index=False)
    pd.concat(xgb_rows, ignore_index=True).to_csv(output_dir / "xgboost_gain_importance_by_fold.csv", index=False)


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
        results_df, predictions, fold_counts, fold_models = run_columns(cohort, raw_columns, k=k_folds, seed=seed)
        print(results_df[["method", "pooled_r2", "per_fold_r2_mean", "per_fold_r2_std"]].to_string(index=False))

        output_dir = model_output_dir(run_name, cohort, split_type="spatial_block_kfold")
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_dir / "metrics.csv", index=False)
        # predictions now carries x/y per row too (spatial_cv_check.py, 2026-08-10). No extra
        # join needed here for a later map/outlier-zoom plot.
        predictions.to_csv(output_dir / "predictions.csv", index=False)
        with open(output_dir / "fold_counts.json", "w") as f:
            json.dump(fold_counts, f, indent=2, default=str)
        save_fold_attribution(fold_models, output_dir)
        print(f"  Saved metrics + predictions + per-fold attribution -> {output_dir}")

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
        "--set-name", default="nested_set2_top10",
        choices=["nested_set2_top10", "nested_set3_gated_terrain_wind_vif", "nested_set4_gated_all_vif", "nested_set5_all_ungated_vif"],
    )
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_one_set(args.cohort, args.set_name, k_folds=args.k_folds, seed=args.seed)


if __name__ == "__main__":
    main()

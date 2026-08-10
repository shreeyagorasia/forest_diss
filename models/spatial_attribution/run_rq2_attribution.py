# Run as: python -m models.spatial_attribution.run_rq2_attribution --cohort 4survey --set-name nested_set2_top10
#     or: python -m models.spatial_attribution.run_rq2_attribution --cohort 4survey --set-name nested_set2_top10 --split-type spatial_block_kfold --fold-index 0
#
# RQ2's FIT step -- deliberately separate from evaluate_rq2_attribution.py, matching this
# project's dnn_noenv/pinn_noenv convention (fit on the cluster, evaluate locally afterwards),
# even though NLME/Elastic Net/XGBoost are all cheap enough that the split isn't about saving
# compute -- it's about keeping the same workflow shape everywhere in this project.
#
# NLME (fit_nlme) is genuinely different from Elastic Net/XGBoost here: it has no held-out
# test-set prediction step at all -- its "evaluation" (fixed_effects_table, residual normality,
# variance explained) is computed directly from the fitted model on its OWN training data, not by
# scoring test rows. So NLME's diagnostics are saved here, at fit time, not in the evaluate
# script -- there is nothing for evaluate_rq2_attribution.py to do for NLME.
#
# Elastic Net and XGBoost's raw fitted model objects are saved via joblib (matching how every
# PyTorch model in this project saves a checkpoint at fit time and reloads it at evaluate time)
# -- evaluate_rq2_attribution.py loads them back, re-derives the identical test split (same
# cohort/split_seed/fold -- spatial_kfold_split()/spatial_block_split() are deterministic given
# the same inputs, so the split itself never needs to be saved to disk), and scores it.

import argparse
import json

import joblib

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
from models.elasticnet_environmental.elasticnet_environmental import fit_with_columns as en_fit_with_columns
from models.elasticnet_environmental.elasticnet_environmental import get_coefficients_table
from models.spatial_attribution.nlme import (
    check_residual_normality,
    fit_nlme,
    fixed_effects_table,
    variance_explained_by_fixed_effects,
)
from models.xgb_environmental.data import load_plots_for_cohort
from models.xgb_environmental.feature_set_builder import load_feature_set
from models.xgb_environmental.xgb_environmental import fit_with_columns as xgb_fit_with_columns

TARGET_COLUMN = "mean_cr_residual"
GROUP_COLUMN = "cpmt"
MODEL_NAME = "rq2_attribution"


def fit_one_set(
    cohort, set_name, split_type="spatial_block", split_seed=SPLIT_SEED,
    held_out_fold=0, k_folds=DEFAULT_K_FOLDS,
):
    fold_suffix = f", fold={held_out_fold}/{k_folds}" if split_type == "spatial_block_kfold" else ""
    print(f"===== RQ2 attribution FIT: {cohort} / {set_name} ({split_type}{fold_suffix}) =====")
    timer = RunTimer().start()
    run_name = f"{MODEL_NAME}_{set_name}"
    hyperparameters = {"set_name": set_name, "split_seed": split_seed}
    if split_type == "spatial_block_kfold":
        hyperparameters.update({"held_out_fold": held_out_fold, "k_folds": k_folds})
    attempt_id = write_started_marker(
        model_name=run_name, cohort=cohort, split_type=split_type,
        run_phase="fit", is_test_run=False, device="cpu", hyperparameters=hyperparameters,
    )

    try:
        feature_columns = load_feature_set("RSQ2", set_name)
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
        train_df = plots_df[plots_df["split"] == "train"]
        print(f"  train rows: {len(train_df):,}  features: {len(feature_columns)}")

        if split_type == "spatial_block_kfold":
            output_dir = model_output_dir(run_name, cohort, f"fold_{held_out_fold}", split_type=split_type)
        else:
            output_dir = model_output_dir(run_name, cohort, split_type=split_type)
        output_dir.mkdir(parents=True, exist_ok=True)

        # ---- NLME: fit + save its own diagnostics now -- no held-out evaluate step exists for it ----
        nlme_result, _ = fit_nlme(train_df, feature_columns, group_col=GROUP_COLUMN, target_col=TARGET_COLUMN)
        nlme_diagnostics = {
            "fixed_effects": fixed_effects_table(nlme_result).to_dict(orient="index"),
            "residual_normality": check_residual_normality(nlme_result),
            "variance_explained": variance_explained_by_fixed_effects(
                train_df, feature_columns, group_col=GROUP_COLUMN, target_col=TARGET_COLUMN,
            ),
        }
        with open(output_dir / "nlme_fixed_effects.json", "w") as f:
            json.dump(nlme_diagnostics, f, indent=2, default=str)

        # ---- Elastic Net + XGBoost: fit and save the checkpoint, evaluate script scores later ----
        en_fitted = en_fit_with_columns(train_df, feature_columns, target_col=TARGET_COLUMN)
        joblib.dump(en_fitted, output_dir / "elastic_net_model.joblib")
        # Coefficients extracted and saved as their own readable CSV too, not just left buried
        # inside the joblib blob -- this is the "which variable matters" table a later plotting
        # script (e.g. a coefficient forest plot) would otherwise have to reload the whole model
        # object just to get.
        get_coefficients_table(en_fitted).to_csv(output_dir / "elastic_net_coefficients.csv", header=True)

        # XGBoost model saved via its OWN native format (save_model), not joblib -- confirmed by
        # smoke-testing across the cluster/local boundary (2026-08-10): joblib-pickling a raw
        # XGBRegressor and reloading it on a different machine/XGBoost version triggers XGBoost's
        # own documented cross-version-pickle warning and silently gave a badly degraded R2
        # (0.15 instead of the correct ~0.24) -- not an error, just a quietly wrong model. Elastic
        # Net (plain sklearn) had no such issue in the same test, so only XGBoost needs this.
        xgb_model = xgb_fit_with_columns(train_df, feature_columns, target_col=TARGET_COLUMN, n_jobs=1)
        xgb_model.save_model(str(output_dir / "xgboost_model.json"))

        with open(output_dir / "feature_columns.json", "w") as f:
            json.dump(feature_columns, f, indent=2)

        print(f"  Saved NLME diagnostics + EN/XGBoost checkpoints -> {output_dir}")

        write_run_log(
            attempt_id=attempt_id, model_name=run_name, cohort=cohort, split_type=split_type,
            run_phase="fit", status="success", is_test_run=False, device="cpu",
            hyperparameters=hyperparameters, metrics=None, error=None,
            output_dir=str(output_dir), runtime_seconds=timer.elapsed_seconds(), n_rows_fit=len(train_df),
        )

    except Exception as error:
        write_run_log(
            attempt_id=attempt_id, model_name=run_name, cohort=cohort, split_type=split_type,
            run_phase="fit", status="failed", is_test_run=False, device="cpu",
            hyperparameters=hyperparameters, metrics=None, error=format_error(error),
            output_dir=None, runtime_seconds=timer.elapsed_seconds(),
        )
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="4survey", choices=["4survey", "6survey"])
    parser.add_argument(
        "--set-name", default="nested_set2_top10",
        choices=["nested_set2_top10", "nested_set3_gated_terrain_wind_vif", "nested_set4_gated_all_vif", "nested_set5_all_ungated_vif"],
    )
    parser.add_argument("--split-type", default="spatial_block", choices=["spatial_block", "spatial_block_kfold"])
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_K_FOLDS)
    args = parser.parse_args()

    fit_one_set(
        args.cohort, args.set_name, split_type=args.split_type, split_seed=args.split_seed,
        held_out_fold=args.fold_index, k_folds=args.n_folds,
    )


if __name__ == "__main__":
    main()

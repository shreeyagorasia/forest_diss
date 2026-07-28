# Run as: python -m models.xgb_environmental.run_xgb_environmental --cohort 4survey
#     or: python -m models.xgb_environmental.run_xgb_environmental --cohort 6survey
#     or: python -m models.xgb_environmental.run_xgb_environmental (both cohorts, one after another)
#
# Fits XGBoost to predict each plot's mean Chapman-Richards residual from its environmental
# variables (terrain, wind, soil, climate, plus the GPKG-derived geometry and neighbour
# features) -- three different feature sets are tried, see FEATURE_SETS in
# xgb_environmental.py. Unlike the DNN/PINN models in this repo, XGBoost fits in seconds on this
# data size, so there is no separate cluster fit-then-local-evaluate split -- this one script
# fits, evaluates, and computes SHAP values all in one pass.
#
# Uses spatial_block_split() (whole forestry compartments held out), the same split every other
# terrain/wind-aware model in this repo uses, for the same reason: a random plot-level split
# would put neighbouring plots -- which share near-identical terrain/wind conditions -- on both
# sides of the split, letting the model look like it generalises when it has really just
# memorised local conditions.

import argparse
import json

from models.common.metrics import compute_metrics
from models.common.run_logging import RunTimer, format_error, write_run_log, write_started_marker
from models.common.saving import model_output_dir as output_dir
from models.common.splits import SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, spatial_block_split
from models.xgb_environmental.data import load_plots_for_cohort
from models.xgb_environmental.xgb_environmental import FEATURE_SETS, compute_shap_values, fit, predict

COHORTS = ["4survey", "6survey"]
SEED = 42
DEVICE = "cpu"  # XGBoost here runs on plain CPU, no GPU needed for this data size

# SHAP values are only computed for these two feature sets -- "all_environmental" is the main
# deliverable, and "all_environmental_no_neighbour" directly answers the standing spatial-lag
# caveat. "terrain_and_wind_only" is a metrics-only comparison, kept for continuity with the
# dissertation plan's original XGB-A/B framing -- no SHAP needed for that one.
FEATURE_SETS_NEEDING_SHAP = ["all_environmental", "all_environmental_no_neighbour"]


def run_one_feature_set(cohort, feature_set_name, plots_df):
    print(f"\n--- {cohort} / {feature_set_name} ---")

    # spatial_block_split() actually returns FOUR possible labels: "train", "val", "test", and
    # "buffer" (train plots too close to a val/test plot, excluded entirely). Earlier code here
    # only read "train"/"test" and silently dropped every "val" row -- val was neither trained
    # on nor evaluated on, it just vanished. That meant the only real held-out numbers available
    # were test-set numbers, so the ablation/feature-selection work in the Tier-2 notebook ended
    # up repeatedly re-using the TEST set to decide which features to keep -- exactly the kind of
    # "peeking" documentation/handover_2026-07-18.md's own next-steps section had already flagged
    # as a risk. Fixed here by keeping val_df as a real, usable split: val is for any
    # feature-selection/comparison decision, test is touched only once, for the final reported
    # number of whichever feature set was already chosen using val.
    split_labels = spatial_block_split(
        plots_df,
        block_col=SPATIAL_BLOCK_COL,
        buffer_distance=SPATIAL_BUFFER_METRES,
        seed=SEED,
    )
    plots_df = plots_df.copy()
    plots_df["split"] = split_labels

    train_df = plots_df[plots_df["split"] == "train"]
    val_df = plots_df[plots_df["split"] == "val"]
    test_df = plots_df[plots_df["split"] == "test"]
    print(f"  train rows: {len(train_df):,} | val rows: {len(val_df):,} | test rows: {len(test_df):,}")

    hyperparameters = {"seed": SEED, "feature_set": feature_set_name, "n_features": len(FEATURE_SETS[feature_set_name])}
    timer = RunTimer().start()
    attempt_id = write_started_marker(
        model_name=f"xgb_environmental_{feature_set_name}", cohort=cohort, split_type="spatial_block",
        run_phase="fit_and_evaluate", is_test_run=False, device=DEVICE, hyperparameters=hyperparameters,
    )

    try:
        model = fit(train_df, feature_set_name, seed=SEED)

        def scored_metrics(df):
            # mre/accuracy_pct divide by the OBSERVED value -- that formula assumes a height
            # (always positive, far from zero), not a residual (centred near zero, often
            # negative). For a residual target those two fields come out as nonsense (e.g. "109%
            # accuracy"), so they are dropped here rather than saved somewhere they could be
            # mistaken for a real result. r2/rmse/mae/bias are all still meaningful and are kept.
            predictions = predict(df, model, feature_set_name)
            row_metrics = compute_metrics(df["mean_cr_residual"].values, predictions)
            del row_metrics["mre"], row_metrics["accuracy_pct"]
            return row_metrics, predictions

        # val is the ONLY split used for feature-selection/comparison decisions (ablation refits,
        # SHAP-based redundancy calls, etc, in the Tier-2 notebook). test is reported here purely
        # as the single, final, only-touched-once number for whichever feature set was already
        # decided using val -- it must never be looped back into a decision about what to keep.
        val_metrics, _ = scored_metrics(val_df)
        test_metrics, test_predictions = scored_metrics(test_df)
        metrics = test_metrics
        print(f"  val R2:  {val_metrics['r2']:.3f} | val RMSE:  {val_metrics['rmse']:.3f}")
        print(f"  test R2: {test_metrics['r2']:.3f} | test RMSE: {test_metrics['rmse']:.3f}")

        save_dir = output_dir("xgb_environmental", feature_set_name, cohort, split_type="spatial_block")
        save_dir.mkdir(parents=True, exist_ok=True)

        predictions_out = test_df[["identification", "cpmt", "mean_cr_residual"]].copy()
        predictions_out["predicted"] = test_predictions
        predictions_out["residual"] = predictions_out["mean_cr_residual"] - predictions_out["predicted"]
        predictions_out.to_csv(save_dir / "predictions.csv", index=False)

        # metrics.json stays the TEST metrics under the same flat keys as before (nothing reading
        # this file needs to change) -- val metrics are saved alongside in their own file so any
        # notebook/script that wants to make a feature-selection decision has an honest, separate
        # place to look, instead of reaching for metrics.json's test numbers by habit.
        with open(save_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        with open(save_dir / "val_metrics.json", "w") as f:
            json.dump(val_metrics, f, indent=2)
        print(f"  Saved metrics + predictions -> {save_dir}")

        if feature_set_name in FEATURE_SETS_NEEDING_SHAP:
            # SHAP is restricted to train+val rows ONLY -- test rows are never passed in here.
            # Computing SHAP over the full plots_df (including test) was flagged as a leakage
            # risk: it let the "held-out" test set be inspected/explained ahead of the one, final,
            # honest evaluation, the same peeking problem the val split above now exists to avoid.
            train_and_val_df = plots_df[plots_df["split"].isin(["train", "val"])]
            shap_df = compute_shap_values(model, train_and_val_df, feature_set_name)
            shap_path = save_dir / "shap_values.parquet"
            shap_df.to_parquet(shap_path, index=False)
            print(f"  Saved SHAP values ({len(shap_df):,} rows, train+val only) -> {shap_path}")

        write_run_log(
            attempt_id=attempt_id,
            model_name=f"xgb_environmental_{feature_set_name}", cohort=cohort, split_type="spatial_block",
            run_phase="fit_and_evaluate", status="success", is_test_run=False, device=DEVICE,
            hyperparameters=hyperparameters, metrics=metrics, error=None,
            output_dir=save_dir, runtime_seconds=timer.elapsed_seconds(), n_rows_fit=len(train_df),
        )
    except Exception as error:
        write_run_log(
            attempt_id=attempt_id,
            model_name=f"xgb_environmental_{feature_set_name}", cohort=cohort, split_type="spatial_block",
            run_phase="fit_and_evaluate", status="failed", is_test_run=False, device=DEVICE,
            hyperparameters=hyperparameters, metrics=None, error=format_error(error),
            output_dir=None, runtime_seconds=timer.elapsed_seconds(), n_rows_fit=None,
        )
        print(f"  WARNING: {cohort}/{feature_set_name} failed: {error}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=COHORTS, default=None, help="Omit to run both cohorts.")
    args = parser.parse_args()

    cohorts_to_run = [args.cohort] if args.cohort else COHORTS

    for cohort in cohorts_to_run:
        print(f"\n=== Cohort: {cohort} ===")
        plots_df = load_plots_for_cohort(cohort)
        print(f"Loaded {len(plots_df):,} plots")

        for feature_set_name in FEATURE_SETS:
            run_one_feature_set(cohort, feature_set_name, plots_df)


if __name__ == "__main__":
    main()

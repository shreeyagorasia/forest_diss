# Run as: python -m models.baselines.evaluate_baselines
# Evaluates the already-fitted Chapman-Richards, average-by-age, linear
# regression, and random forest baselines on the held-out test split. Does
# not refit anything and does not touch the training split.

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.average_by_age.average_by_age import predict as predict_average_by_age
from models.chapman_richards.chapman_richards import chapman_richards
from models.common.data import filter_data, load_cohort_data, load_model_table
from models.common.metrics import compute_metrics
from models.linear_baseline.linear_baseline import predict as predict_linear_baseline
from models.rf_baseline.rf_baseline import load_model as load_rf_model
from models.rf_baseline.rf_baseline import predict as predict_rf_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COHORTS = ["4survey", "6survey"]
MODEL_NAMES = ["chapman_richards", "average_by_age", "linear_baseline", "rf_baseline"]

# Thresholds used to flag results worth a closer look at the end.
MIN_ROWS_TO_TRUST_A_BAND = 30
LARGE_BIAS_METRES = 2.0


def load_test_rows(cohort, table_name):
    # Rebuild the same filtered rows used for fitting (filtering is
    # deterministic, so this reproduces the same rows every time), then
    # attach the split labels that were already saved by run_baselines.py —
    # this does NOT create a new split, and it is the same merge-based
    # approach used there so every model agrees on which rows are test rows.
    #
    # The saved split assignment is a three-way train/val/test split, but
    # none of these four baselines have anything to tune, so they are only
    # ever evaluated on test — val exists in the file for schema consistency
    # with later models and is deliberately never read here.
    #
    # cr_age.csv.gz itself has no yldc column, so Chapman-Richards and
    # average-by-age are evaluated against the same trimmed
    # (identification, LiDAR_year, blk, Age, yldc, Top_Height99) table
    # load_cohort_data() uses for fitting, not a fresh read of cr_age.csv.gz.
    if table_name == "cr_age":
        table = load_cohort_data(cohort)
    else:
        table = load_model_table(cohort, table_name)
    filtered_table = filter_data(table)

    split_path = PROJECT_ROOT / "outputs" / "splits" / cohort / "split_assignment.csv"
    split_assignment = pd.read_csv(split_path)

    merged_df = filtered_table.merge(split_assignment, on=["identification", "LiDAR_year"], how="inner")
    assert len(merged_df) == len(filtered_table), (
        f"Row count changed after merging {table_name} with the saved split assignment — "
        "the saved split may not match the current filtered data."
    )

    return merged_df[merged_df["split"] == "test"].copy()


def evaluate_chapman_richards(cohort):
    test_df = load_test_rows(cohort, "cr_age")

    params_path = PROJECT_ROOT / "outputs" / "chapman_richards" / cohort / "params.json"
    with open(params_path) as f:
        params = json.load(f)

    predicted_heights = chapman_richards(test_df["Age"].values, params["y_max"], params["k"], params["p"])
    return build_results("chapman_richards", cohort, test_df, predicted_heights)


def evaluate_average_by_age(cohort):
    test_df = load_test_rows(cohort, "cr_age")  # same age-only table CR uses

    lookup_path = PROJECT_ROOT / "outputs" / "average_by_age" / cohort / "lookup.json"
    with open(lookup_path) as f:
        lookup_data = json.load(f)

    predicted_heights = predict_average_by_age(
        test_df["Age"].values, lookup_data["lookup_table"], lookup_data["fallback_mean_height"]
    )
    return build_results("average_by_age", cohort, test_df, predicted_heights)


def evaluate_linear_baseline(cohort):
    test_df = load_test_rows(cohort, "linear_baseline")

    params_path = PROJECT_ROOT / "outputs" / "linear_baseline" / cohort / "params.json"
    with open(params_path) as f:
        params = json.load(f)

    predicted_heights = predict_linear_baseline(test_df, params)
    return build_results("linear_baseline", cohort, test_df, predicted_heights)


def evaluate_rf_baseline(cohort):
    test_df = load_test_rows(cohort, "rf_baseline")

    model_dir = PROJECT_ROOT / "outputs" / "rf_baseline" / cohort
    model = load_rf_model(model_dir)

    predicted_heights = predict_rf_baseline(test_df, model)
    return build_results("rf_baseline", cohort, test_df, predicted_heights)


def build_results(model_name, cohort, test_df, predicted_heights):
    observed_heights = test_df["Top_Height99"].values
    predicted_heights = np.asarray(predicted_heights, dtype=float)
    residuals = observed_heights - predicted_heights

    predictions_df = pd.DataFrame({
        "identification": test_df["identification"].values,
        "blk": test_df["blk"].values,
        "Age": test_df["Age"].values,
        "observed_top_height": observed_heights,
        "predicted_top_height": predicted_heights,
        "residual": residuals,
        "split": "test",
    })

    metrics = compute_metrics(observed_heights, predicted_heights, age=test_df["Age"].values)

    output_dir = PROJECT_ROOT / "outputs" / model_name / cohort
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_dir / "predictions.csv", index=False)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  [{model_name}] saved -> {output_dir / 'predictions.csv'}")
    print(f"  [{model_name}] saved -> {output_dir / 'metrics.json'}")

    return metrics


def flag_issues(cohort, model_name, metrics):
    for band in metrics.get("age_bands", []):
        if band["n_rows"] < MIN_ROWS_TO_TRUST_A_BAND:
            print(
                f"  WARNING [{cohort}/{model_name}] age band {band['band']} has only "
                f"{band['n_rows']} rows — treat its metrics with caution."
            )
        if abs(band["bias"]) > LARGE_BIAS_METRES:
            direction = "under-predicting" if band["bias"] > 0 else "over-predicting"
            print(
                f"  WARNING [{cohort}/{model_name}] age band {band['band']} has a large "
                f"one-signed bias ({band['bias']:.2f} m, {direction})."
            )


def main():
    all_results = {}
    evaluators = {
        "chapman_richards": evaluate_chapman_richards,
        "average_by_age": evaluate_average_by_age,
        "linear_baseline": evaluate_linear_baseline,
        "rf_baseline": evaluate_rf_baseline,
    }

    for cohort in COHORTS:
        print(f"===== {cohort} =====")

        cohort_results = {}
        for model_name in MODEL_NAMES:
            metrics = evaluators[model_name](cohort)
            flag_issues(cohort, model_name, metrics)
            cohort_results[model_name] = metrics

        all_results[cohort] = cohort_results
        print()

    print_summary_table(all_results)
    print_age_band_table(all_results)


def print_summary_table(all_results):
    # Printed with more decimals than usual so small differences between
    # models/cohorts are not hidden by rounding — metrics.json always keeps
    # full precision regardless of how this is displayed.
    print("===== Summary: global metrics on the test split =====")
    header = f"{'cohort':<10}{'model':<20}{'MAE':>12}{'RMSE':>12}{'MSE':>14}{'R2':>12}{'MRE':>12}{'Acc%':>12}{'Bias':>12}"
    print(header)
    for cohort in COHORTS:
        for model_name in MODEL_NAMES:
            m = all_results[cohort][model_name]
            print(
                f"{cohort:<10}{model_name:<20}"
                f"{m['mae']:>12.6f}{m['rmse']:>12.6f}{m['mse']:>14.6f}"
                f"{m['r2']:>12.6f}{m['mre']:>12.6f}{m['accuracy_pct']:>12.6f}{m['bias']:>12.6f}"
            )
    print()


def print_age_band_table(all_results):
    print("===== Age-banded breakdown, all models =====")
    for cohort in COHORTS:
        print(f"{cohort}:")
        for model_name in MODEL_NAMES:
            print(f"  {model_name}:")
            header = f"    {'band':<8}{'n_rows':>10}{'MAE':>12}{'RMSE':>12}{'Bias':>12}{'MRE':>12}"
            print(header)
            for band in all_results[cohort][model_name].get("age_bands", []):
                print(
                    f"    {band['band']:<8}{band['n_rows']:>10,}"
                    f"{band['mae']:>12.6f}{band['rmse']:>12.6f}{band['bias']:>12.6f}{band['mre']:>12.6f}"
                )
    print()


if __name__ == "__main__":
    main()

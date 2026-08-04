# Run as: python -m models.baselines.evaluate_baselines
#     or: python -m models.baselines.evaluate_baselines --split-type spatial_block
#     or: python -m models.baselines.evaluate_baselines --split-type temporal
#
# Evaluates the already-fitted Chapman-Richards, average-by-age, linear
# regression, and random forest baselines on the held-out test split. Does
# not refit anything and does not touch the training split.
#
# --split-type must match whichever split_type run_baselines.py was run
# with -- it reads the params/model files and split_assignment.csv that
# script saved for that split type, from the matching outputs/ subtree
# (see output_dir() in run_baselines.py).

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.average_by_age.average_by_age import predict as predict_average_by_age
from models.baselines.run_baselines import MATURITY_AGE_MIN_DEFAULT, SEED, output_dir
from models.chapman_richards.chapman_richards import chapman_richards
from models.common.data import filter_data, load_cohort_data, load_model_table
from models.common.metrics import compute_metrics
from models.common.run_logging import RunTimer, format_error, write_run_log, write_started_marker
from models.common.splits import DEFAULT_K_FOLDS
from models.linear_baseline.linear_baseline import predict as predict_linear_baseline
from models.rf_baseline.rf_baseline import load_model as load_rf_model
from models.rf_baseline.rf_baseline import predict as predict_rf_baseline

# None of these four baselines use a GPU. Recorded in every log entry
# anyway for schema consistency with the DNN/PINN logs.
DEVICE = "cpu"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COHORTS = ["4survey", "6survey"]
MODEL_NAMES = ["chapman_richards", "average_by_age", "linear_baseline", "rf_baseline"]

# Thresholds used to flag results worth a closer look at the end. Both are round-number
# judgment calls (2026-07-30 note), not derived from a specific statistical rule or citation --
# 30 rows matches the conventional rough rule-of-thumb minimum sample size for a mean to start
# behaving normally (central limit theorem heuristic, not calculated for this data
# specifically); 2.0m is roughly the same order of magnitude as this project's own RMSE numbers
# (see baseline_results.ipynb), so a bias at that scale is "as large as the model's typical
# error" rather than a precise threshold. Worth treating as adjustable defaults, not fixed
# constants, if either ever needs defending.
MIN_ROWS_TO_TRUST_A_BAND = 30
LARGE_BIAS_METRES = 2.0


def load_test_rows(cohort, table_name, split_type, maturity_age_min=MATURITY_AGE_MIN_DEFAULT, name_suffix=""):
    # Rebuild the same filtered rows used for fitting (filtering is
    # deterministic, so this reproduces the same rows every time), then
    # attach the split labels that were already saved by run_baselines.py —
    # this does NOT create a new split, and it is the same merge-based
    # approach used there so every model agrees on which rows are test rows.
    #
    # The saved split assignment is a three-way train/val/test split (plus
    # "buffer" for spatial_block, which is neither train nor test), but none
    # of these four baselines have anything to tune, so they are only ever
    # evaluated on test — val exists in the file for schema consistency with
    # later models only and is deliberately never read here.
    #
    # Chapman-Richards and average-by-age are evaluated against the same trimmed
    # (identification, LiDAR_year, blk, cpmt, Age, yldc, elev_percentile_95th) table
    # load_cohort_data() uses for fitting. Every model reads the same consolidated
    # model_table.parquet now (2026-07-28) -- table_name only picks between that trimmed view
    # and the full table, it no longer selects a different source file.
    #
    # maturity_age_min/name_suffix MUST match whatever run_baselines.py was actually run with,
    # or the row-count assertion below fails -- main() builds both the same way run_baselines.py
    # does, from the same --maturity-age-min value.
    if table_name == "cr_age":
        table = load_cohort_data(cohort)
    else:
        table = load_model_table(cohort)
    filtered_table = filter_data(table, maturity_age_min=maturity_age_min)

    split_path = output_dir(f"splits{name_suffix}", cohort, "split_assignment.csv", split_type=split_type)
    split_assignment = pd.read_csv(split_path)

    merged_df = filtered_table.merge(split_assignment, on=["identification", "LiDAR_year"], how="inner")
    assert len(merged_df) == len(filtered_table), (
        f"Row count changed after merging {table_name} with the saved split assignment — "
        "the saved split may not match the current filtered data."
    )

    return merged_df[merged_df["split"] == "test"].copy()


def evaluate_chapman_richards(cohort, split_type, maturity_age_min=MATURITY_AGE_MIN_DEFAULT, name_suffix=""):
    test_df = load_test_rows(cohort, "cr_age", split_type, maturity_age_min=maturity_age_min, name_suffix=name_suffix)

    params_path = output_dir(f"chapman_richards{name_suffix}", cohort, "params.json", split_type=split_type)
    with open(params_path) as f:
        params = json.load(f)

    predicted_heights = chapman_richards(test_df["Age"].values, params["y_max"], params["k"], params["p"])
    return build_results(f"chapman_richards{name_suffix}", cohort, test_df, predicted_heights, split_type)


def evaluate_average_by_age(cohort, split_type, maturity_age_min=MATURITY_AGE_MIN_DEFAULT, name_suffix=""):
    test_df = load_test_rows(cohort, "cr_age", split_type, maturity_age_min=maturity_age_min, name_suffix=name_suffix)  # same age-only table CR uses

    lookup_path = output_dir(f"average_by_age{name_suffix}", cohort, "lookup.json", split_type=split_type)
    with open(lookup_path) as f:
        lookup_data = json.load(f)

    predicted_heights = predict_average_by_age(
        test_df["Age"].values, lookup_data["lookup_table"], lookup_data["fallback_mean_height"]
    )
    return build_results(f"average_by_age{name_suffix}", cohort, test_df, predicted_heights, split_type)


def evaluate_linear_baseline(cohort, split_type, maturity_age_min=MATURITY_AGE_MIN_DEFAULT, name_suffix=""):
    test_df = load_test_rows(cohort, "linear_baseline", split_type, maturity_age_min=maturity_age_min, name_suffix=name_suffix)

    params_path = output_dir(f"linear_baseline{name_suffix}", cohort, "params.json", split_type=split_type)
    with open(params_path) as f:
        params = json.load(f)

    predicted_heights = predict_linear_baseline(test_df, params)
    return build_results(f"linear_baseline{name_suffix}", cohort, test_df, predicted_heights, split_type)


def evaluate_rf_baseline(cohort, split_type, maturity_age_min=MATURITY_AGE_MIN_DEFAULT, name_suffix=""):
    test_df = load_test_rows(cohort, "rf_baseline", split_type, maturity_age_min=maturity_age_min, name_suffix=name_suffix)

    model_dir = output_dir(f"rf_baseline{name_suffix}", cohort, split_type=split_type)
    model = load_rf_model(model_dir)
    with open(model_dir / "model_metadata.json") as f:
        metadata = json.load(f)

    predicted_heights = predict_rf_baseline(test_df, model, metadata["encoded_column_names"])
    return build_results(f"rf_baseline{name_suffix}", cohort, test_df, predicted_heights, split_type)


def build_results(model_name, cohort, test_df, predicted_heights, split_type):
    observed_heights = test_df["elev_percentile_95th"].values
    predicted_heights = np.asarray(predicted_heights, dtype=float)
    residuals = observed_heights - predicted_heights

    predictions_df = pd.DataFrame({
        "identification": test_df["identification"].values,
        "blk": test_df["blk"].values,
        "cpmt": test_df["cpmt"].values,
        "LiDAR_year": test_df["LiDAR_year"].values,
        "Age": test_df["Age"].values,
        "observed_top_height": observed_heights,
        "predicted_top_height": predicted_heights,
        "residual": residuals,
        "split": "test",
    })

    metrics = compute_metrics(observed_heights, predicted_heights, age=test_df["Age"].values)

    results_dir = output_dir(model_name, cohort, split_type=split_type)
    results_dir.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(results_dir / "predictions.csv", index=False)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  [{model_name}] saved -> {results_dir / 'predictions.csv'}")
    print(f"  [{model_name}] saved -> {results_dir / 'metrics.json'}")

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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-type",
        choices=["plot_level", "spatial_block", "spatial_block_kfold", "temporal", "temporal_narrow_gap"],
        default="plot_level",
        help="Must match whichever split_type run_baselines.py was run with.",
    )
    parser.add_argument(
        "--maturity-age-min", type=int, default=MATURITY_AGE_MIN_DEFAULT,
        help=f"Must match whichever --maturity-age-min run_baselines.py was run with (default "
             f"{MATURITY_AGE_MIN_DEFAULT}) -- reads the matching '_agemin<N>'-suffixed outputs.",
    )
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_K_FOLDS)
    parser.add_argument("--fold-index", type=int, default=0)
    args = parser.parse_args()
    split_type = args.split_type
    maturity_age_min = args.maturity_age_min
    suffix_parts = []
    if args.split_seed != SEED:
        suffix_parts.append(f"_splitseed{args.split_seed}")
    if maturity_age_min != MATURITY_AGE_MIN_DEFAULT:
        suffix_parts.append(f"_agemin{maturity_age_min}")
    if split_type == "spatial_block_kfold":
        if not (0 <= args.fold_index < args.n_folds):
            raise ValueError(f"--fold-index must be in [0, {args.n_folds}), got {args.fold_index}")
        suffix_parts.append(f"_fold{args.fold_index}")
    name_suffix = "".join(suffix_parts)

    all_results = {}
    evaluators = {
        "chapman_richards": evaluate_chapman_richards,
        "average_by_age": evaluate_average_by_age,
        "linear_baseline": evaluate_linear_baseline,
        "rf_baseline": evaluate_rf_baseline,
    }

    for cohort in COHORTS:
        print(f"===== {cohort} ({split_type}, maturity_age_min={maturity_age_min}) =====")

        cohort_results = {}
        for model_name in MODEL_NAMES:
            suffixed_model_name = f"{model_name}{name_suffix}"
            timer = RunTimer().start()
            attempt_id = write_started_marker(
                model_name=suffixed_model_name, cohort=cohort, split_type=split_type, run_phase="evaluate",
                is_test_run=False, device=DEVICE, hyperparameters={"maturity_age_min": maturity_age_min},
            )
            try:
                metrics = evaluators[model_name](cohort, split_type, maturity_age_min=maturity_age_min, name_suffix=name_suffix)
                flag_issues(cohort, model_name, metrics)
                cohort_results[model_name] = metrics
                write_run_log(
                    attempt_id=attempt_id,
                    model_name=suffixed_model_name, cohort=cohort, split_type=split_type, run_phase="evaluate",
                    status="success", is_test_run=False, device=DEVICE,
                    hyperparameters={"maturity_age_min": maturity_age_min}, metrics=metrics, error=None,
                    output_dir=output_dir(suffixed_model_name, cohort, split_type=split_type),
                    runtime_seconds=timer.elapsed_seconds(),
                )
            except Exception as error:
                write_run_log(
                    attempt_id=attempt_id,
                    model_name=suffixed_model_name, cohort=cohort, split_type=split_type, run_phase="evaluate",
                    status="failed", is_test_run=False, device=DEVICE,
                    hyperparameters={"maturity_age_min": maturity_age_min}, metrics=None, error=format_error(error),
                    output_dir=None, runtime_seconds=timer.elapsed_seconds(),
                )
                print(f"  WARNING: evaluation failed for {model_name}/{cohort}/{split_type}: {error}")
                cohort_results[model_name] = None

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
            if m is None:
                print(f"{cohort:<10}{model_name:<20}  evaluation failed, see outputs/run_logs/")
                continue
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
            if all_results[cohort][model_name] is None:
                print("    evaluation failed, see outputs/run_logs/")
                continue
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

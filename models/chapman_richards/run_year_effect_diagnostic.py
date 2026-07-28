# Run as: python -m models.chapman_richards.run_year_effect_diagnostic
#
# DIAGNOSTIC ONLY -- see year_effect_diagnostic.py for why this must never be
# wired into run_baselines.py or evaluate_baselines.py. Fits the plain
# age-only Chapman-Richards baseline and the year-effect version on the
# exact same training rows, so the two are directly comparable, then prints
# how many metres each survey year's y_max sits above or below the
# reference year, and how much of the sum-of-squared-error the year effect
# accounts for.

import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.chapman_richards.chapman_richards import chapman_richards
from models.chapman_richards.year_effect_diagnostic import fit_year_effect, save_year_effect_params
from models.common.data import filter_data, load_cohort_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COHORTS = ["4survey", "6survey"]


def load_train_rows(cohort):
    df = load_cohort_data(cohort)
    filtered_df = filter_data(df)

    split_path = PROJECT_ROOT / "outputs" / "splits" / cohort / "split_assignment.csv"
    split_assignment = pd.read_csv(split_path)

    merged = filtered_df.merge(split_assignment, on=["identification", "LiDAR_year"], how="inner")
    return merged[merged["split"] == "train"]


def plain_cr_sum_squared_error(train_df, cohort):
    # Recomputes the plain age-only Chapman-Richards SSE on these exact
    # training rows, using the params already fitted by run_baselines.py, so
    # it is directly comparable to the year-effect model's SSE below.
    params_path = PROJECT_ROOT / "outputs" / "chapman_richards" / cohort / "params.json"
    with open(params_path) as f:
        params = json.load(f)

    predicted = chapman_richards(train_df["Age"].values, params["y_max"], params["k"], params["p"])
    return float(np.sum((train_df["elev_percentile_95th"].values - predicted) ** 2))


def main():
    for cohort in COHORTS:
        print(f"===== {cohort} =====")
        train_df = load_train_rows(cohort)

        plain_sse = plain_cr_sum_squared_error(train_df, cohort)
        result = fit_year_effect(train_df)
        year_effect_sse = result["sum_squared_error"]

        variance_explained_by_year = (plain_sse - year_effect_sse) / plain_sse * 100

        print(f"  Reference year: {result['reference_year']}")
        for year, delta in sorted(result["year_deltas"].items()):
            print(f"  y_max delta for {year}: {delta:+.4f} m")
        print(f"  Plain CR sum-of-squared-error:       {plain_sse:,.2f}")
        print(f"  Year-effect CR sum-of-squared-error: {year_effect_sse:,.2f}")
        print(f"  Share of squared error the year effect accounts for: {variance_explained_by_year:.2f}%")

        # Saved alongside the fitted params so the notebook can display these
        # comparison numbers without needing to reload the raw training rows.
        result["plain_cr_sum_squared_error"] = plain_sse
        result["variance_explained_by_year_pct"] = variance_explained_by_year

        output_path = PROJECT_ROOT / "outputs" / "chapman_richards_year_diagnostic" / cohort / "params.json"
        save_year_effect_params(result, cohort, len(train_df), output_path)
        print(f"  Saved -> {output_path}")
        print()


if __name__ == "__main__":
    main()

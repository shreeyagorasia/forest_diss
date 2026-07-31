# DIAGNOSTIC ONLY -- this file is never imported by run_baselines.py or
# evaluate_baselines.py, and its output never appears in the main
# model-comparison tables. It exists to put a number in metres on the
# survey-year confound visible in the growth-curve/bias-by-year plots (see
# results_notebooks/baseline_results.ipynb, section 7): the same age band is
# measurably taller in later surveys, and the plain age-only Chapman-Richards
# fit (models/chapman_richards/chapman_richards.py) cannot bend to fit every
# era at once.
#
# A year delta has no predictive value outside the specific calendar years it
# was fitted on -- a plot surveyed in some future year has no fitted delta to
# use -- so this must stay a diagnostic, not a competing baseline.

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def make_model(other_years):
    # Builds a Chapman-Richards function where y_max gets an extra additive
    # delta depending on which survey year a row came from. other_years is
    # every year except the reference year (the earliest year in the
    # training data), which keeps delta = 0 by construction.
    def model(inputs, y_max, k, p, *deltas):
        age, year = inputs
        year_delta = np.zeros_like(age, dtype=float)
        for delta, this_year in zip(deltas, other_years):
            year_delta = year_delta + delta * (year == this_year)
        return (y_max + year_delta) * (1 - np.exp(-k * age)) ** p

    return model


def fit_year_effect(train_df, age_col="Age", year_col="LiDAR_year", height_col="elev_percentile_95th"):
    sorted_years = sorted(train_df[year_col].unique())
    reference_year = sorted_years[0]
    other_years = sorted_years[1:]

    age_values = train_df[age_col].values.astype(float)
    year_values = train_df[year_col].values
    height_values = train_df[height_col].values
    max_observed_height = height_values.max()

    model = make_model(other_years)

    # Same y_max/k/p bounds as the plain Chapman-Richards fit, plus one
    # delta per non-reference year. A delta can be positive or negative (a
    # year can sit above or below the reference year), so its bounds are
    # symmetric and generous rather than one-sided.
    #
    # y_max's lower bound is max_observed_height * 1.001, not exactly max_observed_height
    # (2026-07-30 fix) -- this file reintroduced the exact degenerate bound already found and
    # fixed in chapman_richards.py's own fit() on 28-29 July 2026: a real asymptote is
    # approached but never reached, even by the tallest observed tree, so a bound of exactly
    # max_observed_height let curve_fit land precisely on that boundary instead of finding a
    # genuine asymptote. See chapman_richards.py's own comment for the fuller explanation.
    n_deltas = len(other_years)
    lower_bounds = [max_observed_height * 1.001, 0.0001, 0.01] + [-max_observed_height] * n_deltas
    upper_bounds = [max_observed_height * 5, 2.0, 15.0] + [max_observed_height] * n_deltas

    # Same multi-start idea as the plain fit: several starting guesses for
    # (y_max, k, p), always starting every delta at 0 (i.e. "no year effect"
    # until the data pulls it away from that).
    #
    # First guess is max_observed_height * 1.01, not * 1.0 (2026-07-30 fix, same reasoning as
    # the lower_bounds fix above) -- with the bound now strictly above max_observed_height, a
    # starting guess of exactly max_observed_height would sit BELOW the lower bound, and
    # curve_fit rejects an out-of-bounds p0 outright rather than just converging poorly. Matches
    # chapman_richards.py's own first starting guess.
    starting_guesses = [
        [max_observed_height * 1.01, 0.01, 1.0] + [0.0] * n_deltas,
        [max_observed_height * 1.2, 0.02, 1.0] + [0.0] * n_deltas,
        [max_observed_height * 1.5, 0.03, 1.5] + [0.0] * n_deltas,
        [max_observed_height * 2.0, 0.05, 2.0] + [0.0] * n_deltas,
        [max_observed_height * 1.2, 0.1, 3.0] + [0.0] * n_deltas,
    ]

    best_params = None
    best_sum_squared_error = None

    for starting_guess in starting_guesses:
        try:
            fitted_values, _ = curve_fit(
                model,
                (age_values, year_values),
                height_values,
                p0=starting_guess,
                bounds=(lower_bounds, upper_bounds),
                maxfev=40000,
            )
        except RuntimeError:
            # This starting guess did not converge, so try the next one.
            continue

        predicted_heights = model((age_values, year_values), *fitted_values)
        sum_squared_error = float(np.sum((height_values - predicted_heights) ** 2))

        if best_sum_squared_error is None or sum_squared_error < best_sum_squared_error:
            best_sum_squared_error = sum_squared_error
            best_params = fitted_values

    if best_params is None:
        raise RuntimeError("Chapman-Richards year-effect fit did not converge for any starting guess.")

    y_max, k, p, *deltas = best_params
    year_deltas = {int(reference_year): 0.0}
    for delta, year in zip(deltas, other_years):
        year_deltas[int(year)] = float(delta)

    return {
        "y_max": float(y_max),
        "k": float(k),
        "p": float(p),
        "reference_year": int(reference_year),
        "year_deltas": year_deltas,
        "sum_squared_error": float(best_sum_squared_error),
    }


def save_year_effect_params(result, cohort, n_rows_fit, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "y_max": result["y_max"],
        "k": result["k"],
        "p": result["p"],
        "reference_year": result["reference_year"],
        "year_deltas": result["year_deltas"],
        "sum_squared_error": result["sum_squared_error"],
        "plain_cr_sum_squared_error": result.get("plain_cr_sum_squared_error"),
        "variance_explained_by_year_pct": result.get("variance_explained_by_year_pct"),
        "cohort": cohort,
        "n_rows_fit": n_rows_fit,
        "fit_date": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)

    return payload

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def chapman_richards(age, y_max, k, p):
    # The Chapman-Richards growth curve: height rises from 0 towards y_max as
    # age increases. k controls how fast the growth happens, and p controls
    # the shape of the curve near the start.
    return y_max * (1 - np.exp(-k * age)) ** p


def fit(train_df, age_col="Age", height_col="elev_percentile_95th"):
    # Fit y_max, k and p so the curve matches the training data as closely
    # as possible.
    #
    # LIMITATION (2026-07-30, not fixed here -- see below): both cohorts are strict balanced
    # panels (every plot has the same number of survey-year rows), so every row below is
    # pooled and fit as if independent, ignoring that several rows come from the same plot
    # across different years. Because the panel is balanced, this does NOT bias the point
    # estimate the way it would with an unbalanced panel (every plot still gets equal total
    # weight) -- but it does mean this is a single POPULATION-AVERAGE curve, not a
    # plot-specific one, and it doesn't separate how much residual variance sits at the plot
    # level vs. the year level (relevant context for the environmental-attribution stage
    # downstream). The established fix is a nonlinear mixed-effects model with a plot-level
    # random effect on y_max -- already planned in this project's own roadmap for the
    # environmental-attribution stage (fitting y_max ~ terrain/wind with plot random
    # effects), not implemented here. y_max/k/p below remain frozen point estimates used as
    # the PINN's physics anchor and the residual baseline, same as before.
    age_values = train_df[age_col].values
    height_values = train_df[height_col].values
    max_observed_height = height_values.max()

    # y_max is the height the curve approaches as age goes to infinity, so it must be strictly
    # bigger than the tallest tree we actually measured -- a real asymptote is approached but
    # never reached, even by the tallest observed tree. The lower bound used to be exactly
    # max_observed_height (not above it), which let curve_fit land precisely on that boundary
    # instead of finding a genuine asymptote -- confirmed happening for both the old and new
    # height target (2026-07-28, see progress_notes.md), so this is a pre-existing fragility in
    # these bounds, not something the target change caused. A 0.1% buffer is enough to stop the
    # optimizer sitting exactly on the boundary, without meaningfully changing the search space.
    # k must be positive (height should not shrink as age increases), and p
    # shapes the early part of the curve. The upper bounds are generous so
    # curve_fit has room to search without wandering into silly values.
    lower_bounds = [max_observed_height * 1.001, 0.0001, 0.01]
    upper_bounds = [max_observed_height * 5, 2.0, 15.0]

    # A single starting guess can lead curve_fit to a poor local solution, so
    # we try several different starting guesses and keep whichever one fits
    # the training data best (smallest sum of squared errors).
    starting_guesses = [
        [max_observed_height * 1.01, 0.01, 1.0],
        [max_observed_height * 1.2, 0.02, 1.0],
        [max_observed_height * 1.5, 0.03, 1.5],
        [max_observed_height * 2.0, 0.05, 2.0],
        [max_observed_height * 1.2, 0.1, 3.0],
    ]

    best_params = None
    best_sum_squared_error = None

    for starting_guess in starting_guesses:
        try:
            fitted_values, _ = curve_fit(
                chapman_richards,
                age_values,
                height_values,
                p0=starting_guess,
                bounds=(lower_bounds, upper_bounds),
                maxfev=20000,
            )
        except RuntimeError:
            # This starting guess did not converge, so try the next one.
            continue

        predicted_heights = chapman_richards(age_values, *fitted_values)
        sum_squared_error = float(np.sum((height_values - predicted_heights) ** 2))

        if best_sum_squared_error is None or sum_squared_error < best_sum_squared_error:
            best_sum_squared_error = sum_squared_error
            best_params = fitted_values

    if best_params is None:
        raise RuntimeError("Chapman-Richards fit did not converge for any starting guess.")

    y_max, k, p = best_params
    return {"y_max": float(y_max), "k": float(k), "p": float(p)}


def save_params(params, cohort, n_rows_fit, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "y_max": params["y_max"],
        "k": params["k"],
        "p": params["p"],
        "cohort": cohort,
        "n_rows_fit": n_rows_fit,
        "fit_date": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result

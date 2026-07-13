import numpy as np

# Default age bands for the age-banded breakdown. If the data does not reach
# into the top band (for example, no plots older than 80), that band is just
# left out of the results rather than shown with zero rows.
AGE_BAND_EDGES = [20, 40, 60, 80, np.inf]
AGE_BAND_LABELS = ["20-40", "40-60", "60-80", "80+"]


def compute_metrics(observed, predicted, age=None):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    metrics = compute_metrics_for_rows(observed, predicted)

    if age is not None:
        age = np.asarray(age, dtype=float)
        metrics["age_bands"] = compute_age_band_metrics(observed, predicted, age)

    return metrics


def compute_metrics_for_rows(observed, predicted):
    # This does the actual metric maths for one group of rows. It is used
    # both for the overall metrics and for each individual age band.
    errors = observed - predicted

    mae = float(np.mean(np.abs(errors)))
    mse = float(np.mean(errors ** 2))
    rmse = float(np.sqrt(mse))

    # Bias: mean(observed - predicted). Positive means the model tends to
    # under-predict height; negative means it tends to over-predict.
    bias = float(np.mean(errors))

    # R-squared: how much better the model is than just guessing the mean
    # observed value every time. 1.0 is a perfect model, 0.0 is no better
    # than guessing the mean.
    sum_squared_error = float(np.sum(errors ** 2))
    sum_squared_total = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = float(1 - sum_squared_error / sum_squared_total)

    # Mean Relative Error: average of |error| divided by the observed height,
    # so it is a fraction of the true height rather than metres. Accuracy is
    # just the flip side of this, shown as a percentage.
    relative_errors = np.abs(errors) / observed
    mre = float(np.mean(relative_errors))
    accuracy_pct = float((1 - mre) * 100)

    return {
        "n": int(len(observed)),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mre": mre,
        "accuracy_pct": accuracy_pct,
        "bias": bias,
    }


def compute_age_band_metrics(observed, predicted, age):
    # MRE divides by the observed height, so it naturally looks worse on
    # short/young plots even when the model fits just as well in metres.
    # Breaking metrics out by age band shows whether error is spread evenly
    # or concentrated in young/old plots.
    age_band_results = []

    for band_start, band_end, band_label in zip(AGE_BAND_EDGES[:-1], AGE_BAND_EDGES[1:], AGE_BAND_LABELS):
        in_this_band = (age >= band_start) & (age < band_end)
        n_rows_in_band = int(np.sum(in_this_band))

        if n_rows_in_band == 0:
            continue

        band_metrics = compute_metrics_for_rows(observed[in_this_band], predicted[in_this_band])
        age_band_results.append({
            "band": band_label,
            "n_rows": n_rows_in_band,
            "mae": band_metrics["mae"],
            "rmse": band_metrics["rmse"],
            "bias": band_metrics["bias"],
            "mre": band_metrics["mre"],
        })

    return age_band_results

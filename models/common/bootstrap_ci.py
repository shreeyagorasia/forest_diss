"""A small, reusable cluster-bootstrap confidence-interval helper.

Why a CLUSTER bootstrap, not a plain row-level one: plots in the same compartment are NOT
independent (spatial_block_split/spatial_kfold_split hold out whole compartments together, and
neighbouring plots often share similar terrain/wind/management history). Resampling individual
plot rows would pretend there are far more independent data points than there really are.
Resampling whole COMPARTMENTS is the statistically honest unit -- the same pattern already proven
in models/pinn_env_terrain_k/bootstrap_correlation_check.py and
models/growth_curve_attribution/bootstrap_ci_check.py. This file exists so every model's k-fold
summary can report a confidence interval the same way, instead of that pattern being copy-pasted
per model.

Run standalone for a quick manual check: python -m models.common.bootstrap_ci
"""

import numpy as np


def cluster_bootstrap_ci(df, cluster_column, statistic_fn, n_resamples=1000, seed=42, confidence=0.95):
    """Compute a bootstrap confidence interval for whatever `statistic_fn` measures.

    df             -- a pandas DataFrame with one row per observation (e.g. one row per plot-year).
    cluster_column -- the column name to resample WHOLE GROUPS of (e.g. "cpmt").
    statistic_fn   -- a function that takes a DataFrame (a resampled version of `df`) and returns
                       a single number (e.g. R^2, or a correlation).
    n_resamples    -- how many bootstrap resamples to draw. 1000 is the usual default for a 95%
                       CI -- enough that the percentile estimates are stable, not so many that it
                       takes forever.
    seed           -- makes the resampling reproducible.
    confidence     -- 0.95 for a 95% CI (the usual default).

    Returns a dict: {"point_estimate", "ci_low", "ci_high", "n_resamples", "n_clusters"}.
    """
    random_generator = np.random.default_rng(seed)
    unique_clusters = df[cluster_column].unique()
    n_clusters = len(unique_clusters)

    point_estimate = statistic_fn(df)

    bootstrap_statistics = []
    for _ in range(n_resamples):
        # Resample WHOLE clusters with replacement -- some compartments appear more than once in
        # a given resample, some not at all, exactly like a real bootstrap resample of "which
        # compartments exist in this forest", not "which individual plot rows exist".
        resampled_cluster_ids = random_generator.choice(unique_clusters, size=n_clusters, replace=True)
        resampled_rows = df[df[cluster_column].isin(resampled_cluster_ids)]
        bootstrap_statistics.append(statistic_fn(resampled_rows))

    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 - (1 - confidence) / 2) * 100
    ci_low = float(np.percentile(bootstrap_statistics, lower_percentile))
    ci_high = float(np.percentile(bootstrap_statistics, upper_percentile))

    return {
        "point_estimate": float(point_estimate),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_resamples": n_resamples,
        "n_clusters": n_clusters,
    }


def r2_statistic(df, observed_column="observed_top_height", predicted_column="predicted_top_height"):
    """The usual R^2 = 1 - SS_res/SS_tot, as a plain function of a DataFrame -- the default
    `statistic_fn` most callers in this project will want, since every model's predictions.csv
    already uses these two column names."""
    observed = df[observed_column].to_numpy()
    predicted = df[predicted_column].to_numpy()
    residual_sum_of_squares = np.sum((observed - predicted) ** 2)
    total_sum_of_squares = np.sum((observed - observed.mean()) ** 2)
    return 1 - (residual_sum_of_squares / total_sum_of_squares)


if __name__ == "__main__":
    # A tiny synthetic self-check, so this file can be run on its own to confirm it behaves
    # sensibly, without needing any real model output on disk.
    import pandas as pd

    rng = np.random.default_rng(0)
    n_compartments = 30
    rows = []
    for compartment_id in range(n_compartments):
        true_height = rng.uniform(10, 30)
        for _ in range(20):
            observed = true_height + rng.normal(0, 1.0)
            predicted = true_height + rng.normal(0, 1.5)
            rows.append({"cpmt": compartment_id, "observed_top_height": observed, "predicted_top_height": predicted})
    demo_df = pd.DataFrame(rows)

    result = cluster_bootstrap_ci(demo_df, "cpmt", r2_statistic, n_resamples=500)
    print("Demo cluster-bootstrap CI on R^2:")
    print(result)

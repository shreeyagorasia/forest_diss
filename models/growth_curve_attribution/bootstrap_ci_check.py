# Purpose: quantify uncertainty on the plot-level terrain/wind attribution R2 (currently reported
# as a single point estimate, e.g. 0.061/0.103 at split_seed=42 on the cleaned population) --
# directly answering the open question raised by the LLM Council review of this whole avenue
# (documentation/experiment_log.md, 2026-08-04 entry): is the ~0.11-0.13 R2 a real, precise
# result, or a number with wide uncertainty that a small number of held-out compartments makes
# hard to pin down?
#
# Uses a CLUSTER bootstrap, not a plain row-level bootstrap. Val-set rows are NOT independent --
# every row in the same compartment shares the same held-out/train split fate and largely the
# same terrain, so resampling individual PLOT rows would pretend there are far more independent
# data points than there really are (the exact same pseudo-replication issue already flagged and
# fixed for the compartment-ICC check earlier in this project). Resampling whole COMPARTMENTS
# (every plot in a resampled compartment moves together) is the statistically honest unit here,
# since that is the actual unit spatial_block_split() holds out.
#
# The model itself is fit ONCE (the standard split_seed=42 split, matching every other reported
# number) -- this bootstrap is about the uncertainty of the R2 ESTIMATE on a fixed validation set,
# not about retraining variability (that question is already separately covered by the
# multi-split-seed sweep in run_seed_sweep_check.py).

import numpy as np
import pandas as pd

from models.common.splits import SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, SPLIT_SEED, spatial_block_split
from models.elasticnet_environmental.elasticnet_environmental import drop_rows_with_missing_features
from models.elasticnet_environmental.elasticnet_environmental import fit_with_columns as elasticnet_fit
from models.elasticnet_environmental.elasticnet_environmental import predict_with_columns as elasticnet_predict
from models.growth_curve_attribution.scale_comparison_check import TARGET, build_plot_level_table, merge_environmental_features
from models.xgb_environmental.xgb_environmental import fit_with_columns as xgb_fit
from models.xgb_environmental.xgb_environmental import predict_with_columns as xgb_predict


def get_val_predictions_with_cpmt(cohort, split_seed=SPLIT_SEED):
    # Fits both models ONCE on the standard train split, then returns every val row's actual
    # target, both models' predictions, and its compartment -- everything the bootstrap loop
    # below needs, without re-fitting anything inside the loop (that would be the expensive,
    # unnecessary version of this check).
    plot_table = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
    plot_table_with_features, feature_columns = merge_environmental_features(plot_table)

    table = plot_table_with_features.copy()
    coordinates = table[["identification", "x", "y"]]
    table["split"] = spatial_block_split(
        table, block_col=SPATIAL_BLOCK_COL, buffer_distance=SPATIAL_BUFFER_METRES,
        coordinates_df=coordinates, seed=split_seed,
    )

    train = drop_rows_with_missing_features(table[table["split"] == "train"], feature_columns)
    val = drop_rows_with_missing_features(table[table["split"] == "val"], feature_columns).copy()

    elastic = elasticnet_fit(train, feature_columns, target_col=TARGET)
    val["elastic_net_predicted"] = elasticnet_predict(val, elastic, feature_columns)

    tree = xgb_fit(train, feature_columns, val_df=val, target_col=TARGET, n_estimators=500, max_depth=4, learning_rate=0.04)
    val["xgboost_predicted"] = xgb_predict(val, tree, feature_columns)

    return val[["identification", "cpmt", TARGET, "elastic_net_predicted", "xgboost_predicted"]]


def compute_r2(actual, predicted):
    # Same formula as models/common/metrics.py::compute_metrics_for_rows() -- R2 relative to
    # THIS group's own mean, recomputed fresh for every bootstrap resample (standard practice:
    # each resample is its own dataset with its own baseline, not compared against the original
    # sample's mean).
    sum_squared_error = np.sum((actual - predicted) ** 2)
    sum_squared_total = np.sum((actual - actual.mean()) ** 2)
    return 1 - sum_squared_error / sum_squared_total if sum_squared_total > 0 else np.nan


def cluster_bootstrap_r2_ci(val_predictions, predicted_col, n_bootstrap=2000, seed=0):
    # Resamples whole COMPARTMENTS with replacement (not individual plot rows) -- see this
    # module's own header for why that's the statistically honest unit here.
    target_values = val_predictions[TARGET].to_numpy()
    predicted_values = val_predictions[predicted_col].to_numpy()
    cpmt_values = val_predictions["cpmt"].to_numpy()

    compartments = np.unique(cpmt_values)
    n_compartments = len(compartments)
    indices_by_compartment = {compartment: np.where(cpmt_values == compartment)[0] for compartment in compartments}

    rng = np.random.default_rng(seed)
    bootstrap_r2_values = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        resampled_compartments = rng.choice(compartments, size=n_compartments, replace=True)
        resampled_indices = np.concatenate([indices_by_compartment[compartment] for compartment in resampled_compartments])
        bootstrap_r2_values[i] = compute_r2(target_values[resampled_indices], predicted_values[resampled_indices])

    point_estimate_r2 = compute_r2(target_values, predicted_values)

    return {
        "n_val_compartments": n_compartments,
        "n_val_rows": len(val_predictions),
        "point_estimate_r2": point_estimate_r2,
        "bootstrap_mean_r2": float(np.nanmean(bootstrap_r2_values)),
        "bootstrap_std_r2": float(np.nanstd(bootstrap_r2_values)),
        "ci_95_lower": float(np.nanpercentile(bootstrap_r2_values, 2.5)),
        "ci_95_upper": float(np.nanpercentile(bootstrap_r2_values, 97.5)),
        "fraction_of_bootstrap_resamples_below_zero": float(np.nanmean(bootstrap_r2_values < 0)),
    }

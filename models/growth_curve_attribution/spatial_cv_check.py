# Purpose: the fix the bootstrap CI check (bootstrap_ci_check.py) pointed at -- a single
# spatial_block_split only ever evaluates on a small slice of the forest's compartments (39 of
# 232 for 4survey, 7 of 47 for 6survey), which is WHY the confidence interval on the plot-level
# attribution R2 was wide enough to include zero. Rotating which compartments are held out across
# many folds, so every compartment eventually gets evaluated on once, uses the WHOLE population
# instead of one ~17% slice of it -- a direct, cheap precision fix that doesn't touch the model at
# all (matching the LLM Council's advice: the ceiling is a precision/signal problem, not a model-
# capacity problem, so build a better EVALUATION, not a fancier model).
#
# Each fold reuses the exact same fit/predict/buffer logic as the single-split check
# (scale_comparison_check.py) -- same features, same models, same 60m leakage buffer -- just
# repeated K times with a different held-out slice of compartments each time.

import numpy as np
import pandas as pd

from models.common.splits import (
    DEFAULT_K_FOLDS,
    SPATIAL_BLOCK_COL,
    SPATIAL_BUFFER_METRES,
    SPLIT_SEED,
    apply_spatial_buffer,
    assign_spatial_folds,
)
from models.elasticnet_environmental.elasticnet_environmental import drop_rows_with_missing_features
from models.elasticnet_environmental.elasticnet_environmental import fit_with_columns as elasticnet_fit
from models.elasticnet_environmental.elasticnet_environmental import predict_with_columns as elasticnet_predict
from models.growth_curve_attribution.scale_comparison_check import TARGET, build_plot_level_table, merge_environmental_features
from models.xgb_environmental.xgb_environmental import fit_with_columns as xgb_fit
from models.xgb_environmental.xgb_environmental import predict_with_columns as xgb_predict

# assign_spatial_folds() now lives in models/common/splits.py (2026-08-04) -- lifted from here so
# the DNN/PINN k-fold work can share the exact same fold-assignment algorithm instead of a second
# copy that could drift apart from this one. Imported above, not redefined.


def run_spatial_cv(
    table, feature_columns, k=DEFAULT_K_FOLDS, seed=SPLIT_SEED,
    buffer_distance=SPATIAL_BUFFER_METRES, target_col=TARGET,
):
    # Returns one row per (plot, fold-it-was-held-out-in) with both models' out-of-fold
    # predictions -- every plot appears exactly once, always predicted by a model that never saw
    # its own compartment during training. Pooling these across all K folds afterwards gives an
    # R2 computed over the WHOLE population, not one ~17% slice of it.
    fold_assignment, fold_row_counts = assign_spatial_folds(table, k=k, seed=seed)
    table = table.copy()
    table["fold"] = fold_assignment

    coordinates = table[["identification", "x", "y"]]

    out_of_fold_rows = []
    for held_out_fold in range(k):
        val_fold = (held_out_fold + 1) % k
        # apply_spatial_buffer()'s own default only protects against "train" plots sitting near
        # "val"/"test" plots (see models/common/geo.py::find_train_plots_near_holdout) -- "test"
        # is reused here as the held-out label for exactly that reason, not because this is
        # actually a test split in the train/val/test sense elsewhere in this project.
        raw_labels = np.where(
            table["fold"] == held_out_fold,
            "test",
            np.where(table["fold"] == val_fold, "val", "train"),
        )
        buffered_labels = apply_spatial_buffer(table, list(raw_labels), buffer_distance, coordinates)
        table_this_fold = table.copy()
        table_this_fold["split"] = buffered_labels

        train = drop_rows_with_missing_features(table_this_fold[table_this_fold["split"] == "train"], feature_columns)
        val = drop_rows_with_missing_features(table_this_fold[table_this_fold["split"] == "val"], feature_columns)
        held_out = drop_rows_with_missing_features(table_this_fold[table_this_fold["split"] == "test"], feature_columns).copy()

        elastic = elasticnet_fit(train, feature_columns, target_col=target_col)
        held_out["elastic_net_predicted"] = elasticnet_predict(held_out, elastic, feature_columns)

        tree = xgb_fit(train, feature_columns, val_df=val, target_col=target_col, n_estimators=500, max_depth=4, learning_rate=0.04)
        held_out["xgboost_predicted"] = xgb_predict(held_out, tree, feature_columns)

        held_out["fold"] = held_out_fold
        held_out["n_train"] = len(train)
        out_of_fold_rows.append(held_out[["identification", "cpmt", "fold", "n_train", target_col, "elastic_net_predicted", "xgboost_predicted"]])

    out_of_fold_predictions = pd.concat(out_of_fold_rows, ignore_index=True)
    return out_of_fold_predictions, fold_row_counts


def compute_r2(actual, predicted):
    sum_squared_error = np.sum((actual - predicted) ** 2)
    sum_squared_total = np.sum((actual - actual.mean()) ** 2)
    return 1 - sum_squared_error / sum_squared_total if sum_squared_total > 0 else np.nan


def summarize_spatial_cv(out_of_fold_predictions, predicted_col, target_col=TARGET):
    # Pooled R2: every plot's out-of-fold prediction, all folds concatenated, ONE R2 computed
    # over the whole population -- the headline number, since it uses every compartment.
    pooled_r2 = compute_r2(out_of_fold_predictions[target_col].to_numpy(), out_of_fold_predictions[predicted_col].to_numpy())

    # Per-fold R2: how much the estimate would have varied if only ONE fold's worth of
    # compartments had been used as the held-out set (i.e. what the old single-split check was
    # actually doing) -- shown alongside the pooled number so the precision gain is visible
    # directly, not just asserted.
    per_fold = out_of_fold_predictions.groupby("fold").apply(
        lambda rows: compute_r2(rows[target_col].to_numpy(), rows[predicted_col].to_numpy()), include_groups=False
    )

    return {
        "pooled_r2": pooled_r2,
        "n_plots": len(out_of_fold_predictions),
        "n_compartments": out_of_fold_predictions["cpmt"].nunique(),
        "per_fold_r2_mean": float(per_fold.mean()),
        "per_fold_r2_std": float(per_fold.std()),
        "per_fold_r2_values": per_fold.to_dict(),
    }

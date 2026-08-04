# Purpose: does the variable ranking from explain_signal.py hold up across different held-out
# compartment sets, or would a different spatial split reorder it? Cheap version: XGBoost's own
# built-in gain-based feature_importances_ per fold's TRAIN-only fit (already computed as a
# byproduct of fitting, no separate SHAP recomputation needed) -- not full SHAP per fold, which
# would be five times the cost of the single fit already done for no real extra insight into
# RANK stability specifically.

import pandas as pd

from models.common.splits import SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, SPLIT_SEED, apply_spatial_buffer, assign_spatial_folds
from models.elasticnet_environmental.elasticnet_environmental import drop_rows_with_missing_features
from models.growth_curve_attribution.explain_signal import FINAL_FEATURE_COLUMNS, build_full_table
from models.growth_curve_attribution.scale_comparison_check import TARGET
from models.xgb_environmental.xgb_environmental import fit_with_columns as xgb_fit

import numpy as np


def per_fold_gain_importance(table, feature_columns, k=5, seed=SPLIT_SEED):
    fold_assignment, _ = assign_spatial_folds(table, k=k, seed=seed)
    table = table.copy()
    table["fold"] = fold_assignment
    coordinates = table[["identification", "x", "y"]]

    rows = []
    for held_out_fold in range(k):
        raw_labels = np.where(table["fold"] == held_out_fold, "test", "train")
        buffered_labels = apply_spatial_buffer(table, list(raw_labels), SPATIAL_BUFFER_METRES, coordinates)
        train = drop_rows_with_missing_features(
            table[pd.Series(buffered_labels, index=table.index) == "train"], feature_columns
        )

        model = xgb_fit(train, feature_columns, target_col=TARGET, n_estimators=500, max_depth=4, learning_rate=0.04)
        importances = model.get_booster().get_score(importance_type="gain")
        for column in feature_columns:
            rows.append({"fold": held_out_fold, "variable": column, "gain": importances.get(column, 0.0)})

    return pd.DataFrame(rows)


def summarize_rank_stability(per_fold_gain):
    # Rank within each fold (1 = most important), then look at how much a variable's rank moves
    # across the 5 folds -- a variable that's consistently top-5 tells a more trustworthy story
    # than one that's #2 in one fold and #14 in another.
    per_fold_gain = per_fold_gain.copy()
    per_fold_gain["rank"] = per_fold_gain.groupby("fold")["gain"].rank(ascending=False)
    summary = per_fold_gain.groupby("variable")["rank"].agg(["mean", "std", "min", "max"]).sort_values("mean")
    return summary.rename(columns={"mean": "rank_mean", "std": "rank_std", "min": "rank_best", "max": "rank_worst"})

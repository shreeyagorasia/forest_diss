# Purpose: a proper PAIRED fold-by-fold comparison between tuned XGBoost and DNN on RQ1's task
# (Set3, both cohorts, spatial_block_kfold, seed 42) -- both models are evaluated on the exact
# same 5 folds (same split_seed=42), so the right comparison is fold-by-fold (XGBoost_r2_fold_i
# minus DNN_r2_fold_i), not a comparison of each model's own marginal SD in isolation. A marginal-
# SD comparison ignores that some folds are simply harder for every model (shared difficulty), so
# it understates how consistent the direction of the gap actually is.
#
# No new fitting -- reads DNN's already-saved per-fold R2 from its kfold_summary.json, and
# XGBoost's per-fold R2 from a direct refit using the winning config found by
# rq1_xgb_hyperparameter_search.py (n_estimators=500, max_depth=6, learning_rate=0.02 for
# 4survey; n_estimators=300, max_depth=3, learning_rate=0.08 for 6survey).
#
# Run as: python -m models.baselines.rq1_xgb_vs_dnn_paired_folds

import json

import numpy as np
import xgboost as xgb

from models.baselines.rq1_xgb_hyperparameter_search import build_fold_features, fit_and_score
from models.baselines.run_baselines import DEFAULT_K_FOLDS

WINNING_CONFIGS = {
    "4survey": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.02},
    "6survey": {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.08},
}

DNN_KFOLD_SUMMARY_PATHS = {
    "4survey": "outputs/spatial_block_kfold/rq1_dnn_env_terrain_nested_set3_gated_terrain_wind_vif_seed42/4survey/kfold_summary.json",
    "6survey": "outputs/spatial_block_kfold/rq1_dnn_env_terrain_nested_set3_gated_terrain_wind_vif_seed42/6survey/kfold_summary.json",
}


def xgb_per_fold_test_r2(cohort, params):
    r2_values = []
    for fold in range(DEFAULT_K_FOLDS):
        fold_data = build_fold_features(cohort, fold)
        _, test_r2 = fit_and_score(fold_data, params)
        r2_values.append(test_r2)
    return r2_values


def main():
    for cohort in ["4survey", "6survey"]:
        xgb_r2 = xgb_per_fold_test_r2(cohort, WINNING_CONFIGS[cohort])
        with open(DNN_KFOLD_SUMMARY_PATHS[cohort]) as f:
            dnn_r2 = json.load(f)["per_fold_r2_values"]

        diffs = [x - d for x, d in zip(xgb_r2, dnn_r2)]
        wins = sum(1 for v in diffs if v > 0)
        print(f"=== {cohort} ===")
        print(f"  XGBoost per-fold R2: {[round(v, 4) for v in xgb_r2]}")
        print(f"  DNN per-fold R2:     {[round(v, 4) for v in dnn_r2]}")
        print(f"  Per-fold diff (XGB-DNN): {[round(v, 4) for v in diffs]}")
        print(f"  Mean diff: {np.mean(diffs):.4f}  SD: {np.std(diffs, ddof=1):.4f}  "
              f"XGBoost wins {wins}/{DEFAULT_K_FOLDS} folds")
        print()


if __name__ == "__main__":
    main()

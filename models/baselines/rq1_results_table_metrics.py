# Purpose: recompute the RQ1 results table's per-fold R2/RMSE/MAE mean+-SD directly from each
# model's already-saved metrics.json files, for every row in the table (Linear, RF, XGBoost raw
# defaults, DNN, PINN, PINN_k -- Set3, both cohorts, spatial_block_kfold, seed 42). No retraining
# -- these fits already exist on disk from the main RQ1 sweep and the baseline-env rerun.
#
# Why this script exists: the results table's headline says "per-fold mean+-SD, same method every
# row," but no single TEMP_results note had ever assembled every row from ONE script using ONE
# consistent formula -- R2 point estimates were checked cell-by-cell across several different
# notes during an audit and matched, but the RMSE/MAE +- SD values for DNN/PINN/PINN_k specifically
# could not be traced to any existing note (2026-08-16). This script is that missing single source,
# so every number in the table is reproducible from one place. Sample SD (pandas .std(), ddof=1) is
# used throughout, matching the convention already used by every other row/table in this project.
#
# Run as: python -m models.baselines.rq1_results_table_metrics

import json
from pathlib import Path

import pandas as pd

OUTPUTS_ROOT = Path("outputs/spatial_block_kfold")
COHORTS = ["4survey", "6survey"]
N_FOLDS = 5

# Each entry: (display name, run_name template with {fold} placeholder, folder layout)
# Baselines (Linear/RF/XGBoost) fold number is IN the run_name; DNN/PINN/PINN_k fold number is a
# subfolder INSIDE one run_name (all 5 folds share one run_name, since they trained together as a
# single sweep) -- both layouts are handled below.
BASELINE_MODELS = {
    "Linear": "linear_baseline_env_nested_set3_gated_terrain_wind_vif_fold{fold}",
    "RF": "rf_baseline_env_nested_set3_gated_terrain_wind_vif_fold{fold}",
    "XGBoost (raw defaults)": "xgb_baseline_env_nested_set3_gated_terrain_wind_vif_fold{fold}",
}
NEURAL_MODELS = {
    "DNN": "rq1_dnn_env_terrain_nested_set3_gated_terrain_wind_vif_seed42",
    "PINN": "rq1_pinn_env_terrain_nested_set3_gated_terrain_wind_vif_seed42",
    "PINN_k": "rq1_pinn_env_terrain_k_nested_set3_gated_terrain_wind_vif_seed42",
}


def load_metrics(path):
    with open(path) as f:
        data = json.load(f)
    return {"r2": data["r2"], "rmse": data["rmse"], "mae": data["mae"]}


def collect_baseline_fold_metrics(run_name_template, cohort):
    rows = []
    for fold in range(N_FOLDS):
        run_name = run_name_template.format(fold=fold)
        metrics_path = OUTPUTS_ROOT / run_name / cohort / "metrics.json"
        rows.append(load_metrics(metrics_path))
    return pd.DataFrame(rows)


def collect_neural_fold_metrics(run_name, cohort):
    rows = []
    for fold in range(N_FOLDS):
        metrics_path = OUTPUTS_ROOT / run_name / cohort / f"fold_{fold}" / "metrics.json"
        rows.append(load_metrics(metrics_path))
    return pd.DataFrame(rows)


def summarize(fold_df):
    # ddof=1 (pandas default, sample SD) -- matches the convention already used for every other
    # row in this table (Linear/RF/XGBoost were already computed this way in earlier notes).
    mean = fold_df.mean()
    std = fold_df.std()
    return {
        "r2": f"{mean['r2']:.3f}±{std['r2']:.3f}",
        "rmse": f"{mean['rmse']:.3f}±{std['rmse']:.3f}",
        "mae": f"{mean['mae']:.3f}±{std['mae']:.3f}",
    }


def main():
    for cohort in COHORTS:
        print(f"=== {cohort} ===")
        for name, template in BASELINE_MODELS.items():
            fold_df = collect_baseline_fold_metrics(template, cohort)
            summary = summarize(fold_df)
            print(f"  {name}: R2={summary['r2']}  RMSE={summary['rmse']}  MAE={summary['mae']}")
        for name, run_name in NEURAL_MODELS.items():
            fold_df = collect_neural_fold_metrics(run_name, cohort)
            summary = summarize(fold_df)
            print(f"  {name}: R2={summary['r2']}  RMSE={summary['rmse']}  MAE={summary['mae']}")
        print()


if __name__ == "__main__":
    main()

# Purpose: check whether GNNWR's residuals show LESS leftover spatial autocorrelation than
# Elastic Net/XGBoost/the plain DNN's residuals -- a stronger, more specific piece of evidence
# than a bare R2 comparison for whether GNNWR is capturing real local structure, not just fitting
# noise. Directly prompted by a paper comparison (GWDNN for down dead wood volume): its strongest
# evidence for the GWDNN architecture beating plain GWR wasn't the R2 bump alone, it was that
# residual Moran's I dropped to ~0 (no leftover spatial pattern), vs OLS's 0.24.
#
# Uses a semivariogram-informed distance-band, the SAME method as
# models/spatial_attribution/spatial_autocorrelation.py::global_morans_i() (used elsewhere in this
# project for a DNN/PINN model's own residual check and RQ2b's category-ablation check) -- unified
# here rather than the k=8 nearest-neighbour weights this used previously. That k=8 choice was
# deliberate (avoided the wildly uneven neighbour COUNTS a fixed distance band gives on this data,
# 681 to 7,141 neighbours per plot at the Stage 1 CR-residual's 3,956m range), but a
# neighbour-count of 8 is too small to test real REGIONAL clustering -- it mostly answers "is this
# plot like the plot right next to it" (see spatial_autocorrelation.py's own header for the same
# point). Decided to accept the uneven-neighbour-count trade-off in exchange for actually testing
# clustering at the right physical scale, and for having one consistent Moran's I method across
# the whole project instead of two.
#
# Runs entirely from data already on disk (the DNN/GNNWR test-prediction CSVs, and this file's own
# quick local rerun of the existing spatial CV code for Elastic Net/XGBoost) -- no cluster
# resubmission needed for any of this.

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from models.growth_curve_attribution.broad_environmental_check import columns_for_groups
from models.growth_curve_attribution.gnnwr_check import build_scope_table
from models.growth_curve_attribution.scale_comparison_check import TARGET, merge_environmental_features
from models.growth_curve_attribution.spatial_cv_check import run_spatial_cv
from models.spatial_attribution.spatial_autocorrelation import global_morans_i, semivariogram_range

DEFAULT_MAX_DISTANCE_M = 5000  # matches semivariogram_range()'s own default search window

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GNNWR_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "growth_curve_attribution" / "gnnwr"
SIMPLE_DNN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "growth_curve_attribution" / "simple_dnn"
COMPARTMENT_MIXED_DNN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "growth_curve_attribution" / "compartment_mixed_dnn"


def compute_residual_morans_i(x, y, residual, max_distance=DEFAULT_MAX_DISTANCE_M, permutations=999, seed=42):
    # semivariogram_range() finds the real distance scale to test at FIRST -- picking an
    # arbitrary distance (or a fixed neighbour-count) risks testing "is this plot like the plot
    # right next to it" instead of real regional clustering (spatial_autocorrelation.py's own
    # header comment explains this in full).
    #
    # The fitted range is NOT the same for every model/check -- it depends on that specific
    # residual field's own spatial structure, so it is returned alongside Moran's I (as
    # range_m/range_status) rather than only printed. Any report that states a Moran's I value
    # from this function should also state the range it was computed at -- two models' Moran's I
    # aren't on the same yardstick if they were tested at different distances.
    mask = ~(np.isnan(residual) | np.isnan(x) | np.isnan(y))
    x, y, residual = np.asarray(x)[mask], np.asarray(y)[mask], np.asarray(residual)[mask]

    range_estimate, status = semivariogram_range(x, y, residual, max_distance=max_distance)
    if status not in ("resolved", "exceeds_window"):
        print(f"  (semivariogram range not resolved: {status} -- Moran's I skipped)")
        return np.nan, np.nan, mask.sum(), np.nan, status

    print(f"  semivariogram range = {range_estimate:.0f}m (status: {status})")
    morans_i, p_value = global_morans_i(x, y, residual, distance=range_estimate, permutations=permutations, seed=seed)
    return morans_i, p_value, mask.sum(), range_estimate, status


def check_gnnwr_and_dnn_outputs(cohort: str, scope: str) -> pd.DataFrame:
    rows = []

    # ----- GNNWR: every saved CSV already has x/y (joined back via GNNWR.getCoefs()) -----
    for csv_path in sorted(GNNWR_OUTPUT_DIR.glob(f"gnnwr_{scope}_{cohort}*_test_predictions.csv")):
        table = pd.read_csv(csv_path)
        residual = table[TARGET] - table["denormalized_pred_result"]
        morans_i, p_value, n, range_m, range_status = compute_residual_morans_i(table["x"], table["y"], residual)
        rows.append({
            "model": f"GNNWR ({csv_path.stem})", "morans_i": morans_i, "p_value": p_value, "n_plots": n,
            "range_m": range_m, "range_status": range_status,
        })

    # ----- Simple DNN / compartment-mixed DNN: saved CSVs only have identification/cpmt/target/
    # predicted, no x/y -- merge coordinates back in from build_scope_table() (cheap, local,
    # no retraining -- this rebuilds the SAME table the model was trained on, not a new one). -----
    reference_table, _ = build_scope_table(cohort, scope)
    coordinate_lookup = reference_table[["identification", "x", "y"]].drop_duplicates("identification")

    dnn_sources = [
        ("Simple DNN", SIMPLE_DNN_OUTPUT_DIR / f"simple_dnn_{scope}_{cohort}_test_predictions.csv"),
        ("Compartment-mixed DNN (fixed effects)", COMPARTMENT_MIXED_DNN_OUTPUT_DIR / f"compartment_mixed_dnn_{scope}_{cohort}_test_predictions.csv"),
    ]
    for label, csv_path in dnn_sources:
        if not csv_path.exists():
            print(f"  Skipping {label}: {csv_path} not found")
            continue
        table = pd.read_csv(csv_path).merge(coordinate_lookup, on="identification", how="left", validate="many_to_one")
        residual = table[TARGET] - table["predicted"]
        morans_i, p_value, n, range_m, range_status = compute_residual_morans_i(table["x"], table["y"], residual)
        rows.append({
            "model": label, "morans_i": morans_i, "p_value": p_value, "n_plots": n,
            "range_m": range_m, "range_status": range_status,
        })

    return pd.DataFrame(rows)


def check_en_xgboost(cohort: str) -> pd.DataFrame:
    # Elastic Net/XGBoost don't have a saved per-plot prediction CSV with coordinates from
    # earlier spatial CV runs -- rerunning run_spatial_cv() is fast and entirely local (no
    # cluster), so this just calls the SAME already-existing function used for their headline
    # 0.125/0.117 numbers, and keeps the per-plot out-of-fold predictions this time instead of
    # only the pooled summary.
    # build_scope_table() applies spatial_block_split(), which run_spatial_cv() does not need
    # (it does its own k-fold split) -- reuse the SAME cleaned/merged population, not the split.
    # columns_for_groups(["terrain_wind"]) -- NOT xgb_environmental.TERRAIN_AND_WIND_COLUMNS,
    # confirmed directly these differ (the old 16-column list vs the actual 17-column
    # FINAL_FEATURE_COLUMNS GNNWR/DNN use, post wind-swap/TPI-scale checks) -- using the wrong
    # one would silently compare EN/XGBoost on different features than GNNWR/DNN actually saw.
    from models.growth_curve_attribution.scale_comparison_check import build_plot_level_table
    plot_table = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
    merged, feature_columns = merge_environmental_features(plot_table, feature_columns=columns_for_groups(["terrain_wind"]))

    # run_spatial_cv()'s out_of_fold_predictions already includes x/y directly (it keeps them
    # for its own leakage-buffer logic) -- a stale comment here used to claim otherwise and
    # merged coordinate_lookup back in anyway, which duplicated x/y into pandas' auto-suffixed
    # x_x/x_y columns (both sides of the merge had x/y) and broke the plain ["x"] lookup below.
    out_of_fold_predictions, _ = run_spatial_cv(merged, feature_columns)

    rows = []
    for predicted_col, label in [("elastic_net_predicted", "Elastic Net"), ("xgboost_predicted", "XGBoost")]:
        residual = out_of_fold_predictions[TARGET] - out_of_fold_predictions[predicted_col]
        morans_i, p_value, n, range_m, range_status = compute_residual_morans_i(
            out_of_fold_predictions["x"], out_of_fold_predictions["y"], residual,
        )
        rows.append({
            "model": label, "morans_i": morans_i, "p_value": p_value, "n_plots": n,
            "range_m": range_m, "range_status": range_status,
        })
    return pd.DataFrame(rows)


def main():
    print("Elastic Net / XGBoost (terrain_wind, 5-fold pooled out-of-fold residuals):")
    en_xgb_results = check_en_xgboost("4survey")
    print(en_xgb_results.to_string(index=False))

    for scope in ["terrain_wind", "terrain_wind_plus_management"]:
        print(f"\nGNNWR / DNN models ({scope}, single-split test residuals):")
        results = check_gnnwr_and_dnn_outputs("4survey", scope)
        print(results.to_string(index=False))

    print(
        "\nReading this: Moran's I near 0 (and p_value NOT significant) means the residuals look "
        "spatially random -- the model has soaked up the local structure. A positive, significant "
        "Moran's I means nearby plots' errors are still correlated -- real local pattern the model "
        "is missing, the same kind of evidence the GWDNN paper used to argue for its architecture."
    )


if __name__ == "__main__":
    main()

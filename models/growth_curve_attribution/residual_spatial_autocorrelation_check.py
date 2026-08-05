# Purpose: check whether GNNWR's residuals show LESS leftover spatial autocorrelation than
# Elastic Net/XGBoost/the plain DNN's residuals -- a stronger, more specific piece of evidence
# than a bare R2 comparison for whether GNNWR is capturing real local structure, not just fitting
# noise. Directly prompted by a paper comparison (GWDNN for down dead wood volume): its strongest
# evidence for the GWDNN architecture beating plain GWR wasn't the R2 bump alone, it was that
# residual Moran's I dropped to ~0 (no leftover spatial pattern), vs OLS's 0.24.
#
# Uses k-nearest-neighbour weights (k=8), NOT a fixed distance band -- this project already found
# directly (see models/spatial_attribution/lisa.py's own note) that a fixed distance like the
# Stage 1 CR-residual's 3,956m semivariogram range gives wildly uneven neighbour COUNTS across
# this data (681 to 7,141 neighbours per plot at that one distance), which would make the
# comparison across models unfair if models differ in HOW their residuals are spatially arranged,
# not just how large the residuals are. k=8 matches this project's own established LISA choice.
#
# Runs entirely from data already on disk (the DNN/GNNWR test-prediction CSVs, and this file's own
# quick local rerun of the existing spatial CV code for Elastic Net/XGBoost) -- no cluster
# resubmission needed for any of this.

from __future__ import annotations

import warnings
from pathlib import Path

import esda
import libpysal
import numpy as np
import pandas as pd

from models.growth_curve_attribution.broad_environmental_check import columns_for_groups
from models.growth_curve_attribution.gnnwr_check import build_scope_table
from models.growth_curve_attribution.scale_comparison_check import TARGET, merge_environmental_features
from models.growth_curve_attribution.spatial_cv_check import run_spatial_cv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GNNWR_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "growth_curve_attribution" / "gnnwr"
SIMPLE_DNN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "growth_curve_attribution" / "simple_dnn"
COMPARTMENT_MIXED_DNN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "growth_curve_attribution" / "compartment_mixed_dnn"

DEFAULT_K = 8  # matches models/spatial_attribution/lisa.py's own established choice


def compute_residual_morans_i(x, y, residual, k=DEFAULT_K, permutations=999, seed=42):
    mask = ~(np.isnan(residual) | np.isnan(x) | np.isnan(y))
    x, y, residual = np.asarray(x)[mask], np.asarray(y)[mask], np.asarray(residual)[mask]
    coordinates = np.column_stack([x, y])

    with warnings.catch_warnings():
        # Aberfoyle is genuinely several separate forest blocks -- a nearest-neighbour graph
        # over disconnected blocks is expected here, not a bug (same note as lisa.py's own).
        warnings.simplefilter("ignore", category=UserWarning)
        weights = libpysal.weights.KNN.from_array(coordinates, k=k)
    weights.transform = "r"

    np.random.seed(seed)  # esda.moran.Moran's permutation test uses numpy's global random state, no seed= kwarg
    result = esda.moran.Moran(residual, weights, permutations=permutations)
    return result.I, result.p_sim, mask.sum()


def check_gnnwr_and_dnn_outputs(cohort: str, scope: str) -> pd.DataFrame:
    rows = []

    # ----- GNNWR: every saved CSV already has x/y (joined back via GNNWR.getCoefs()) -----
    for csv_path in sorted(GNNWR_OUTPUT_DIR.glob(f"gnnwr_{scope}_{cohort}*_test_predictions.csv")):
        table = pd.read_csv(csv_path)
        residual = table[TARGET] - table["denormalized_pred_result"]
        morans_i, p_value, n = compute_residual_morans_i(table["x"], table["y"], residual)
        rows.append({"model": f"GNNWR ({csv_path.stem})", "morans_i": morans_i, "p_value": p_value, "n_plots": n})

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
        morans_i, p_value, n = compute_residual_morans_i(table["x"], table["y"], residual)
        rows.append({"model": label, "morans_i": morans_i, "p_value": p_value, "n_plots": n})

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

    out_of_fold_predictions, _ = run_spatial_cv(merged, feature_columns)
    # run_spatial_cv() doesn't keep x/y in its returned columns (it only needs them internally,
    # for the leakage buffer) -- merge them back in from the same table it was given.
    coordinate_lookup = merged[["identification", "x", "y"]].drop_duplicates("identification")
    out_of_fold_predictions = out_of_fold_predictions.merge(coordinate_lookup, on="identification", how="left", validate="many_to_one")

    rows = []
    for predicted_col, label in [("elastic_net_predicted", "Elastic Net"), ("xgboost_predicted", "XGBoost")]:
        residual = out_of_fold_predictions[TARGET] - out_of_fold_predictions[predicted_col]
        morans_i, p_value, n = compute_residual_morans_i(out_of_fold_predictions["x"], out_of_fold_predictions["y"], residual)
        rows.append({"model": label, "morans_i": morans_i, "p_value": p_value, "n_plots": n})
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

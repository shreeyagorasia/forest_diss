# Purpose: does RQ2a's own residual-REDUCTION value (how much environmental conditioning helped
# or hurt a given row, |CR error| - |model error|) cluster spatially, or is it scattered randomly
# across Aberfoyle? This was flagged as an open question during the figure-planning pass -- a
# spatial "help map" was explicitly NOT proposed there because no supporting spatial-clustering
# statistic existed yet for the reduction values themselves (only for raw residuals, in RQ2b/RQ3).
# This closes that gap using entirely existing pieces, no new fitting:
#   - compute_residual_reduction() (models/spatial_attribution/rq2_residual_reduction.py) for the
#     per-row reduction value, already the exact function behind RQ2a's own quartile tables.
#   - load_plot_coordinates() (models/common/geo.py) for x/y.
#   - compute_residual_morans_i() (models/growth_curve_attribution/
#     residual_spatial_autocorrelation_check.py) for the actual test -- the same function, same
#     k=8 nearest-neighbour weights, same convention already used for RQ2b's and RQ3's own
#     residual Moran's I.
#
# Run as: python -m models.spatial_attribution.rq2a_reduction_morans_i

import pandas as pd

from models.common.geo import load_plot_coordinates
from models.growth_curve_attribution.residual_spatial_autocorrelation_check import compute_residual_morans_i
from models.spatial_attribution.rq2_residual_reduction import compute_residual_reduction, load_predictions

N_FOLDS = 5
COHORTS = ["4survey", "6survey"]
MODELS = {
    "DNN (env-conditioned, Set3)": "rq1_dnn_env_terrain_nested_set3_gated_terrain_wind_vif_seed42",
    "XGBoost (env-conditioned, Set3)": "xgb_env_terrain_fixed_nested_set3_gated_terrain_wind_vif",
}


def pooled_reduction_with_xy(model_run, cohort, coordinates):
    cr_frames, model_frames = [], []
    for fold in range(N_FOLDS):
        cr_frames.append(load_predictions(f"chapman_richards_fold{fold}", cohort, "spatial_block_kfold"))
        model_frames.append(load_predictions(model_run, cohort, "spatial_block_kfold", held_out_fold=fold))
    cr_all = pd.concat(cr_frames, ignore_index=True)
    model_all = pd.concat(model_frames, ignore_index=True)
    _, _, merged = compute_residual_reduction(cr_all, model_all)
    merged = merged.merge(coordinates, on="identification", how="inner")
    return merged


def main():
    coordinates = load_plot_coordinates()
    for cohort in COHORTS:
        print(f"=== {cohort} ===")
        for name, run in MODELS.items():
            merged = pooled_reduction_with_xy(run, cohort, coordinates)
            morans_i, p_value, n = compute_residual_morans_i(merged["x"], merged["y"], merged["reduction"])
            print(f"  {name}: n={n:,}  Moran's I={morans_i:+.4f}  p={p_value:.3f}")
        print()


if __name__ == "__main__":
    main()

# Purpose: Moran's I on a DNN/PINN model's own held-out residuals -- "does this model still
# leave real, unexplained spatial pattern in its errors", the same question
# models/xgb_environmental/grouped_analysis.py::residual_morans_i() answers for the
# environmental-attribution notebook, but applied directly to an ALREADY-EVALUATED model's saved
# predictions.csv instead of refitting -- cheap, since evaluate_*.py already computed and saved
# every residual this needs.
#
# Scoped deliberately (2026-08-05, per documentation/experiment_log.md's dissertation-argument
# framing): only worth running for spatial_block/spatial_block_kfold -- Moran's I answers a
# spatial-pattern question, so it's meaningless for temporal (rows aren't held out by location
# there) and low-value for plot_level (the split's own random shuffling already tends to erase
# whatever spatial clustering existed). Run it for the models that carry the dissertation's
# spatial argument, not blanket across all ~50 result-table cells.
#
# DESIGN NOTE: predictions.csv has one row per plot-YEAR (a plot appears 4-6 times), but plot
# coordinates are static per plot. Computing Moran's I directly on every row would treat a
# plot's own repeated, near-identical-location rows as independent spatial observations --
# exactly the pseudo-replication problem already flagged for compartment-level resampling
# elsewhere in this project (see models/growth_curve_attribution/bootstrap_ci_check.py's own
# header). Aggregated to one MEAN residual per plot first, matching that established "the plot
# is the real independent unit" convention, before running the spatial statistics.
#
# Run as: python -m models.common.spatial_pattern_check --model-name pinn_env_terrain_k --cohort 4survey --split-type spatial_block
#     or: python -m models.common.spatial_pattern_check --model-name pinn_env_terrain_k --cohort 4survey --split-type spatial_block_kfold --n-folds 5

import argparse
import contextlib
import io
import json

import pandas as pd

from models.common.geo import load_plot_coordinates
from models.common.saving import model_output_dir
from models.spatial_attribution.spatial_autocorrelation import global_morans_i, semivariogram_range


def load_one_row_per_plot_residuals(model_name, cohort, split_type, run_name=None, n_folds=5):
    output_model_name = run_name if run_name else model_name

    if split_type == "spatial_block_kfold":
        # Pool every fold's own held-out residuals first -- same "whole population, not one
        # ~20% slice" reasoning as models/common/kfold_summary.py.
        all_fold_predictions = []
        for fold_index in range(n_folds):
            predictions_path = model_output_dir(output_model_name, cohort, f"fold_{fold_index}", split_type=split_type) / "predictions.csv"
            all_fold_predictions.append(pd.read_csv(predictions_path))
        predictions_df = pd.concat(all_fold_predictions, ignore_index=True)
    else:
        predictions_path = model_output_dir(output_model_name, cohort, split_type=split_type) / "predictions.csv"
        predictions_df = pd.read_csv(predictions_path)

    one_row_per_plot = (
        predictions_df.groupby("identification", as_index=False)
        .agg(mean_residual=("residual", "mean"), cpmt=("cpmt", "first"))
    )
    return one_row_per_plot


def compute_spatial_pattern(model_name, cohort, split_type, run_name=None, n_folds=5, max_distance=5000):
    plot_residuals = load_one_row_per_plot_residuals(model_name, cohort, split_type, run_name=run_name, n_folds=n_folds)

    coordinates = load_plot_coordinates()
    merged = plot_residuals.merge(coordinates, on="identification", how="inner")

    range_estimate, status = semivariogram_range(
        merged["x"].values, merged["y"].values, merged["mean_residual"].values, max_distance=max_distance,
    )
    result = {"n_plots": len(merged), "variogram_status": status, "variogram_range_m": float(range_estimate)}

    if status not in ("resolved", "exceeds_window"):
        result["morans_i"] = None
        result["morans_i_p"] = None
        return result

    # Aberfoyle is genuinely several separate forest blocks -- libpysal prints one "island (no
    # neighbors)" line per isolated plot directly to stdout, not a catchable Python warning,
    # which can run to thousands of lines. Expected, not an error -- suppressed so it doesn't
    # flood the terminal (same suppression residual_morans_i() already uses).
    with contextlib.redirect_stdout(io.StringIO()):
        morans_i, morans_p = global_morans_i(
            merged["x"].values, merged["y"].values, merged["mean_residual"].values, distance=range_estimate,
        )
    result["morans_i"] = float(morans_i)
    result["morans_i_p"] = float(morans_p)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default=None, help="Omit to check both cohorts.")
    parser.add_argument("--split-type", choices=["spatial_block", "spatial_block_kfold"], default="spatial_block")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--n-folds", type=int, default=5)
    args = parser.parse_args()

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]
    for cohort in cohorts:
        output_model_name = args.run_name if args.run_name else args.model_name
        try:
            result = compute_spatial_pattern(
                args.model_name, cohort, args.split_type, run_name=args.run_name, n_folds=args.n_folds,
            )
        except FileNotFoundError as error:
            print(f"{output_model_name} ({cohort}, {args.split_type}): SKIPPED -- {error}")
            continue

        print(f"===== {output_model_name} ({cohort}, {args.split_type}) =====")
        print(f"  n_plots={result['n_plots']:,}  variogram_status={result['variogram_status']}  range={result['variogram_range_m']:.0f}m")
        if result["morans_i"] is None:
            print("  Moran's I: not computed (variogram did not resolve)")
        else:
            print(f"  Moran's I: {result['morans_i']:.4f}  (p={result['morans_i_p']:.3f})")

        output_dir = model_output_dir(output_model_name, cohort, split_type=args.split_type)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "spatial_pattern.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved -> {output_dir / 'spatial_pattern.json'}")
        print()


if __name__ == "__main__":
    main()

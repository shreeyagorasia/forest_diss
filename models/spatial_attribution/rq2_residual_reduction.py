# Run as: python -m models.spatial_attribution.rq2_residual_reduction --model-name dnn_env_terrain --cr-run-name chapman_richards --cohort 4survey --split-type spatial_block
#
# RQ2's DNN/PINN angle: reuses RQ1's ALREADY-FIT model predictions. No new fitting at all, just
# a new way of reading the same predictions.csv every DNN/PINN model already saves. RQ1 asks "how
# accurate is this model at predicting height"; RQ2 asks a related but different question: "does
# giving the model environmental information shrink the systematic departure from the shared
# Chapman-Richards curve". I.e. does the model's own error get smaller specifically on the
# plots the shared curve does WORST on, not just smaller on average.
#
# Works because CR baseline's predictions.csv and every DNN/PINN model's predictions.csv share
# the IDENTICAL schema (identification, LiDAR_year, observed_top_height, predicted_top_height,
# residual, split). Confirmed by reading real files, not assumed. Join the two on
# (identification, LiDAR_year), and CR's own residual for that row becomes "how far the shared
# curve was off," directly comparable to the model's residual for the SAME row.

import argparse

import pandas as pd

from models.common.geo import load_plot_coordinates
from models.common.saving import model_output_dir

ROW_KEY = ["identification", "LiDAR_year"]


def load_predictions(run_name, cohort, split_type, held_out_fold=None):
    if held_out_fold is not None:
        output_dir = model_output_dir(run_name, cohort, f"fold_{held_out_fold}", split_type=split_type)
    else:
        output_dir = model_output_dir(run_name, cohort, split_type=split_type)
    predictions = pd.read_csv(output_dir / "predictions.csv")
    return predictions[predictions["split"] == "test"]


def compute_residual_reduction(cr_predictions, model_predictions):
    merged = cr_predictions.merge(
        model_predictions, on=ROW_KEY, how="inner", suffixes=("_cr", "_model"),
    )
    if len(merged) == 0:
        raise ValueError("No matching rows between CR baseline and model predictions. Check cohort/split_type/fold all match.")

    merged["cr_abs_residual"] = merged["residual_cr"].abs()
    merged["model_abs_residual"] = merged["residual_model"].abs()
    merged["reduction"] = merged["cr_abs_residual"] - merged["model_abs_residual"]  # positive = model is better

    # The real question isn't just "is the average residual smaller". It's "does the model
    # help MOST on the plots the shared curve does WORST on" (your own RQ2 framing: does it pull
    # predictions towards the shared relationship where it matters). Splits rows into CR-residual
    # quartiles and reports the reduction separately per quartile, not just one overall number.
    merged["cr_residual_quartile"] = pd.qcut(merged["cr_abs_residual"], 4, labels=["Q1 (smallest CR error)", "Q2", "Q3", "Q4 (largest CR error)"])

    overall = {
        "n_rows": len(merged),
        "cr_mean_abs_residual": merged["cr_abs_residual"].mean(),
        "model_mean_abs_residual": merged["model_abs_residual"].mean(),
        "mean_reduction": merged["reduction"].mean(),
        "pct_rows_improved": (merged["reduction"] > 0).mean() * 100,
    }
    by_quartile = merged.groupby("cr_residual_quartile", observed=True).agg(
        n_rows=("reduction", "size"),
        cr_mean_abs_residual=("cr_abs_residual", "mean"),
        model_mean_abs_residual=("model_abs_residual", "mean"),
        mean_reduction=("reduction", "mean"),
        pct_rows_improved=("reduction", lambda s: (s > 0).mean() * 100),
    )
    return overall, by_quartile, merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=True, help="RQ1 run_name, e.g. dnn_env_terrain, rq1_pinn_env_terrain_k_nested_set2_top10_seed42, final_dnn_seed42.")
    parser.add_argument("--cr-run-name", default="chapman_richards", help="Baseline CR run_name for the SAME split/fold this model was fit under.")
    parser.add_argument("--cohort", default="4survey", choices=["4survey", "6survey"])
    parser.add_argument("--split-type", default="spatial_block", choices=["plot_level", "spatial_block", "spatial_block_kfold", "temporal"])
    parser.add_argument("--fold-index", type=int, default=None, help="Only for spatial_block_kfold. Omit for a single-split model.")
    parser.add_argument(
        "--no-save-per-row", action="store_true",
        help="Skip saving the per-row merged CSV (coordinates joined in). Saved by default so "
             "a later, plot-heavy script has everything needed for maps/outlier-zoom without "
             "rerunning this analysis. Cheap either way (a join + a CSV write), so left on unless "
             "you explicitly don't want it.",
    )
    args = parser.parse_args()

    cr_run_name = args.cr_run_name if args.fold_index is None else f"{args.cr_run_name}_fold{args.fold_index}"
    cr_predictions = load_predictions(cr_run_name, args.cohort, args.split_type)
    model_predictions = load_predictions(args.model_name, args.cohort, args.split_type, held_out_fold=args.fold_index)

    overall, by_quartile, merged = compute_residual_reduction(cr_predictions, model_predictions)

    print(f"===== RQ2 residual reduction: {args.model_name} vs {args.cr_run_name} ({args.cohort}, {args.split_type}) =====")
    print(f"  n_rows={overall['n_rows']:,}")
    print(f"  CR mean |residual|:    {overall['cr_mean_abs_residual']:.4f} m")
    print(f"  Model mean |residual|: {overall['model_mean_abs_residual']:.4f} m")
    print(f"  Mean reduction: {overall['mean_reduction']:+.4f} m  ({overall['pct_rows_improved']:.1f}% of rows improved)")
    print()
    print("  By CR-residual quartile (does the model help most where CR does worst?):")
    print(by_quartile.to_string())

    if not args.no_save_per_row:
        # Coordinates joined in here (not saved in predictions.csv itself, which every DNN/PINN
        # model shares the same schema for). This is the one join a later plotting script would
        # otherwise have to redo itself for every map/outlier-zoom figure, so it's done once, here.
        coordinates = load_plot_coordinates()
        merged_with_xy = merged.merge(coordinates[["identification", "x", "y"]], on="identification", how="left")

        if args.fold_index is not None:
            output_dir = model_output_dir(args.model_name, args.cohort, f"fold_{args.fold_index}", split_type=args.split_type)
        else:
            output_dir = model_output_dir(args.model_name, args.cohort, split_type=args.split_type)
        save_path = output_dir / "rq2_residual_reduction.csv"
        merged_with_xy.to_csv(save_path, index=False)
        print(f"\n  Saved per-row merged data (with x/y) -> {save_path}")


if __name__ == "__main__":
    main()

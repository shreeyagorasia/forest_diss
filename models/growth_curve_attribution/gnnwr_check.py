# Purpose: test whether GNNWR (Geographically Neural Network Weighted Regression, Du et al. 2020,
# https://doi.org/10.1080/13658816.2019.1707834) beats the established static Elastic Net /
# XGBoost models on the per-plot growth-curve target, by letting the environment-to-growth
# relationship vary smoothly across space instead of fitting one fixed global relationship.
#
# Uses the SAME cleaned target, SAME compartment-based spatial_block_split (60 m leakage buffer),
# and SAME feature-scope machinery (columns_for_groups/SCOPE_GROUPS) as the rest of this project,
# so GNNWR's test-set R2 is directly comparable to the already-verified EN/XGBoost numbers:
#   17-feature terrain/wind scope: EN 0.125, XGB 0.117 (outputs/growth_curve_attribution/
#   broad_environmental_spatial_cv_4survey.csv / terrain_wind_management_comparison.csv).
#
# IMPORTANT hardware note (found by reading the installed gnnwr package source directly, not
# assumed): GNNWR's spatial-weighting sub-network (SWNN) takes each plot's full distance-to-every-
# reference-point vector as input, so its input layer width equals the size of the reference set
# (by default, the whole training set). For this project's ~31,000-plot training set that is a
# first dense layer with roughly 500 million parameters, plus multi-GB pairwise distance
# matrices -- this does not fit in this laptop's 8.6 GB RAM. It is designed to run on the
# cluster GPU (see jobs/growth_curve_attribution/run_gnnwr.sh). --subsample-train exists ONLY for
# a quick local smoke test of the code path on a tiny slice; a subsampled run is not a real result
# and must never be reported as one.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models.common.metrics import compute_metrics
from models.common.splits import SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, SPLIT_SEED, spatial_block_split
from models.elasticnet_environmental.elasticnet_environmental import drop_rows_with_missing_features
from models.growth_curve_attribution.broad_environmental_check import columns_for_groups
from models.growth_curve_attribution.scale_comparison_check import TARGET, build_plot_level_table, merge_environmental_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "growth_curve_attribution" / "gnnwr"

# Only scopes that are pure continuous terrain/wind/management columns (no categorical CEH
# classes) are supported here -- the two scopes the council/user agreed to test first. If a
# scope with categorical columns is needed later, it will need one-hot encoding first, the same
# way broad_environmental_check.py does it for Elastic Net/XGBoost.
SCOPES = {
    "terrain_wind": ["terrain_wind"],
    "terrain_wind_plus_management": ["terrain_wind", "management"],
}


def build_scope_table(cohort: str, scope: str, split_seed: int = SPLIT_SEED):
    """Build the cleaned, feature-merged, split-labelled plot table for one feature scope."""
    feature_columns = columns_for_groups(SCOPES[scope])
    plot_table = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
    merged, available_columns = merge_environmental_features(plot_table, feature_columns=feature_columns)
    merged = drop_rows_with_missing_features(merged, available_columns)
    merged["split"] = spatial_block_split(
        merged, block_col=SPATIAL_BLOCK_COL, buffer_distance=SPATIAL_BUFFER_METRES,
        coordinates_df=merged[["identification", "x", "y"]], seed=split_seed,
    )
    return merged, available_columns


def maybe_subsample_train(train_df: pd.DataFrame, subsample_train: int | None, seed: int) -> pd.DataFrame:
    """ONLY for a local smoke test of the code path -- see module docstring. Not a real result."""
    if subsample_train is None or subsample_train >= len(train_df):
        return train_df
    return train_df.sample(n=subsample_train, random_state=seed)


def run_gnnwr(
    cohort: str,
    scope: str,
    max_epoch: int,
    early_stop: int,
    use_gpu: bool,
    subsample_train: int | None = None,
    split_seed: int = SPLIT_SEED,
):
    # Imported here, not at module level -- gnnwr pulls in torch, and this module's
    # build_scope_table()/maybe_subsample_train() are useful even where torch isn't installed
    # (e.g. quick unit checks of the split logic).
    #
    # WORKAROUND (2026-08-04): gnnwr.models.GNNWR() unconditionally constructs a
    # torch.utils.tensorboard.SummaryWriter, whose background protobuf thread crashes the whole
    # process (segfault) on this machine (macOS, Python 3.13, torch 2.13.0) when GNNWR's own
    # PyTorch layers are built in the same process. Confirmed this is not the modelling code or
    # data pipeline: replicating GNNWR.__init__ line-by-line, every actual modelling step (OLS
    # fit, SWNN construction, optimizer, a real forward pass, a 2-epoch training loop) succeeded
    # on its own with this writer swapped for a no-op stand-in, in a minimal process. In THIS
    # module's real import order (after xgboost/shap are already loaded via
    # broad_environmental_check/explain_signal), the same patch was not enough to guarantee a
    # clean run every time -- consistent with a broader native-threading conflict between
    # simultaneously-loaded PyTorch, XGBoost, and TensorBoard on macOS, not one single bug. Kept
    # here anyway since it is free and removes one real contributor. Net conclusion: this
    # machine is not reliable for actually training GNNWR (on top of the separate, definite
    # memory ceiling documented in the module docstring) -- use --subsample-train only to sanity
    # check that build_scope_table()/init_dataset_split() wiring is correct up to the point
    # GNNWR() is constructed, and run the real experiment on the cluster
    # (jobs/growth_curve_attribution/run_gnnwr.sh), whose Linux/CUDA stack every other DNN/PINN
    # job in this project already trains on successfully.
    import torch.utils.tensorboard as _tensorboard_module

    class _NoOpSummaryWriter:
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_graph(self, *args, **kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass

    _tensorboard_module.SummaryWriter = _NoOpSummaryWriter

    from gnnwr.datasets import init_dataset_split
    from gnnwr.models import GNNWR

    table, feature_columns = build_scope_table(cohort, scope, split_seed=split_seed)

    train = table[table["split"] == "train"].copy()
    val = table[table["split"] == "val"].copy()
    test = table[table["split"] == "test"].copy()
    train = maybe_subsample_train(train, subsample_train, seed=split_seed)

    print(f"{cohort} / {scope}: train={len(train):,}  val={len(val):,}  test={len(test):,}  features={len(feature_columns)}")
    if subsample_train is not None:
        print(f"  ** SMOKE TEST: train subsampled to {len(train):,} rows -- NOT a real result **")

    # id_column is deliberately left at its default (None) so gnnwr auto-creates a plain integer
    # 'id' column. Passing our own id_column name (e.g. 'identification') breaks GNNWR.getCoefs(),
    # which hardcodes the literal column name 'id' when joining predictions back onto the original
    # rows -- confirmed by reading models.py directly, not assumed.
    train_dataset, valid_dataset, test_dataset = init_dataset_split(
        train_data=train,
        val_data=val,
        test_data=test,
        x_column=feature_columns,
        y_column=[TARGET],
        spatial_column=["x", "y"],
    )

    run_name = f"gnnwr_{scope}_{cohort}"
    model = GNNWR(
        train_dataset,
        valid_dataset,
        test_dataset,
        use_gpu=use_gpu,
        model_name=run_name,
        model_save_path=str(OUTPUT_DIR / "models"),
        write_path=str(OUTPUT_DIR / "runs" / run_name),
        log_path=str(OUTPUT_DIR / "logs"),
    )
    model.run(max_epoch=max_epoch, early_stop=early_stop)

    print(model.result())

    # model.result_data already joins predictions back onto the original plot-level columns
    # (including our TARGET column) via GNNWR.getCoefs(), called at the end of run().
    result_data = model.result_data
    test_rows = result_data[result_data["dataset_belong"] == "test"]
    metrics = compute_metrics(test_rows[TARGET], test_rows["denormalized_pred_result"])
    print(f"\nIndependently recomputed test R2 (this project's own compute_metrics): {metrics['r2']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / f"{run_name}_test_predictions.csv"
    test_rows.to_csv(result_path, index=False)
    print(f"Saved test predictions to {result_path}")

    return metrics, result_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default="4survey")
    parser.add_argument("--scope", choices=list(SCOPES), default="terrain_wind")
    parser.add_argument("--max-epoch", type=int, default=200)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument(
        "--subsample-train", type=int, default=None,
        help="ONLY for a local smoke test of the code path on a tiny slice -- see module docstring.",
    )
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    run_gnnwr(
        cohort=args.cohort,
        scope=args.scope,
        max_epoch=args.max_epoch,
        early_stop=args.early_stop,
        use_gpu=args.use_gpu,
        subsample_train=args.subsample_train,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()

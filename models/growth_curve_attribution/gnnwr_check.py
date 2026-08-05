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
# IMPORTANT hardware notes (found by reading the installed gnnwr package source directly, then
# confirmed empirically on the cluster). Two SEPARATE bottlenecks, both fixed below by shrinking
# what GNNWR is asked to process, not by requesting bigger hardware -- a specific high-VRAM GPU
# (tried: RTX A6000) is not reliably available on this cluster's queue (jobs sat PD for a full
# day with "ReqNodeNotAvail"), so the fix needs to work on the GENERIC "gpu:1" pool instead.
#
# Bottleneck 1: GNNWR's spatial-weighting sub-network (SWNN) takes each plot's full distance-to-
# every-reference-point vector as input, so its input layer width equals the size of the
# reference set (by default, the whole training set, ~31,000 plots) -- OOM'd a 10.57 GiB GPU on
# the very first optimizer step. Fixed here by capping the reference set to REFERENCE_SET_SIZE
# rows (see subsample_reference_set()), sampled proportionally from every compartment so a
# smaller reference set still covers the whole forest geographically, not just a lucky/unlucky
# random slice. This DOES mean GNNWR sees fewer reference points than EN/XGBoost/the DNN
# baselines see training rows -- a genuine, disclosed methodological difference, not hidden.
#
# Bottleneck 2: separately, gnnwr's DIAGNOSIS class (used only for the per-epoch "Train AIC"
# progress-bar number, and rebuilt once more for the final train/valid/test result() report)
# builds a classic-GWR "hat matrix" by tiling the whole feature matrix passed to it against
# itself -- an O(n^2 * n_features) tensor, where n is THAT DATASET'S OWN row count (train,
# valid, or test), independent of the SWNN reference-set size above. At our scale this is 54 GB
# for the ~31,000-row training set, and ~18 GB even for the ~11,600-row validation/test sets --
# both exceed a 10.57 GiB GPU regardless of how small the reference set is shrunk. This is NOT
# used for the actual gradient step or for choosing which epoch's model to keep (validation R2
# is computed separately, with a plain formula, no hat matrix involved -- confirmed by reading
# __valid() directly) -- it only feeds cosmetic AIC/F-test numbers. Patched below (_FastDiagnosis
# in run_gnnwr()) to skip the hat-matrix construction entirely for any dataset above
# HAT_MATRIX_ROW_LIMIT rows, set low enough to cover train, valid, AND test at this project's
# scale. R2/RMSE/Adjust_R2 do not depend on the hat matrix and stay exact either way -- only
# AIC/AICc (approximated via plain feature count instead of the true GWR-corrected effective
# degrees of freedom) and F1/F2/F3 (unavailable) are affected, and R2/RMSE is what this project
# actually compares against EN/XGBoost/the DNN baselines throughout, so this is a disclosed,
# deliberate trade-off, not a silent loss of rigor.

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


# Default reference-set cap -- see module docstring, "Bottleneck 1". Comfortably below the
# ~11,600-row validation/test sets too, which matters for HAT_MATRIX_ROW_LIMIT below (train must
# stay smaller than val/test for the size-based threshold there to treat all three consistently).
DEFAULT_REFERENCE_SET_SIZE = 6_000


def subsample_reference_set(train_df: pd.DataFrame, reference_set_size: int | None, seed: int) -> pd.DataFrame:
    """Shrink GNNWR's reference/training set to a memory-safe size (see module docstring).

    Samples roughly the same FRACTION of rows from every compartment (not a plain random sample
    of the whole table), so a smaller reference set still covers the whole forest geographically
    instead of over- or under-representing individual compartments by chance. Pass
    reference_set_size=None to use the full population (only safe with a high-VRAM GPU -- see
    module docstring).
    """
    if reference_set_size is None or reference_set_size >= len(train_df):
        return train_df
    fraction = reference_set_size / len(train_df)
    # GroupBy.sample() (not .apply(lambda g: g.sample(...))) -- pandas 3.x's groupby-apply
    # permanently drops the grouping column from what the function receives (confirmed directly:
    # include_groups=True raises "no longer allowed"), which silently deleted 'cpmt' here. The
    # dedicated GroupBy.sample() method has no such issue.
    sampled = train_df.groupby("cpmt", group_keys=False).sample(frac=fraction, random_state=seed)
    return sampled


def run_gnnwr(
    cohort: str,
    scope: str,
    max_epoch: int,
    early_stop: int,
    use_gpu: bool,
    reference_set_size: int | None = DEFAULT_REFERENCE_SET_SIZE,
    split_seed: int = SPLIT_SEED,
):
    # Imported here, not at module level -- gnnwr pulls in torch, and this module's
    # build_scope_table()/subsample_reference_set() are useful even where torch isn't installed
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
    # machine is not reliable for actually training GNNWR end-to-end -- local runs are useful for
    # sanity-checking that build_scope_table()/init_dataset_split()/a few epochs all wire up
    # correctly, but the real experiment still belongs on the cluster
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

    import torch

    from gnnwr.datasets import init_dataset_split
    import gnnwr.models as _gnnwr_models
    from gnnwr.models import GNNWR
    from gnnwr.utils import DIAGNOSIS as _RealDiagnosis

    # WORKAROUND (2026-08-04/05, found on the cluster): gnnwr's DIAGNOSIS class (used for the
    # per-epoch "Train AIC" progress-bar number, and rebuilt once more for the final train/valid/
    # test report) computes a classic-GWR "hat matrix" by literally tiling the whole feature
    # matrix passed to it against itself: `x_data.repeat(n, 1)` where n is THAT DATASET'S OWN row
    # count -- train, valid, or test, independent of the SWNN reference-set size shrunk above.
    # This is 54 GB for the ~31,000-row training set, and ~18 GB even for the ~11,600-row
    # validation/test sets -- both exceed the generic 10.57 GiB cluster GPU. This computation is
    # NOT used for the actual gradient step or for choosing which epoch's model to keep
    # (validation R2 is computed separately, with a plain formula, no hat matrix involved --
    # confirmed by reading __valid() directly) -- it only feeds cosmetic AIC/F-test numbers. So
    # HAT_MATRIX_ROW_LIMIT is set low enough to cover train, valid, AND test at this project's
    # scale (all comfortably above it), guaranteeing the cheap path everywhere rather than
    # relying on a specific GPU's VRAM being large enough for the real one. R2/RMSE/Adjust_R2 do
    # not depend on the hat matrix at all and stay exact either way -- only AIC/AICc (here
    # approximated with the plain feature count instead of the true GWR-corrected effective
    # degrees of freedom) and F1/F2/F3 (unavailable) are affected, and R2/RMSE is what this
    # project actually compares against EN/XGBoost/the DNN baselines throughout.
    HAT_MATRIX_ROW_LIMIT = 2_000

    class _FastDiagnosis(_RealDiagnosis):
        def __init__(self, weight, x_data, y_data, y_pred):
            if len(y_data) <= HAT_MATRIX_ROW_LIMIT:
                super().__init__(weight, x_data, y_data, y_pred)
                return
            # Same cheap bookkeeping the real class does, using its own private attribute names
            # (Python name-mangles them to _DIAGNOSIS__x on any subclass) so the inherited
            # R2()/RMSE()/Adjust_R2()/AIC()/AICc() methods keep working unchanged.
            k = x_data.shape[1]
            self.__dict__["_DIAGNOSIS__weight"] = weight.clone()
            self.__dict__["_DIAGNOSIS__x_data"] = x_data.clone()
            self.__dict__["_DIAGNOSIS__y_data"] = y_data.clone()
            self.__dict__["_DIAGNOSIS__y_pred"] = y_pred.clone()
            self.__dict__["_DIAGNOSIS__n"] = len(y_data)
            self.__dict__["_DIAGNOSIS__k"] = k
            self.__dict__["_DIAGNOSIS__residual"] = y_data - y_pred
            self.__dict__["_DIAGNOSIS__ssr"] = torch.sum((y_pred - y_data) ** 2)
            # Must be a tensor, not a plain float -- run()'s own progress bar calls
            # .AIC().data.cpu().numpy() on this every epoch, which only works on a tensor.
            self.__dict__["_DIAGNOSIS__S"] = torch.tensor(float(k), device=weight.device)  # approx effective degrees of freedom

    # Python looks up "DIAGNOSIS" inside gnnwr.models.__train()/__evaluate() from the models
    # module's own namespace at call time, not at gnnwr.models' own import time -- so patching
    # this attribute here (after gnnwr.models has already been imported) still takes effect for
    # every DIAGNOSIS(...) call made from inside that module from now on.
    _gnnwr_models.DIAGNOSIS = _FastDiagnosis

    table, feature_columns = build_scope_table(cohort, scope, split_seed=split_seed)

    train = table[table["split"] == "train"].copy()
    val = table[table["split"] == "val"].copy()
    test = table[table["split"] == "test"].copy()
    full_train_size = len(train)
    train = subsample_reference_set(train, reference_set_size, seed=split_seed)

    print(f"{cohort} / {scope}: train={len(train):,}  val={len(val):,}  test={len(test):,}  features={len(feature_columns)}")
    if len(train) < full_train_size:
        print(f"  Reference set capped to {len(train):,} of {full_train_size:,} training plots (compartment-stratified) -- see module docstring, Bottleneck 1")

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
        "--reference-set-size", type=int, default=DEFAULT_REFERENCE_SET_SIZE,
        help=(
            "Cap GNNWR's reference/training set to this many plots (compartment-stratified) so it "
            "fits a generic ~10.5 GiB GPU -- see module docstring, Bottleneck 1. Pass 0 to use the "
            "full training population instead (only safe with a high-VRAM GPU such as an A6000)."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    run_gnnwr(
        cohort=args.cohort,
        scope=args.scope,
        max_epoch=args.max_epoch,
        early_stop=args.early_stop,
        use_gpu=args.use_gpu,
        reference_set_size=args.reference_set_size if args.reference_set_size > 0 else None,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()

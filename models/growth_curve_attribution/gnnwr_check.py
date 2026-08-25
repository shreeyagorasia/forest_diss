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
# what GNNWR is asked to process, not by requesting bigger hardware. A specific high-VRAM GPU
# (tried: RTX A6000) is not reliably available on this cluster's queue (jobs sat PD for a full
# day with "ReqNodeNotAvail"), so the fix needs to work on the GENERIC "gpu:1" pool instead.
#
# Bottleneck 1: GNNWR's spatial-weighting sub-network (SWNN) takes each plot's full distance-to-
# every-reference-point vector as input, so its input layer width equals the size of the
# reference set (by default, the whole training set, ~31,000 plots). OOM'd a 10.57 GiB GPU on
# the very first optimizer step. Fixed here by capping the reference set to REFERENCE_SET_SIZE
# rows (see subsample_reference_set()), sampled proportionally from every compartment so a
# smaller reference set still covers the whole forest geographically, not just a lucky/unlucky
# random slice. This DOES mean GNNWR sees fewer reference points than EN/XGBoost/the DNN
# baselines see training rows. A genuine, disclosed methodological difference, not hidden.
#
# Bottleneck 2: separately, gnnwr's DIAGNOSIS class (used only for the per-epoch "Train AIC"
# progress-bar number, and rebuilt once more for the final train/valid/test result() report)
# builds a classic-GWR "hat matrix" by tiling the whole feature matrix passed to it against
# itself. An O(n^2 * n_features) tensor, where n is THAT DATASET'S OWN row count (train,
# valid, or test), independent of the SWNN reference-set size above. At our scale this is 54 GB
# for the ~31,000-row training set, and ~18 GB even for the ~11,600-row validation/test sets --
# both exceed a 10.57 GiB GPU regardless of how small the reference set is shrunk. This is NOT
# used for the actual gradient step or for choosing which epoch's model to keep (validation R2
# is computed separately, with a plain formula, no hat matrix involved. Confirmed by reading
# __valid() directly). It only feeds cosmetic AIC/F-test numbers. Patched below (_FastDiagnosis
# in run_gnnwr()) to skip the hat-matrix construction entirely for any dataset above
# HAT_MATRIX_ROW_LIMIT rows, set low enough to cover train, valid, AND test at this project's
# scale. R2/RMSE/Adjust_R2 do not depend on the hat matrix and stay exact either way. Only
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
from models.common.splits import (
    DEFAULT_K_FOLDS,
    SPATIAL_BLOCK_COL,
    SPATIAL_BUFFER_METRES,
    SPLIT_SEED,
    spatial_block_split,
    spatial_kfold_split,
)
from models.elasticnet_environmental.elasticnet_environmental import CATEGORICAL_COLUMNS, drop_rows_with_missing_features
from models.growth_curve_attribution.broad_environmental_check import (
    add_missing_dummy_columns_as_zero,
    columns_for_groups,
    prepare_broad_table,
)
from models.growth_curve_attribution.scale_comparison_check import TARGET, build_plot_level_table, merge_environmental_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "growth_curve_attribution" / "gnnwr"

# terrain_wind and terrain_wind_plus_management are pure continuous columns. Build_scope_table
# below still builds them with its ORIGINAL, unchanged merge (never touched, since this project's
# already-published, significance-tested GNNWR numbers were computed with it).
#
# broad_environment_plus_management (added 2026-08-08) is the one broader scope EN/XGBoost show
# a real signal for (0.290/0.318) that GNNWR has not yet been tested on, and the only scope where
# checking GNNWR is a genuinely open question rather than repeating an already-established null
# (climate/soil/edge alone showed no robust improvement for either EN or XGBoost. See
# broad_environmental_check.py's own "Main finding"). This scope needs the SAME cohort-suffixed
# climate-column resolution and one-hot-encoded soil categoricals (ceh_pedotope,
# ceh_subsurface_drainage, ceh_textural_composition) that EN/XGBoost's own broader-scope numbers
# use. Handled below by reusing broad_environmental_check.py's prepare_broad_table() rather than
# reimplementing it, so GNNWR sees an identical feature representation to the models it is being
# compared against.
SCOPES = {
    "terrain_wind": ["terrain_wind"],
    "terrain_wind_plus_management": ["terrain_wind", "management"],
    "broad_environment_plus_management": [
        "terrain_wind", "climate", "soil_site", "edge_position", "management",
    ],
}


def build_scope_table(
    cohort: str,
    scope: str,
    split_seed: int = SPLIT_SEED,
    held_out_fold: int | None = None,
    k_folds: int = DEFAULT_K_FOLDS,
):
    """Build the cleaned, feature-merged, split-labelled plot table for one feature scope.

    held_out_fold=None (the default) keeps this function's original behaviour EXACTLY: one
    single train/val/test split via spatial_block_split(), same as every run so far tonight.
    Passing an integer 0..k_folds-1 instead switches to ONE FOLD of a proper K-fold spatial CV
    (spatial_kfold_split(). The same helper Elastic Net/XGBoost's own headline numbers, e.g.
    the 0.125/0.117 this project compares GNNWR against, are pooled across). Running this once
    per held_out_fold and pooling the out-of-fold test predictions gives a GNNWR number that is
    actually comparable to those EN/XGBoost numbers, instead of the single-split estimate used so
    far, which this project's own seed-sweep work already showed has real variance (EN's 5-fold
    pooled R2 itself moves +/-0.023 across different fold-assignment seeds for 4survey).
    """
    if scope in ("terrain_wind", "terrain_wind_plus_management"):
        # ORIGINAL path, unchanged. Exactly what this project's already-published,
        # significance-tested GNNWR numbers were computed with.
        feature_columns = columns_for_groups(SCOPES[scope])
        plot_table = build_plot_level_table(cohort, apply_disturbance_cleaning=True)
        merged, available_columns = merge_environmental_features(plot_table, feature_columns=feature_columns)
        merged = drop_rows_with_missing_features(merged, available_columns)
    else:
        # NEW (2026-08-08) path for scopes with cohort-suffixed climate columns and/or
        # categorical soil columns. Reuses EN/XGBoost's own cohort-resolution/one-hot logic
        # (prepare_broad_table) instead of merge_environmental_features, which would silently
        # drop cohort-suffixed columns and cannot one-hot encode categoricals at all.
        raw_columns = columns_for_groups(SCOPES[scope])
        merged, available_columns = prepare_broad_table(cohort, raw_columns)
    if held_out_fold is None:
        merged["split"] = spatial_block_split(
            merged, block_col=SPATIAL_BLOCK_COL, buffer_distance=SPATIAL_BUFFER_METRES,
            coordinates_df=merged[["identification", "x", "y"]], seed=split_seed,
        )
    else:
        merged["split"] = spatial_kfold_split(
            merged, block_col=SPATIAL_BLOCK_COL, k=k_folds, held_out_fold=held_out_fold,
            buffer_distance=SPATIAL_BUFFER_METRES, coordinates_df=merged[["identification", "x", "y"]],
            seed=split_seed,
        )

    # Found 2026-08-08 (broad_environment_plus_management, folds 0 and 4): a one-hot dummy
    # column for a rare CEH category (e.g. ceh_pedotope=2.0) can be entirely absent from ONE
    # fold's own training compartments purely by chance, leaving it constant (all zero) in that
    # fold's train split. GNNWR's own MinMax scaling ((x - min) / (max - min)) then divides by
    # zero and feeds NaN into its internal OLS init, crashing before training even starts.
    # Checked directly against this fold's own train rows (not the whole population). A column
    # can be fine in 3 of 5 folds and constant in the other 2, since which compartments land in
    # train varies by fold. Dropped per fold rather than globally, and printed, so this is a
    # disclosed per-fold difference, not a silent one.
    train_rows = merged[merged["split"] == "train"]
    zero_variance_columns = [
        column for column in available_columns if train_rows[column].max() == train_rows[column].min()
    ]
    if zero_variance_columns:
        print(f"  Dropping {len(zero_variance_columns)} column(s) constant in this fold's own training set: {zero_variance_columns}")
        available_columns = [column for column in available_columns if column not in zero_variance_columns]

    return merged, available_columns


def build_table_from_columns(
    cohort: str,
    raw_columns: list[str],
    split_seed: int = SPLIT_SEED,
    held_out_fold: int | None = None,
    k_folds: int = DEFAULT_K_FOLDS,
):
    """Raw-column-list sibling of build_scope_table(). Same cohort-suffix resolution, one-hot
    encoding, fold-aware split, and zero-variance-column dropping, but takes an explicit column
    list directly instead of a named scope resolved through SCOPES. Built for the new
    rank-aggregate environmental-feature methodology (see
    models/xgb_environmental/feature_set_builder.py). A new function, not a parameter added to
    build_scope_table() itself, so no existing named-scope caller's behaviour can change.

    raw_columns can mix plain continuous names with SPECIFIC one-hot dummy names already in
    "category=value" form (e.g. "ceh_textural_composition=5.0"). Exactly what
    documentation/env_feature_sets_manifest.csv stores for RSQ3's Set4/Set5. Same unpack/filter
    approach as broad_environmental_check.py's own run_columns() sibling, reused here rather than
    duplicated, since prepare_broad_table() (called below, same as build_scope_table()'s own
    broader-scope path) only knows how to expand a BARE categorical name to every one of its
    dummies, not a specific requested value.
    """
    bare_columns = []
    requested_dummy_columns = []
    for column in raw_columns:
        if "=" in column:
            bare_category = column.split("=")[0]
            if bare_category not in bare_columns:
                bare_columns.append(bare_category)
            requested_dummy_columns.append(column)
        else:
            bare_columns.append(column)

    merged, all_available_columns = prepare_broad_table(cohort, bare_columns)
    merged, zero_filled_dummies = add_missing_dummy_columns_as_zero(merged, requested_dummy_columns, all_available_columns)
    if zero_filled_dummies:
        print(f"  Added {len(zero_filled_dummies)} requested dummy column(s) as all-zero (absent from {cohort}'s population): {zero_filled_dummies}")

    continuous_requested = [column for column in raw_columns if "=" not in column]
    available_columns = continuous_requested + requested_dummy_columns

    if held_out_fold is None:
        merged["split"] = spatial_block_split(
            merged, block_col=SPATIAL_BLOCK_COL, buffer_distance=SPATIAL_BUFFER_METRES,
            coordinates_df=merged[["identification", "x", "y"]], seed=split_seed,
        )
    else:
        merged["split"] = spatial_kfold_split(
            merged, block_col=SPATIAL_BLOCK_COL, k=k_folds, held_out_fold=held_out_fold,
            buffer_distance=SPATIAL_BUFFER_METRES, coordinates_df=merged[["identification", "x", "y"]],
            seed=split_seed,
        )

    # Same per-fold zero-variance guard as build_scope_table(). A rare dummy category can be
    # entirely absent from one fold's own training compartments by chance (see that function's
    # own comment for the full MinMax-divide-by-zero mechanism this protects against).
    train_rows = merged[merged["split"] == "train"]
    zero_variance_columns = [
        column for column in available_columns if train_rows[column].max() == train_rows[column].min()
    ]
    if zero_variance_columns:
        print(f"  Dropping {len(zero_variance_columns)} column(s) constant in this fold's own training set: {zero_variance_columns}")
        available_columns = [column for column in available_columns if column not in zero_variance_columns]

    return merged, available_columns


# Default reference-set cap. See module docstring, "Bottleneck 1". Sized against the SAME cost
# model that predicted the real observed crash at the full 31,117 rows almost exactly (est. 12.9
# GB vs the actual "10.51 GiB in use" OOM on the 10.57 GiB generic GPU). So this number is
# calibrated, not guessed. That model's estimate for 16,000 rows is ~4.1 GB, comfortably inside
# the 10.57 GiB generic "gpu:1" pool (the only GPU type reliably schedulable on this cluster's
# queue. A specific high-VRAM GPU, e.g. RTX A6000, sat pending for a full day with
# "ReqNodeNotAvail" when tried). 16,000 rows is roughly 2.7x the plots the earlier 6,000-row runs
# used, while still leaving several GB of margin. (Memory does not grow smoothly with this
# number. It jumps in steps, because the SWNN's first layer width snaps to the nearest power of
# 2 below the reference-set size. So 16,000 was chosen because it sits comfortably inside a
# step, not right at its edge.) HAT_MATRIX_ROW_LIMIT below is a SEPARATE, already-patched
# bottleneck (gnnwr's own O(n^2) diagnostic tensor) and no longer constrains this number at all.
DEFAULT_REFERENCE_SET_SIZE = 16_000


def subsample_reference_set(train_df: pd.DataFrame, reference_set_size: int | None, seed: int) -> pd.DataFrame:
    """Shrink GNNWR's reference/training set to a memory-safe size (see module docstring).

    Samples roughly the same FRACTION of rows from every compartment (not a plain random sample
    of the whole table), so a smaller reference set still covers the whole forest geographically
    instead of over- or under-representing individual compartments by chance. Pass
    reference_set_size=None to use the full population (only safe with a high-VRAM GPU. See
    module docstring).
    """
    if reference_set_size is None or reference_set_size >= len(train_df):
        return train_df
    fraction = reference_set_size / len(train_df)
    # GroupBy.sample() (not .apply(lambda g: g.sample(...))). Pandas 3.x's groupby-apply
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
    held_out_fold: int | None = None,
    k_folds: int = DEFAULT_K_FOLDS,
    raw_columns: list[str] | None = None,
    compartment_subset: list | None = None,
):
    # Imported here, not at module level. Gnnwr pulls in torch, and this module's
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
    # clean run every time. Consistent with a broader native-threading conflict between
    # simultaneously-loaded PyTorch, XGBoost, and TensorBoard on macOS, not one single bug. Kept
    # here anyway since it is free and removes one real contributor. Net conclusion: this
    # machine is not reliable for actually training GNNWR end-to-end. Local runs are useful for
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
    # count. Train, valid, or test, independent of the SWNN reference-set size shrunk above.
    # This is 54 GB for the ~31,000-row training set, and ~18 GB even for the ~11,600-row
    # validation/test sets. Both exceed the generic 10.57 GiB cluster GPU. This computation is
    # NOT used for the actual gradient step or for choosing which epoch's model to keep
    # (validation R2 is computed separately, with a plain formula, no hat matrix involved --
    # confirmed by reading __valid() directly). It only feeds cosmetic AIC/F-test numbers. So
    # HAT_MATRIX_ROW_LIMIT is set low enough to cover train, valid, AND test at this project's
    # scale (all comfortably above it), guaranteeing the cheap path everywhere rather than
    # relying on a specific GPU's VRAM being large enough for the real one. R2/RMSE/Adjust_R2 do
    # not depend on the hat matrix at all and stay exact either way. Only AIC/AICc (here
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
            # Must be a tensor, not a plain float. Run()'s own progress bar calls
            # .AIC().data.cpu().numpy() on this every epoch, which only works on a tensor.
            self.__dict__["_DIAGNOSIS__S"] = torch.tensor(float(k), device=weight.device)  # approx effective degrees of freedom

    # Python looks up "DIAGNOSIS" inside gnnwr.models.__train()/__evaluate() from the models
    # module's own namespace at call time, not at gnnwr.models' own import time. So patching
    # this attribute here (after gnnwr.models has already been imported) still takes effect for
    # every DIAGNOSIS(...) call made from inside that module from now on.
    _gnnwr_models.DIAGNOSIS = _FastDiagnosis

    # raw_columns (added 2026-08-10, for the new rank-aggregate environmental-feature
    # methodology): when given, resolves the table via build_table_from_columns() instead of the
    # named-scope build_scope_table(). Everything below this point (training loop, GPU
    # workarounds, output saving) is completely unchanged either way, since both paths return the
    # identical (table, feature_columns) shape. Default None preserves this function's exact
    # existing behaviour for every current named-scope caller.
    if raw_columns is not None:
        table, feature_columns = build_table_from_columns(
            cohort, raw_columns, split_seed=split_seed, held_out_fold=held_out_fold, k_folds=k_folds,
        )
    else:
        table, feature_columns = build_scope_table(
            cohort, scope, split_seed=split_seed, held_out_fold=held_out_fold, k_folds=k_folds,
        )
    if held_out_fold is not None:
        print(f"  K-fold spatial CV: fold {held_out_fold} of {k_folds} held out as test (seed={split_seed})")

    # compartment_subset (added 2026-08-22): restricts the table to a specific list of compartment
    # IDs, applied AFTER the spatial train/val/test split is computed above. So buffer/block
    # logic is unaffected, this only removes rows belonging to compartments outside the subset.
    # Built for the 6-survey-collapse investigation: tests whether GNNWR's R2 collapses when
    # restricted to FEW compartments at FULL point density (matching 6survey's 47-compartment
    # count), as opposed to the already-tested "fewer points, same ~231 compartments" ablation,
    # which did NOT trigger a collapse. Default None preserves existing behaviour exactly.
    if compartment_subset is not None:
        before_rows = len(table)
        table = table[table["cpmt"].isin(compartment_subset)].copy()
        print(f"  compartment_subset: restricted to {len(compartment_subset)} compartments, {len(table):,} of {before_rows:,} rows kept")

    train = table[table["split"] == "train"].copy()
    val = table[table["split"] == "val"].copy()
    test = table[table["split"] == "test"].copy()
    full_train_size = len(train)
    train = subsample_reference_set(train, reference_set_size, seed=split_seed)

    print(f"{cohort} / {scope}: train={len(train):,}  val={len(val):,}  test={len(test):,}  features={len(feature_columns)}")
    if len(train) < full_train_size:
        print(f"  Reference set capped to {len(train):,} of {full_train_size:,} training plots (compartment-stratified). See module docstring, Bottleneck 1")

    # Found 2026-08-08 (broad_environment_plus_management, fold 4: 31,489 train rows): gnnwr's
    # own train DataLoader (init_dataset_split's default batch_size=32) never drops a leftover
    # partial batch, so if the training set size mod batch_size is exactly 1, every epoch's
    # final batch has a single row. Which crashes PyTorch's BatchNorm (needs >=2 samples to
    # compute a batch variance; confirmed via the exact traceback: "Expected more than 1 value
    # per channel when training, got input size torch.Size([1, ...])"). Only 1 in 32 possible
    # remainders trigger this, so it hasn't hit any of this project's other folds by chance, but
    # nothing stops a future cohort/scope/fold combination from landing on it too. Fixed
    # generally (bump batch_size by 1 only in the unlucky case) rather than special-cased to this
    # one run. A batch size of 32 vs 33 is a far smaller optimisation difference than the
    # fold-to-fold R2 variance already documented for this project (std 0.090 across
    # terrain_wind's 5 folds). A disclosed, negligible asymmetry, not a hidden one.
    GNNWR_BATCH_SIZE = 32
    if len(train) % GNNWR_BATCH_SIZE == 1:
        print(f"  Training set size ({len(train):,}) leaves a final batch of exactly 1 row at batch_size={GNNWR_BATCH_SIZE} (BatchNorm cannot train on 1 sample). Using batch_size={GNNWR_BATCH_SIZE + 1} for this run instead.")
        GNNWR_BATCH_SIZE += 1

    # id_column is deliberately left at its default (None) so gnnwr auto-creates a plain integer
    # 'id' column. Passing our own id_column name (e.g. 'identification') breaks GNNWR.getCoefs(),
    # which hardcodes the literal column name 'id' when joining predictions back onto the original
    # rows. Confirmed by reading models.py directly, not assumed.
    train_dataset, valid_dataset, test_dataset = init_dataset_split(
        train_data=train,
        val_data=val,
        test_data=test,
        x_column=feature_columns,
        y_column=[TARGET],
        spatial_column=["x", "y"],
        batch_size=GNNWR_BATCH_SIZE,
    )

    # Includes reference_set_size AND (when in k-fold mode) the held-out fold index in the name,
    # so different reference-set-size runs and different folds all get their own model checkpoint
    # and CSV output instead of silently overwriting each other's results.
    #
    # split_seed label added 2026-08-12: caught before ever being run on the cluster. Run_name
    # never included split_seed at all, so a reseed at a different split_seed (same scope/cohort/
    # fold) would have silently overwritten the original seed's output with an identical filename.
    # Only appended when split_seed differs from the project default (SPLIT_SEED=42), so every
    # already-existing seed-42 run's filename is completely unchanged. Purely additive.
    reference_set_label = "full" if reference_set_size is None else str(reference_set_size)
    fold_label = "" if held_out_fold is None else f"_fold{held_out_fold}of{k_folds}"
    seed_label = "" if split_seed == SPLIT_SEED else f"_seed{split_seed}"
    run_name = f"gnnwr_{scope}_{cohort}_ref{reference_set_label}{fold_label}{seed_label}"
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

    # model.result() prints gnnwr's own classical-GWR diagnostics report (R2/RMSE/AIC/F-tests).
    # F1_Global()/F2_Global()/F3_Local() need the real hat matrix (_DIAGNOSIS__hat), which
    # _FastDiagnosis deliberately never builds for any dataset above HAT_MATRIX_ROW_LIMIT. At
    # this project's scale that is train, valid, AND test, so result() always hits this. Wrapped
    # defensively rather than fixed properly: result() is not needed for anything below (R2/RMSE
    # come from model.result_data, set inside run() itself, independent of result()), so a
    # failure here should never cost the actual results.
    try:
        print(model.result())
    except AttributeError as error:
        print(f"model.result()'s classical GWR diagnostics unavailable under the DIAGNOSIS patch ({error}). R2/RMSE below are unaffected and remain the authoritative comparison metric.")

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
            "fits a generic ~10.5 GiB GPU. See module docstring, Bottleneck 1. Pass 0 to use the "
            "full training population instead (only safe with a high-VRAM GPU such as an A6000)."
        ),
    )
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--held-out-fold", type=int, default=None,
        help=(
            "Run ONE fold of a proper K-fold spatial CV instead of the default single "
            "train/val/test split. Pass 0..k_folds-1, and run this once per fold value, then "
            "pool the resulting test-prediction CSVs for a headline number that's actually "
            "comparable to the EN/XGBoost baseline's own pooled 5-fold R2. Omit (the default) "
            "for the original single-split behaviour."
        ),
    )
    parser.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS)
    args = parser.parse_args()

    run_gnnwr(
        cohort=args.cohort,
        scope=args.scope,
        max_epoch=args.max_epoch,
        early_stop=args.early_stop,
        use_gpu=args.use_gpu,
        reference_set_size=args.reference_set_size if args.reference_set_size > 0 else None,
        split_seed=args.split_seed,
        held_out_fold=args.held_out_fold,
        k_folds=args.k_folds,
    )


if __name__ == "__main__":
    main()

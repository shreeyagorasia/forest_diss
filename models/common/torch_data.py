# Shared data loading, scaling, and encoding helpers for the PyTorch models
# (dnn_noenv, pinn_noenv). Keeping this in one place means both models build
# their tensors the exact same way, so any difference in results is due to
# the model/loss, not to two slightly different data pipelines.
#
# Written in the same plain, explicit style as the rest of models/common/ --
# small named functions, no custom Dataset/Sampler classes, no cleverness.

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from models.common.data import filter_data, load_model_table
from models.common.splits import (
    DEFAULT_K_FOLDS,
    SPATIAL_BLOCK_COL,
    SPATIAL_BUFFER_METRES,
    SPLIT_SEED,
    TEMPORAL_YEARS,
    TEMPORAL_YEARS_NARROW_GAP,
    assert_no_split_columns_in_features,
    spatial_block_split,
    spatial_kfold_split,
    temporal_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The full no-environment feature set every DNN/PINN table already has.
# yldc removed 2026-07-28 -- real ablation showed it hurts test R2 here (0.606->0.647 without
# it), see progress_notes.md. Still present in model_table.parquet for audit, just not selected
# as a feature.
# DEAD CODE, not currently read anywhere: build_tensors() below builds its actual feature set
# from NUMERIC_SCALED_COLUMNS + BINARY_PASSTHROUGH_COLUMNS + Age + thinning_status's one-hot
# encoding, not from this list. Kept as a single "what DNN/PINN use, in one place" summary for a
# reader, but nothing enforces it staying in sync with the lists actually used below -- verify
# against those three if this list is ever suspected of drifting.
NOENV_FEATURE_COLUMNS = [
    "Age",
    "CanopyCover",
    "Thin",
    "time_since_thinning",
    "time_since_thinning_missing",
    "recent_thinning_5yr",
    "thinning_status",
]

# Truly continuous numeric features -- these get standard-scaled as one
# group. Age is scaled SEPARATELY (its own scaler, see fit_scalers below) so
# its training-split standard deviation can be read back out cleanly for the
# physics loss's chain-rule correction.
NUMERIC_SCALED_COLUMNS = ["CanopyCover", "time_since_thinning"]

# Already 0/1 binary flags -- passed through unscaled, same as the one-hot
# thinning_status columns below.
BINARY_PASSTHROUGH_COLUMNS = ["Thin", "time_since_thinning_missing", "recent_thinning_5yr"]

# These two lists (plus Age, scaled separately above, and thinning_status's one-hot encoding)
# are the REAL feature-selection logic build_tensors() reads -- NOENV_FEATURE_COLUMNS above is
# dead code (see its own comment), so guarding it wouldn't actually protect anything.
assert_no_split_columns_in_features(NUMERIC_SCALED_COLUMNS + BINARY_PASSTHROUGH_COLUMNS, "torch_data (DNN/PINN)")

# Raw, unadjusted 95th-percentile LiDAR height (2026-07-28, was Top_Height99 -- see
# progress_notes.md for the full reasoning: Top_Height99 is retired entirely, no fallback
# target kept).
TARGET_COLUMN = "elev_percentile_95th"

# Named, swappable terrain/wind feature sets for dnn_env_terrain/pinn_env_terrain's
# y_max(terrain, wind) sub-network -- same "named FEATURE_SETS dict, picked by name at the CLI"
# convention models/xgb_environmental/xgb_environmental.py already uses, extended here rather
# than hardcoding one single list (2026-08-01 fix -- the first version of this file only offered
# "terrain_wind_solid", with no way to compare against a different scope without editing code).
#
#   - terrain_wind_solid (the default, and the only one actually cross-checked so far): the five
#     variables that were positive on BOTH the category-level XGBoost refit ablation (terrain/
#     wind carry real signal, climate/soil/stand_structure don't) AND the per-variable refit
#     ablation (most of terrain's other members are redundant or near-zero once these five are
#     accounted for) -- decided 2026-07-31, see experiment_log.md's entries that date.
#     `slope_degrees`/`inverse_slope_proxy` deliberately excluded (confirmed exact duplicates,
#     rho=-1.0). Circularity-checked against `Age`'s own known issue (chapman_richards.py::fit()
#     only ever reads Age and elev_percentile_95th -- none of these five touch the CR curve's own
#     construction).
#   - terrain_wind_extended: adds the two BORDERLINE variables (real but modest individual
#     refit-ablation signal) on top of the solid five.
#   - broad: adds the individually-strong variables from OTHER categories (climate/soil/
#     edge-effects) that scored positively in the per-variable refit ablation despite their whole
#     CATEGORY testing badly as a bundle -- this is the wider-scope option flagged, not yet
#     decided on, in experiment_log.md's 2026-07-31 entry. Included as a real option to run and
#     compare, not a silent default -- see --feature-set on run_dnn_env_terrain.py/
#     run_pinn_env_terrain.py. `CanopyCover` deliberately NOT included here even though it scored
#     individually-positive in that same ablation -- it's already one of the no-env features
#     feeding the main network (NUMERIC_SCALED_COLUMNS below), so adding it here too would feed
#     it in twice (once to the main network, once to the y_max sub-network), not add new
#     information -- assert_env_terrain_features_disjoint_from_noenv() below guards against this
#     exact mistake happening again for a different column.
ENV_TERRAIN_FEATURE_SETS = {
    "terrain_wind_solid": ["ceh_twi", "eastness", "elevation", "northness", "topex"],
    "terrain_wind_extended": ["ceh_twi", "eastness", "elevation", "northness", "topex", "whcl", "plan_curvature"],
    "broad": [
        "ceh_twi", "eastness", "elevation", "northness", "topex", "whcl", "plan_curvature",
        "chelsa_bio1_celsius", "soilgrids_ph", "dist_to_watercourse",
    ],
    # Added 2026-08-03: NONE of the sets above contain a genuine wind measurement --
    # "terrain_wind_solid"/"extended"/"broad" all use `topex` (generic topographic exposure, any
    # direction) but never `gwa_wind_speed_10m` (modelled wind climatology, Global Wind Atlas),
    # `windward_topex` (exposure specifically to the PREVAILING wind direction), or `whcl`
    # (external, pre-existing Windthrow Hazard Class rating, ordinal 0-6 -- a real forestry
    # wind-damage assessment, not a proxy). A worst-predicted-plot check this session found
    # windward_topex elevated in the worst-predicted compartments, scaling with error severity --
    # motivated adding these here. Exactly `models.xgb_environmental.xgb_environmental
    # .TERRAIN_AND_WIND_COLUMNS` (this project's own single definition of "terrain+wind", already
    # used for the XGBoost/SHAP/permutation-importance attribution chain) -- not redefined here,
    # imported as a literal list so the two definitions can't drift apart. See
    # documentation/experiment_log.md's 2026-08-03 entry for the reasoning.
    "terrain_wind_full": [
        "elevation", "slope_degrees", "northness", "eastness", "profile_curvature",
        "plan_curvature", "tpi", "elevation_roughness", "ceh_twi", "solar_radiation_index",
        "inverse_slope_proxy", "frost_hollow_flag", "topex", "windward_topex", "whcl",
        # Swapped 10m->50m 2026-08-06: Avenue 2's own council review flagged 10m GWA wind as a
        # SUB-CANOPY measurement, backwards for a mature-canopy target, then formally resolved
        # this by swapping to 50m (documentation/variable_registry_av1_av2.csv). Same physical
        # argument applies here (this project's own trees are mature Sitka spruce) -- adopting
        # AV2's resolved choice, not just the caveat this file already carried unresolved.
        "gwa_wind_speed_50m",
    ],
    # Added 2026-08-03: every column this project has screened and NOT found to be a leak,
    # circular, or otherwise misleading -- xgb_environmental.ALL_FEATURE_COLUMNS (the current,
    # leak-fixed 37-variable set) minus two things: (a) CATEGORY_GROUPS["stand_structure"]
    # (CanopyCover/Thin/time_since_thinning/time_since_thinning_missing/recent_thinning_5yr) --
    # NOT excluded for being untrustworthy, excluded because they're ALREADY fed to the main
    # network via NUMERIC_SCALED_COLUMNS/BINARY_PASSTHROUGH_COLUMNS; including them again here
    # would double-feed the same information through two pathways, exactly what
    # assert_env_terrain_features_disjoint_from_noenv() below exists to catch; (b) ceh_pedotope/
    # ceh_subsurface_drainage/ceh_textural_composition -- unordered categorical class IDs (see
    # the 2026-08-01 LISA-test bug entry in experiment_log.md for why treating these as
    # continuous is invalid), and this project's terrain-tensor pipeline
    # (fit_terrain_scaler/build_terrain_tensor) only knows how to StandardScaler continuous
    # columns, not one-hot-encode categoricals. Deliberately excluded, not silently dropped --
    # a real scoping decision, worth reconsidering if a categorical-aware encoding path is
    # ever added to this pipeline (elasticnet_environmental.py already one-hot-encodes these
    # three, precedent exists).
    #
    # `tas_mean`/`groundfrost_mean` also excluded (2026-08-03, found by hitting the actual
    # error, not anticipated): these are COHORT-SPECIFIC columns in
    # plot_environmental_features.parquet (`tas_mean_4survey`/`tas_mean_6survey`, per
    # models/xgb_environmental/data.py::COHORT_SPECIFIC_COLUMNS) -- xgb_environmental's own
    # pipeline resolves the cohort-suffixed name before use; load_split_table_with_terrain()
    # does not do that resolution, so a plain "tas_mean"/"groundfrost_mean" here fails with a
    # KeyError. A real, structural gap in this pipeline's cohort-agnostic feature-set design,
    # not a reason these two variables are untrustworthy -- worth fixing properly (cohort-aware
    # column resolution in load_split_table_with_terrain()) if HadUK-Grid climate ever needs to
    # be in an env_terrain feature set; deliberately not fixed here to stay scoped to today's
    # actual ask.
    "broad_legitimate": [
        "elevation", "slope_degrees", "northness", "eastness", "profile_curvature",
        "plan_curvature", "tpi", "elevation_roughness", "solar_radiation_index",
        "inverse_slope_proxy", "frost_hollow_flag", "topex", "windward_topex",
        "dist_to_cpmt_boundary", "dist_to_forest_perimeter", "dist_to_scpt_boundary",
        "dist_to_block_boundary", "cpmt_compactness_ratio", "dist_to_road", "dist_to_watercourse",
        # Swapped 10m->50m 2026-08-06 -- same reasoning as terrain_wind_full above.
        "gwa_wind_speed_50m", "soilgrids_ph", "ceh_twi", "chelsa_bio1_celsius",
        "chelsa_gdd5_degc", "chelsa_bio12_precip_mm", "whcl",
    ],
    # Added 2026-08-06: every entry above was built ad hoc -- a variable got added whenever
    # someone had an idea, with correlation checked (if at all) AFTER fitting, as interpretive
    # context. notebooks/environmental_data/multicollinearity_screen_av1.ipynb replaces that with
    # a mechanical, pre-registered screen (deterministic-duplicate removal, then near-exact
    # empirical duplicates at |rho|>=0.95 -- see models/xgb_environmental/
    # multicollinearity_screen.py), reviewed and revised by an llm-council pass before being
    # coded (documentation/experiment_log.md's 2026-08-06 entry has the full review). These four
    # are cumulative and staged, matching how the evidence is actually built up: terrain only,
    # then +wind, then +whichever other categories (climate/soil/edge) show real individual
    # signal, then +every environmental category unfiltered as a "for the sake of proof" check.
    # Deliberately NO variance-discarding VIF step here (unlike the notebook's OTHER output, the
    # "attribution-safe" tiers meant for xgb_environmental's SHAP/permutation tooling) -- per the
    # council review, a network being FED redundant raw inputs isn't hurt the way a linear
    # coefficient or a SHAP value is, so removing correlated-but-real variance from what the PINN
    # sees would cost real information for no established benefit.
    "stage1_terrain": [
        "elevation", "slope_degrees", "northness", "eastness", "profile_curvature",
        "plan_curvature", "tpi", "elevation_roughness", "solar_radiation_index",
        "frost_hollow_flag", "ceh_twi", "tpi_500m", "local_relief_500m",
    ],
    "stage2_terrain_wind": [
        "elevation", "slope_degrees", "northness", "eastness", "profile_curvature",
        "plan_curvature", "tpi", "elevation_roughness", "solar_radiation_index",
        "frost_hollow_flag", "ceh_twi", "tpi_500m", "local_relief_500m",
        "gwa_wind_speed_10m", "topex", "windward_topex", "whcl",
        "gwa_weibull_a_10m", "gwa_weibull_k_10m", "gwa_weibull_a_50m", "gwa_weibull_k_50m",
    ],
    # stage3 and stage4 happened to come out identical this run -- every remaining category
    # (climate/soil/edge) cleared the notebook's signal-strength bar, so "the ones deemed
    # useful" and "all of them" are the same set right now. Kept as two separately named tiers
    # anyway: the filter is a real check every time this notebook is re-run (e.g. after a new
    # candidate variable is added), not something guaranteed to always produce two different lists.
    # tas_mean/groundfrost_mean deliberately excluded from stage3/stage4 below -- same
    # cohort-suffix resolution gap that already excludes them from broad_legitimate above
    # (load_split_table_with_terrain() doesn't resolve tas_mean_4survey/_6survey the way
    # xgb_environmental's own pipeline does; a plain "tas_mean" here would KeyError at run
    # time). The notebook's OTHER output, the attribution-safe tiers (for xgb_environmental,
    # which does resolve the cohort suffix), keeps both columns -- see the notebook itself.
    "stage3_terrain_wind_plus": [
        "elevation", "slope_degrees", "northness", "eastness", "profile_curvature",
        "plan_curvature", "tpi", "elevation_roughness", "solar_radiation_index",
        "frost_hollow_flag", "ceh_twi", "tpi_500m", "local_relief_500m",
        "gwa_wind_speed_10m", "topex", "windward_topex", "whcl",
        "gwa_weibull_a_10m", "gwa_weibull_k_10m", "gwa_weibull_a_50m", "gwa_weibull_k_50m",
        "chelsa_gdd5_degc", "chelsa_bio12_precip_mm",
        "soilgrids_ph", "dist_to_watercourse", "dist_to_cpmt_boundary",
        "dist_to_forest_perimeter", "dist_to_scpt_boundary", "dist_to_block_boundary",
        "cpmt_compactness_ratio", "dist_to_road",
    ],
    "stage4_all_environmental": [
        "elevation", "slope_degrees", "northness", "eastness", "profile_curvature",
        "plan_curvature", "tpi", "elevation_roughness", "solar_radiation_index",
        "frost_hollow_flag", "ceh_twi", "tpi_500m", "local_relief_500m",
        "gwa_wind_speed_10m", "topex", "windward_topex", "whcl",
        "gwa_weibull_a_10m", "gwa_weibull_k_10m", "gwa_weibull_a_50m", "gwa_weibull_k_50m",
        "chelsa_gdd5_degc", "chelsa_bio12_precip_mm",
        "soilgrids_ph", "dist_to_watercourse", "dist_to_cpmt_boundary",
        "dist_to_forest_perimeter", "dist_to_scpt_boundary", "dist_to_block_boundary",
        "cpmt_compactness_ratio", "dist_to_road",
    ],
}
DEFAULT_ENV_TERRAIN_FEATURE_SET = "terrain_wind_solid"


def assert_env_terrain_features_disjoint_from_noenv(feature_columns):
    # Guards against the exact mistake almost made while writing ENV_TERRAIN_FEATURE_SETS above:
    # a column already in the no-env main-network input (NUMERIC_SCALED_COLUMNS/
    # BINARY_PASSTHROUGH_COLUMNS/"Age") accidentally also listed as a terrain/wind feature would
    # feed the same information into the model twice, through two different pathways, rather
    # than adding anything new. Same "catch a silent feature-overlap mistake at import/call
    # time" spirit as assert_no_split_columns_in_features() in models/common/splits.py.
    noenv_columns = set(NUMERIC_SCALED_COLUMNS) | set(BINARY_PASSTHROUGH_COLUMNS) | {"Age", "thinning_status"}
    overlap = noenv_columns & set(feature_columns)
    if overlap:
        raise ValueError(
            f"These columns are in BOTH the no-env feature set and this terrain/wind feature "
            f"set: {sorted(overlap)} -- would feed the same information to the model twice, "
            "through two different pathways. Remove them from one side or the other."
        )

ENVIRONMENTAL_FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "environmental" / "plot_environmental_features.parquet"


def select_device():
    # Prefer a real GPU if one is available -- CUDA on the SLURM cluster
    # this will eventually train on, MPS (Apple Silicon) for a quick local
    # test on this Mac -- and fall back to plain CPU otherwise. The rest of
    # this codebase never has to know or care which one it got.
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split_table(cohort, split_type, split_seed=SPLIT_SEED, k_folds=DEFAULT_K_FOLDS, held_out_fold=0):
    # split_seed=SPLIT_SEED (2026-08-02 addition) keeps every existing call exactly as before --
    # only spatial_block_split's own block-shuffle seed changes when a caller explicitly passes a
    # different value (see run_dnn_noenv.py/run_dnn_env_terrain.py's --split-seed), for the
    # spatial_block_split robustness check documentation/experiment_log.md's 2026-08-02 entries
    # flag as still open: every result up to this point used the SAME fixed partition of
    # compartments into train/val/test, never varied. temporal_split()/temporal_narrow_gap don't
    # take a seed at all (year assignment is fixed, not randomised), so split_seed only ever
    # affects the spatial_block branch below.
    #
    # Both dnn_noenv and pinn_noenv read the same consolidated model_table.parquet (2026-07-28 --
    # there is no separate per-model file anymore, see data_processing/export_model_tables.py).
    # Applies the same maturity + yield-class filter every other baseline uses, then labels every
    # row train/val/test using either temporal_split() or spatial_block_split() -- same two split
    # functions and shared constants (models/common/splits.py) that
    # models/baselines/run_baselines.py uses, so a DNN/PINN run and a baseline run under the same
    # split_type see identical train/val/test membership.
    table = load_model_table(cohort)
    filtered_table = filter_data(table)
    filtered_table = filtered_table.copy()

    if split_type == "temporal":
        filtered_table["split"] = temporal_split(
            filtered_table,
            year_col="LiDAR_year",
            **TEMPORAL_YEARS[cohort],
        )
    elif split_type == "temporal_narrow_gap":
        # Same temporal_split() function, different year assignment -- see
        # TEMPORAL_YEARS_NARROW_GAP in models/common/splits.py for why this
        # is a separate dict, not a variant of TEMPORAL_YEARS.
        filtered_table["split"] = temporal_split(
            filtered_table,
            year_col="LiDAR_year",
            **TEMPORAL_YEARS_NARROW_GAP[cohort],
        )
    elif split_type == "spatial_block":
        # coordinates_df=None makes spatial_block_split() load plot
        # centroids itself. Whole compartments go to train/val/test
        # together, so (unlike temporal_split) every survey year of a
        # train-plot is "train" -- see load_trajectory_pairs() below for why
        # that matters for the PINN specifically.
        filtered_table["split"] = spatial_block_split(
            filtered_table,
            block_col=SPATIAL_BLOCK_COL,
            buffer_distance=SPATIAL_BUFFER_METRES,
            seed=split_seed,
        )
    elif split_type == "spatial_block_kfold":
        # A single spatial_block_split() only ever evaluates on one arbitrary ~20% slice of
        # compartments -- k_folds/held_out_fold rotate which slice is held out, so a full sweep
        # (held_out_fold=0..k_folds-1) evaluates the WHOLE population instead. split_seed controls
        # the same underlying compartment shuffle spatial_block_split uses (assign_spatial_folds
        # inside spatial_kfold_split), so held_out_fold=0 with k_folds=5 is NOT the same partition
        # as plain "spatial_block" above (2 buckets vs. 5), even at the same seed -- this is a
        # genuinely different split_type, not an alias.
        filtered_table["split"] = spatial_kfold_split(
            filtered_table,
            block_col=SPATIAL_BLOCK_COL,
            k=k_folds,
            held_out_fold=held_out_fold,
            buffer_distance=SPATIAL_BUFFER_METRES,
            seed=split_seed,
        )
    else:
        raise ValueError(f"Unknown split_type: {split_type!r}")

    return filtered_table


def load_split_table_with_terrain(
    cohort, split_type, feature_columns, split_seed=SPLIT_SEED, k_folds=DEFAULT_K_FOLDS, held_out_fold=0,
):
    # split_seed/k_folds/held_out_fold passed straight through to load_split_table() -- see that
    # function's own note.
    #
    # Same as load_split_table() above, plus the chosen terrain/wind feature_columns merged in
    # from the environmental feature export (built by aux_data_resolution_check.ipynb, the same
    # file models/xgb_environmental/data.py reads). feature_columns is a real list, not looked up
    # by name here -- callers pass ENV_TERRAIN_FEATURE_SETS[name] themselves (see
    # run_dnn_env_terrain.py/run_pinn_env_terrain.py's --feature-set), same "take a raw column
    # list, not a name" convention xgb_environmental.py's fit_with_columns() already uses. Kept
    # as a SEPARATE function rather than changing load_split_table() itself, so dnn_noenv/
    # pinn_noenv keep working exactly as before -- they never call this one.
    assert_env_terrain_features_disjoint_from_noenv(feature_columns)

    split_df = load_split_table(cohort, split_type, split_seed=split_seed, k_folds=k_folds, held_out_fold=held_out_fold)
    environmental_features = pd.read_parquet(ENVIRONMENTAL_FEATURES_PATH)

    # BUG FIX (2026-08-01): model_table.parquet already carries its own "whcl" column (used
    # elsewhere in the pipeline, not by build_tensors() below), which is a DIFFERENT column from
    # environmental_features' "whcl" -- merging without handling this collision doesn't error at
    # the merge itself, it silently renames both to whcl_x/whcl_y, so the failure only surfaced
    # later (a confusing KeyError at split_df[feature_columns], or in run_*_env_terrain.py's own
    # crash, far from the real cause) whenever "whcl" was requested (terrain_wind_extended/broad
    # -- terrain_wind_solid never hit this, none of its 5 columns collide). Dropping any
    # requested column that already exists in split_df BEFORE merging guarantees every name in
    # feature_columns ends up clean and unsuffixed afterward, regardless of which specific column
    # happens to collide -- not just a one-off fix for "whcl".
    columns_already_present = [c for c in feature_columns if c in split_df.columns]
    if columns_already_present:
        split_df = split_df.drop(columns=columns_already_present)

    before_rows = len(split_df)
    split_df = split_df.merge(
        environmental_features[["identification"] + feature_columns],
        on="identification", how="left",
    )

    # ceh_twi is genuinely missing for 39 of 71,766 plots (confirmed directly against
    # plot_environmental_features.parquet, 2026-08-01) -- not a merge bug, so this is a real
    # gap to filter out, not something to hard-error on. Whole-plot dropped (every survey year
    # of an affected plot, not just the row missing the value), same convention as
    # models/common/data.py::filter_data()'s age/yldc filters, so a plot never contributes a
    # partial trajectory to load_trajectory_pairs() downstream.
    rows_with_missing_terrain = split_df[feature_columns].isna().any(axis=1)
    missing_plot_ids = set(split_df.loc[rows_with_missing_terrain, "identification"])
    if missing_plot_ids:
        removed_rows = split_df["identification"].isin(missing_plot_ids).sum()
        print(
            f"  WARNING: {len(missing_plot_ids)} plot(s) have no value for one of {feature_columns} "
            f"(whole plot dropped): {removed_rows} of {before_rows} rows"
        )
        split_df = split_df[~split_df["identification"].isin(missing_plot_ids)].copy()

    return split_df


def fit_terrain_scaler(train_df, feature_columns):
    # Its own StandardScaler, separate from scaler_other_features -- the y_max sub-network reads
    # ONLY these columns, never mixed into the main network's "other features" tensor (see
    # models/pinn_env_terrain/pinn_env_terrain.py's own top-of-file note for why terrain/wind
    # stays out of the main network entirely, feeding the y_max sub-network only).
    return StandardScaler().fit(train_df[feature_columns])


def build_terrain_tensor(df, scaler_terrain, feature_columns, device):
    terrain_scaled = scaler_terrain.transform(df[feature_columns])
    return torch.tensor(terrain_scaled, dtype=torch.float32, device=device)


def fill_missing_time_since_thinning(df):
    # time_since_thinning is NaN for never-thinned plots (matching
    # time_since_thinning_missing=True for those same rows). Fill with 0 so
    # it can go into a tensor -- the missingness flag is what actually tells
    # the model "this is a placeholder, not a real elapsed time", same
    # convention as models/rf_baseline/rf_baseline.py::prepare_features().
    df = df.copy()
    df["time_since_thinning"] = df["time_since_thinning"].fillna(0)
    return df


def encode_thinning_status(df, encoded_column_names=None):
    # Same pattern as models/linear_baseline/linear_baseline.py::
    # encode_features(): one-hot encode thinning_status, then (if given)
    # reindex onto the exact columns seen during training. A category seen
    # only in training becomes a column of 0s here at evaluation time; a
    # category never seen in training is dropped, so the model never sees
    # an unexpected column.
    encoded = pd.get_dummies(df[["thinning_status"]], columns=["thinning_status"])
    if encoded_column_names is not None:
        encoded = encoded.reindex(columns=encoded_column_names, fill_value=0)
    return encoded


def fit_scalers(train_df):
    # Three SEPARATE StandardScaler instances, fit on the training split
    # only, matching Lynch (2025). Age and the target each get their own
    # scaler (not folded into a joint one) so their individual training-split
    # standard deviations can be read back out for the physics loss's
    # chain-rule correction -- see documentation/model_instructions/
    # age_only_dnn_pinn_instructions.md, section 3.
    scaler_age = StandardScaler().fit(train_df[["Age"]])
    scaler_other_features = StandardScaler().fit(train_df[NUMERIC_SCALED_COLUMNS])
    scaler_height = StandardScaler().fit(train_df[[TARGET_COLUMN]])
    return scaler_age, scaler_other_features, scaler_height


def build_tensors(
    df,
    scaler_age,
    scaler_other_features,
    scaler_height,
    encoded_column_names,
    device,
    include_target=True,
):
    # Turns one row-per-observation dataframe into the three tensors a
    # forward pass needs: age (its own tensor, scaled), other_features (every
    # other no-environment feature, scaled/encoded and concatenated into one
    # tensor), and target (scaled target, or None if include_target is
    # False -- used when building trajectory-pair endpoints, which predict
    # their own height rather than being compared to an observed one).
    df = fill_missing_time_since_thinning(df)

    age_scaled = scaler_age.transform(df[["Age"]])
    other_numeric_scaled = scaler_other_features.transform(df[NUMERIC_SCALED_COLUMNS])
    binary_values = df[BINARY_PASSTHROUGH_COLUMNS].astype(float).values

    thinning_status_onehot = encode_thinning_status(df, encoded_column_names=encoded_column_names)
    onehot_values = thinning_status_onehot.astype(float).values

    other_features = np.concatenate([other_numeric_scaled, binary_values, onehot_values], axis=1)

    age_tensor = torch.tensor(age_scaled, dtype=torch.float32, device=device)
    other_features_tensor = torch.tensor(other_features, dtype=torch.float32, device=device)

    target_tensor = None
    if include_target:
        target_scaled = scaler_height.transform(df[[TARGET_COLUMN]])
        target_tensor = torch.tensor(target_scaled, dtype=torch.float32, device=device)

    return age_tensor, other_features_tensor, target_tensor


# Column names as they appear on the LATER endpoint of a transition pair
# (the plain, unprefixed no-environment feature columns).
_LATER_ENDPOINT_COLUMNS = [
    "Age", "CanopyCover", "Thin", "time_since_thinning",
    "time_since_thinning_missing", "recent_thinning_5yr", "thinning_status",
    # yldc removed 2026-07-28, see NOENV_FEATURE_COLUMNS's own comment above.
]

# Same features, but as they appear on the EARLIER endpoint -- the
# previous_* columns build_transition_table() now provides for all of them
# (see data_processing/export_model_tables.py).
_EARLIER_ENDPOINT_COLUMNS = {
    "previous_age": "Age",
    "previous_canopy_cover": "CanopyCover",
    "previous_Thin": "Thin",
    "previous_time_since_thinning": "time_since_thinning",
    "previous_time_since_thinning_missing": "time_since_thinning_missing",
    "previous_recent_thinning_5yr": "recent_thinning_5yr",
    "previous_thinning_status": "thinning_status",
}


def load_trajectory_pairs(cohort, split_df):
    # Loads the transition (growth-between-surveys) table and keeps only
    # pairs usable for the trajectory loss: BOTH endpoints must themselves
    # be labelled "train" in split_df (the same table load_split_table()
    # just built). This one rule is correct under either split type,
    # without needing to branch on which one is active:
    #   - under temporal_split, a plot's rows are split by YEAR, so this
    #     keeps only pairs whose two years are both training years --
    #     exactly what the old train_years-based filter did.
    #   - under spatial_block_split, a whole plot goes to train/val/test
    #     together (every survey year gets the same split label), so this
    #     keeps only pairs whose PLOT itself was assigned to train -- any
    #     two of its survey years qualify.
    # Never matches a val/test row either way, so validation/test
    # information can never leak into the trajectory loss (see
    # documentation/model_instructions/age_only_dnn_pinn_instructions.md,
    # section 1's leakage rule). split_df already reflects filter_data(), so
    # a plot that didn't survive that filter has no "train" rows here and
    # is automatically excluded too -- no separate eligible-plots check
    # needed.
    transitions_path = PROJECT_ROOT / "data" / "processed" / "transitions" / f"transition_growth_{cohort}.parquet"
    pairs = pd.read_parquet(transitions_path).reset_index(drop=True)

    train_keys = set(zip(
        split_df.loc[split_df["split"] == "train", "identification"],
        split_df.loc[split_df["split"] == "train", "LiDAR_year"],
    ))
    earlier_is_train = pd.Series(zip(pairs["identification"], pairs["previous_lidar_year"])).isin(train_keys)
    later_is_train = pd.Series(zip(pairs["identification"], pairs["LiDAR_year"])).isin(train_keys)

    return pairs[(earlier_is_train & later_is_train).values].reset_index(drop=True)


def build_pair_tensors(pairs_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device):
    # Builds tensors for BOTH endpoints of every trajectory pair, plus
    # delta_age, age_mid, and the observed growth rate (annual_height_increment)
    # the trajectory loss needs. Neither endpoint's tensors include a target
    # -- the trajectory loss compares the network's OWN two predictions to
    # each other, never to the observed heights directly.
    later_df = pairs_df[_LATER_ENDPOINT_COLUMNS].copy()
    earlier_df = pairs_df[list(_EARLIER_ENDPOINT_COLUMNS.keys())].rename(columns=_EARLIER_ENDPOINT_COLUMNS)

    age_earlier, other_earlier, _ = build_tensors(
        earlier_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device,
        include_target=False,
    )
    age_later, other_later, _ = build_tensors(
        later_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device,
        include_target=False,
    )

    delta_age = pairs_df["Age"].values - pairs_df["previous_age"].values
    age_mid = (pairs_df["Age"].values + pairs_df["previous_age"].values) / 2

    delta_age_tensor = torch.tensor(delta_age.reshape(-1, 1), dtype=torch.float32, device=device)
    age_mid_tensor = torch.tensor(age_mid.reshape(-1, 1), dtype=torch.float32, device=device)
    observed_growth_rate_tensor = torch.tensor(
        pairs_df["annual_height_increment"].values.reshape(-1, 1), dtype=torch.float32, device=device,
    )

    return age_earlier, other_earlier, age_later, other_later, delta_age_tensor, age_mid_tensor, observed_growth_rate_tensor


def build_pair_terrain_tensor(pairs_df, scaler_terrain, feature_columns, device):
    # Terrain/wind is a STATIC, per-plot property (doesn't vary by survey year), so a trajectory
    # pair's earlier and later endpoint always share the exact same terrain values -- one tensor
    # per pair is enough, not a separate "earlier"/"later" pair the way age/no-env features need
    # (those genuinely differ between the two endpoints).
    environmental_features = pd.read_parquet(ENVIRONMENTAL_FEATURES_PATH)
    pairs_with_terrain = pairs_df[["identification"]].merge(
        environmental_features[["identification"] + feature_columns],
        on="identification", how="left",
    )
    missing_terrain = pairs_with_terrain[feature_columns].isna().any(axis=1).sum()
    if missing_terrain > 0:
        raise ValueError(
            f"{missing_terrain} of {len(pairs_df)} trajectory pairs have no matching terrain/wind "
            "row -- same identification-based merge as load_split_table_with_terrain(), should "
            "never happen for a real plot in this dataset."
        )
    return build_terrain_tensor(pairs_with_terrain, scaler_terrain, feature_columns, device)


MIN_TRAJECTORY_PAIR_COVERAGE_FRACTION = 0.30


def print_pre_training_diagnostic(cohort, split_df, pairs_df):
    # Printed before any training happens, for both the DNN and the PINN --
    # see documentation/model_instructions/age_only_dnn_pinn_instructions.md,
    # section 8. The PINN is the only one that actually trains on pairs_df,
    # but this is about understanding the shared underlying data, so it is
    # printed for both models.
    train_df = split_df[split_df["split"] == "train"]
    test_df = split_df[split_df["split"] == "test"]

    n_train_plots = train_df["identification"].nunique()
    n_test_plots = test_df["identification"].nunique()
    n_plots_with_a_pair = pairs_df["identification"].nunique()

    print(f"  ----- Pre-training diagnostic: {cohort} -----")
    print(f"  Training plots (unique identification): {n_train_plots:,}")
    print(f"  Test plots (unique identification): {n_test_plots:,}")
    print(f"  Training plots with at least one usable trajectory pair: {n_plots_with_a_pair:,}")

    if len(pairs_df) > 0:
        growth_rate_mean = pairs_df["annual_height_increment"].mean()
        growth_rate_std = pairs_df["annual_height_increment"].std()
        print(f"  annual_height_increment across training pairs: mean={growth_rate_mean:.4f} m/yr, std={growth_rate_std:.4f} m/yr")

    print("  Training rows by survey year:")
    for year, row_count in train_df["LiDAR_year"].value_counts().sort_index().items():
        print(f"    {year}: {row_count:,} rows")

    coverage_fraction = n_plots_with_a_pair / n_train_plots if n_train_plots > 0 else 0.0
    if coverage_fraction < MIN_TRAJECTORY_PAIR_COVERAGE_FRACTION:
        print(
            f"  WARNING: only {coverage_fraction:.1%} of training plots have a usable trajectory "
            f"pair (below the {MIN_TRAJECTORY_PAIR_COVERAGE_FRACTION:.0%} expected minimum) -- "
            "the trajectory loss will have very little signal. Check the maturity filter and "
            "pairing logic before trusting PINN results."
        )
    print()

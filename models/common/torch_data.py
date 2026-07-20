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
    SPATIAL_BLOCK_COL,
    SPATIAL_BUFFER_METRES,
    SPLIT_SEED,
    TEMPORAL_YEARS,
    TEMPORAL_YEARS_NARROW_GAP,
    spatial_block_split,
    temporal_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The full no-environment feature set every DNN/PINN table already has.
NOENV_FEATURE_COLUMNS = [
    "Age",
    "CanopyCover",
    "Thin",
    "time_since_thinning",
    "time_since_thinning_missing",
    "recent_thinning_5yr",
    "thinning_status",
    "yldc",
]

# Truly continuous numeric features -- these get standard-scaled as one
# group. Age is scaled SEPARATELY (its own scaler, see fit_scalers below) so
# its training-split standard deviation can be read back out cleanly for the
# physics loss's chain-rule correction.
NUMERIC_SCALED_COLUMNS = ["CanopyCover", "time_since_thinning", "yldc"]

# Already 0/1 binary flags -- passed through unscaled, same as the one-hot
# thinning_status columns below.
BINARY_PASSTHROUGH_COLUMNS = ["Thin", "time_since_thinning_missing", "recent_thinning_5yr"]

TARGET_COLUMN = "Top_Height99"


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


def load_split_table(cohort, table_name, split_type):
    # table_name is "dnn_noenv" or "pinn_noenv". Applies the same maturity +
    # yield-class filter every other baseline uses, then labels every row
    # train/val/test using either temporal_split() or spatial_block_split()
    # -- same two split functions and shared constants (models/common/
    # splits.py) that models/baselines/run_baselines.py uses, so a DNN/PINN
    # run and a baseline run under the same split_type see identical
    # train/val/test membership.
    table = load_model_table(cohort, table_name)
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
            seed=SPLIT_SEED,
        )
    else:
        raise ValueError(f"Unknown split_type: {split_type!r}")

    return filtered_table


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
    # only, matching Lynch (2025). Age and Top_Height99 each get their own
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
    # tensor), and target (scaled Top_Height99, or None if include_target is
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
    "time_since_thinning_missing", "recent_thinning_5yr", "thinning_status", "yldc",
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
    "previous_yldc": "yldc",
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
    # delta_age, age_mid, and the observed growth rate (annual_height99_increment)
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
        pairs_df["annual_height99_increment"].values.reshape(-1, 1), dtype=torch.float32, device=device,
    )

    return age_earlier, other_earlier, age_later, other_later, delta_age_tensor, age_mid_tensor, observed_growth_rate_tensor


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
        growth_rate_mean = pairs_df["annual_height99_increment"].mean()
        growth_rate_std = pairs_df["annual_height99_increment"].std()
        print(f"  annual_height99_increment across training pairs: mean={growth_rate_mean:.4f} m/yr, std={growth_rate_std:.4f} m/yr")

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

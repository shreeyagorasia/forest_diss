# Run as: python -m models.common.test_splits
#
# Exercises spatial_block_split() and temporal_split() against the real
# 4survey/6survey data -- both are now used by the baselines and by
# dnn_noenv/pinn_noenv (see models/common/splits.py), so this is a
# regression check for that shared code, not a check on unused code.

from pathlib import Path

import pandas as pd

from models.common.geo import find_train_plots_near_holdout, load_plot_coordinates
from models.common.splits import TEMPORAL_YEARS, TEMPORAL_YEARS_NARROW_GAP, spatial_block_split, temporal_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COHORTS = ["4survey", "6survey"]

# cpmt (forestry compartment) is used as the spatial block, not blk. blk only
# has 8 values with very uneven sizes (see the comment in splits.py), which
# is too coarse to split into sensible train/val/test fractions.
SPATIAL_BLOCK_COLUMN = "cpmt"

# Most plots are 20m grid cells, so 60m is a clean 3-cell-width buffer
# (50m was an awkward 2.5x). A 100m buffer was tried first but excluded
# 2-3x more plots for the same leakage protection at close range (see the
# notebook comparison in data_exploration_gpkg).
TEST_BUFFER_DISTANCE_METRES = 60

# If a split ends up with fewer rows or blocks than this share of the total,
# print a warning so it does not go unnoticed.
SMALL_SPLIT_WARNING_FRACTION = 0.05


def load_master(cohort):
    path = PROJECT_ROOT / "data" / "processed" / "master" / f"clean_master_{cohort}.parquet"
    return pd.read_parquet(path)


def check_no_group_in_two_splits(df, group_col, split_labels):
    # For every value of group_col (a block id, for example), every row with
    # that value should have been given the same split label. This checks
    # that directly rather than assuming the split function got it right.
    check_df = pd.DataFrame({group_col: df[group_col].values, "split": split_labels})
    splits_per_group = check_df.groupby(group_col)["split"].nunique()
    groups_in_more_than_one_split = splits_per_group[splits_per_group > 1]
    return groups_in_more_than_one_split


def test_spatial_block_split(cohort):
    print(f"----- spatial_block_split: {cohort} (block_col='{SPATIAL_BLOCK_COLUMN}') -----")
    df = load_master(cohort)

    split_labels = spatial_block_split(df, block_col=SPATIAL_BLOCK_COLUMN, seed=42)
    df = df.copy()
    df["split"] = split_labels

    total_rows = len(df)
    for split_name in ["train", "val", "test"]:
        split_rows = df[df["split"] == split_name]
        n_rows = len(split_rows)
        n_blocks = split_rows[SPATIAL_BLOCK_COLUMN].nunique()
        n_plots = split_rows["identification"].nunique()
        row_share = n_rows / total_rows
        print(f"  {split_name:<6} rows={n_rows:>7,} ({row_share:>5.1%})   blocks={n_blocks:>4}   plots={n_plots:>6,}")

        if row_share < SMALL_SPLIT_WARNING_FRACTION:
            print(f"  WARNING: '{split_name}' split has only {row_share:.1%} of rows — check block sizes for '{SPATIAL_BLOCK_COLUMN}'.")
        if n_blocks == 0:
            print(f"  WARNING: '{split_name}' split has zero blocks.")

    leaked_blocks = check_no_group_in_two_splits(df, SPATIAL_BLOCK_COLUMN, split_labels)
    if len(leaked_blocks) == 0:
        print(f"  OK: no '{SPATIAL_BLOCK_COLUMN}' value appears in more than one split.")
    else:
        print(f"  FAIL: these '{SPATIAL_BLOCK_COLUMN}' values appear in more than one split:")
        print(leaked_blocks)

    print()


def test_spatial_block_split_with_buffer(cohort, coordinates_df):
    print(f"----- spatial_block_split with buffer_distance={TEST_BUFFER_DISTANCE_METRES}m: {cohort} -----")
    df = load_master(cohort)

    split_labels = spatial_block_split(
        df,
        block_col=SPATIAL_BLOCK_COLUMN,
        buffer_distance=TEST_BUFFER_DISTANCE_METRES,
        coordinates_df=coordinates_df,
        seed=42,
    )
    df = df.copy()
    df["split"] = split_labels

    total_rows = len(df)
    for split_name in ["train", "val", "test", "buffer"]:
        split_rows = df[df["split"] == split_name]
        n_rows = len(split_rows)
        row_share = n_rows / total_rows
        print(f"  {split_name:<6} rows={n_rows:>7,} ({row_share:>5.1%})")

    # Re-run the same nearest-neighbour check the buffer used, but on the
    # FINAL labels (with "buffer" plots already excluded). If the buffer
    # worked, no remaining TRAIN plot should be within range of a val/test
    # plot. val/test are never touched by buffering (see splits.py), so
    # they are not re-checked against each other here — two held-out plots
    # sitting near each other is not a leak.
    plot_to_split = dict(zip(df["identification"], split_labels))
    plot_to_split = {plot_id: split for plot_id, split in plot_to_split.items() if split != "buffer"}
    still_too_close = find_train_plots_near_holdout(coordinates_df, plot_to_split, TEST_BUFFER_DISTANCE_METRES)

    if len(still_too_close) == 0:
        print(f"  OK: after buffering, no remaining train plot is within {TEST_BUFFER_DISTANCE_METRES}m of a val/test plot.")
    else:
        print(f"  FAIL: {len(still_too_close):,} train plots are still within {TEST_BUFFER_DISTANCE_METRES}m of a val/test plot after buffering.")

    print()


def test_temporal_split(cohort, years=None, label="temporal_split"):
    years = years if years is not None else TEMPORAL_YEARS[cohort]
    print(f"----- {label}: {cohort} (train={years['train_years']}, val={years['val_years']}, test={years['test_years']}) -----")
    df = load_master(cohort)

    split_labels = temporal_split(df, year_col="LiDAR_year", **years)
    df = df.copy()
    df["split"] = split_labels

    total_rows = len(df)
    for split_name in ["train", "val", "test", "unassigned"]:
        split_rows = df[df["split"] == split_name]
        n_rows = len(split_rows)
        if n_rows == 0 and split_name == "unassigned":
            continue
        years_in_split = sorted(split_rows["LiDAR_year"].unique())
        n_plots = split_rows["identification"].nunique()
        row_share = n_rows / total_rows
        print(f"  {split_name:<11} rows={n_rows:>7,} ({row_share:>5.1%})   years={years_in_split}   plots={n_plots:>6,}")

        if split_name != "unassigned" and row_share < SMALL_SPLIT_WARNING_FRACTION:
            print(f"  WARNING: '{split_name}' split has only {row_share:.1%} of rows.")

    # Unlike the plot-level and spatial splits, plots are expected to repeat
    # across train/val/test here — this just measures how much overlap there is.
    plots_by_split = {
        split_name: set(df.loc[df["split"] == split_name, "identification"])
        for split_name in ["train", "val", "test"]
    }
    plots_in_train_and_test = plots_by_split["train"] & plots_by_split["test"]
    print(f"  Plots appearing in both train and test (expected, not a leak for this split type): {len(plots_in_train_and_test):,}")

    print()


def main():
    coordinates_df = load_plot_coordinates()

    for cohort in COHORTS:
        test_spatial_block_split(cohort)
        test_spatial_block_split_with_buffer(cohort, coordinates_df)
        test_temporal_split(cohort)
        test_temporal_split(cohort, years=TEMPORAL_YEARS_NARROW_GAP[cohort], label="temporal_split (narrow gap)")


if __name__ == "__main__":
    main()

import numpy as np


def plot_level_split(df, plot_col="identification", test_size=0.2, val_size=0.2, seed=42):
    # Splitting by plot (not by row) means every survey year for one plot ends
    # up in the same split. This stops the same plot leaking into both
    # training and test data.
    #
    # Default is a 60/20/20 train/val/test split. This three-way shape is
    # used by every split function in this file (plot_level_split,
    # spatial_block_split, temporal_split) so predictions.csv from every
    # model uses the same "train"/"val"/"test" labels, and a comparison
    # notebook does not need special-casing per model.

    # get the list of unique plot ids and shuffle them into a random order
    plot_ids = df[plot_col].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(plot_ids)

    # work out how many plots belong in the test and validation groups
    total_plots = len(plot_ids)
    n_test = int(total_plots * test_size)
    n_val = int(total_plots * val_size)

    # slice the shuffled list into three separate groups of plot ids
    test_plot_ids = set(plot_ids[:n_test])
    val_plot_ids = set(plot_ids[n_test:n_test + n_val])
    # everything left over goes into the training group

    # label every row depending on which group its plot id belongs to
    split_labels = []
    for plot_id in df[plot_col]:
        if plot_id in test_plot_ids:
            split_labels.append("test")
        elif plot_id in val_plot_ids:
            split_labels.append("val")
        else:
            split_labels.append("train")

    return split_labels


def spatial_block_split(
    df,
    block_col,
    test_blocks_frac=0.2,
    val_blocks_frac=0.2,
    buffer_distance=None,
    coordinates_df=None,
    seed=42,
):
    # Assigns WHOLE spatial blocks (e.g. forestry compartments) to
    # train/val/test, so no block appears in more than one split.
    #
    # This is a different question to plot_level_split: a random or
    # plot-level split can put neighbouring plots (which share near-identical
    # terrain and wind exposure) on both sides of the split, letting a model
    # look like it generalises when it has really just memorised local
    # conditions. Roberts et al. (2017) is the standard reference for why
    # random cross-validation is invalid for spatially autocorrelated data.
    #
    # block_col should be a spatial grouping column, not a plot identifier.
    # In this dataset, 'blk' only has 8 values with very uneven sizes (81 to
    # 29,182 plots), which is too coarse and lopsided to split into
    # reasonable train/val/test fractions. 'cpmt' (forestry compartment) has
    # 296 values and is a genuinely contiguous spatial unit, so it is the
    # right grouping to pass as block_col. No plot ever changes blk or cpmt
    # across survey years, so either choice is a stable per-plot grouping.

    # Compartments vary hugely in size (1 to over 1,300 plots), so picking
    # blocks purely by COUNT can land far from the requested row-count
    # fractions (this happened for the 6survey cohort: only 48 compartments
    # exist there, and a plain count-based split gave a test split with just
    # 11.5% of rows against a 20% target). Instead, shuffle the blocks into a
    # random order, then walk through them one at a time, adding each block
    # to whichever of test/val is currently furthest below its row-count
    # target. This keeps the assignment random while landing much closer to
    # the requested fractions.
    rows_per_block = df.groupby(block_col).size()
    block_ids = rows_per_block.index.to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(block_ids)

    total_rows = rows_per_block.sum()
    test_row_target = total_rows * test_blocks_frac
    val_row_target = total_rows * val_blocks_frac

    test_block_ids = set()
    val_block_ids = set()
    test_rows_so_far = 0
    val_rows_so_far = 0

    for block_id in block_ids:
        block_size = rows_per_block[block_id]
        test_rows_needed = test_row_target - test_rows_so_far
        val_rows_needed = val_row_target - val_rows_so_far

        if test_rows_needed <= 0 and val_rows_needed <= 0:
            break  # both targets already reached; everything left over is train
        elif test_rows_needed >= val_rows_needed:
            test_block_ids.add(block_id)
            test_rows_so_far += block_size
        else:
            val_block_ids.add(block_id)
            val_rows_so_far += block_size
    # everything left over (not added to test_block_ids or val_block_ids)
    # goes into the training group

    split_labels = []
    for block_id in df[block_col]:
        if block_id in test_block_ids:
            split_labels.append("test")
        elif block_id in val_block_ids:
            split_labels.append("val")
        else:
            split_labels.append("train")

    if buffer_distance is not None:
        split_labels = apply_spatial_buffer(df, split_labels, buffer_distance, coordinates_df)

    return split_labels


def apply_spatial_buffer(df, split_labels, buffer_distance, coordinates_df):
    # Only TRAIN plots within buffer_distance of a val/test plot are
    # relabelled "buffer" and excluded. val and test are never touched here,
    # so they always keep every plot the block assignment gave them —
    # leakage only happens when a plot the model TRAINS on sits next to a
    # plot it is EVALUATED on; two held-out plots sitting near each other
    # causes no leakage, since neither one updates the model. This also
    # means a compartment with an oddly-shaped or long shared border with
    # another split only loses plots on its train side, not its held-out
    # side, however large that border is.
    from models.common.geo import find_train_plots_near_holdout, load_plot_coordinates

    if coordinates_df is None:
        coordinates_df = load_plot_coordinates()

    plot_to_split = dict(zip(df["identification"], split_labels))
    train_plots_too_close = find_train_plots_near_holdout(plot_to_split=plot_to_split, coordinates_df=coordinates_df, buffer_distance=buffer_distance)

    print(
        f"  Buffer: {len(train_plots_too_close):,} train plots are within {buffer_distance:g} m of "
        "a val/test plot and are excluded ('buffer'). val/test plots are never removed."
    )

    buffered_labels = []
    for plot_id, label in zip(df["identification"], split_labels):
        if label == "train" and plot_id in train_plots_too_close:
            buffered_labels.append("buffer")
        else:
            buffered_labels.append(label)

    return buffered_labels


def temporal_split(df, year_col, train_years, test_years, val_years=None):
    # Splits purely by survey year, ignoring block and plot identity
    # completely. The same plot can legitimately appear in both train and
    # test here, because this asks a different question to plot_level_split
    # and spatial_block_split: not "does this generalise to new plots or new
    # places", but "does this generalise to a new point in time". Keeping the
    # spatial and temporal questions in separate functions means a failure
    # can be attributed to one or the other, not both at once.

    val_years = val_years if val_years is not None else []

    years_present = set(df[year_col].unique())
    years_assigned = set(train_years) | set(test_years) | set(val_years)
    years_not_assigned = years_present - years_assigned
    if years_not_assigned:
        print(
            f"  NOTE: years {sorted(years_not_assigned)} are present in the data but not "
            "listed in train_years/val_years/test_years — those rows are labelled "
            "'unassigned'."
        )

    split_labels = []
    for year in df[year_col]:
        if year in test_years:
            split_labels.append("test")
        elif year in val_years:
            split_labels.append("val")
        elif year in train_years:
            split_labels.append("train")
        else:
            split_labels.append("unassigned")

    return split_labels

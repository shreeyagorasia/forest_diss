from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COORDINATES_PATH = PROJECT_ROOT / "data" / "interim" / "plot_coordinates.csv.gz"


def load_plot_coordinates():
    # One row per plot: identification, x, y (metres, EPSG:27700).
    # Built once by models/common/export_coordinates.py.
    return pd.read_csv(COORDINATES_PATH)


def find_plots_near_split_boundary(coordinates_df, plot_to_split, buffer_distance):
    # For every plot, find the distance to its nearest neighbouring plot
    # that landed in a DIFFERENT split. If that distance is under
    # buffer_distance, the plot sits right next to a split boundary, so its
    # measurements could leak spatial information across the split (two
    # neighbouring grid cells share almost identical terrain and wind
    # exposure). Returns the set of plot ids that are too close to a
    # different split and should be excluded.

    coordinates = coordinates_df.copy()
    coordinates["split"] = coordinates["identification"].map(plot_to_split)
    coordinates = coordinates.dropna(subset=["split"])

    split_names = sorted(coordinates["split"].unique())

    # Build one KDTree of plot coordinates per split, so "nearest plot in a
    # different split" can be looked up quickly instead of comparing every
    # plot to every other plot.
    coordinates_by_split = {}
    trees_by_split = {}
    for split_name in split_names:
        rows_in_split = coordinates[coordinates["split"] == split_name]
        coordinates_by_split[split_name] = rows_in_split
        trees_by_split[split_name] = cKDTree(rows_in_split[["x", "y"]].values)

    plots_near_boundary = set()

    for split_name in split_names:
        rows_in_split = coordinates_by_split[split_name]
        other_split_names = [name for name in split_names if name != split_name]

        # start with "infinitely far away", then shrink it down as we check
        # each other split
        nearest_other_split_distance = np.full(len(rows_in_split), np.inf)
        for other_split_name in other_split_names:
            other_tree = trees_by_split[other_split_name]
            distances, _ = other_tree.query(rows_in_split[["x", "y"]].values)
            nearest_other_split_distance = np.minimum(nearest_other_split_distance, distances)

        too_close = nearest_other_split_distance < buffer_distance
        plot_ids_too_close = rows_in_split.loc[too_close, "identification"]
        plots_near_boundary.update(plot_ids_too_close)

    return plots_near_boundary


def find_train_plots_near_holdout(coordinates_df, plot_to_split, buffer_distance, holdout_splits=("val", "test")):
    # Leakage only happens when a TRAINING plot sits close to a plot the
    # model is evaluated on — a val plot sitting next to a test plot causes
    # no leakage, because neither one ever updates the model. So only train
    # needs thinning near a held-out boundary; val and test are never
    # touched here, which keeps their full row count no matter how oddly
    # shaped or how long a border a compartment happens to share with them.
    # This is standard practice in spatial CV (e.g. buffered spatial CV),
    # and it is the fix for a large/irregular compartment losing a big
    # fraction of its own plots just because of its shape or perimeter —
    # only the train side of that shape gets thinned, not the held-out side.

    coordinates = coordinates_df.copy()
    coordinates["split"] = coordinates["identification"].map(plot_to_split)
    coordinates = coordinates.dropna(subset=["split"])

    train_coordinates = coordinates[coordinates["split"] == "train"]
    holdout_coordinates = coordinates[coordinates["split"].isin(holdout_splits)]

    if len(train_coordinates) == 0 or len(holdout_coordinates) == 0:
        return set()

    holdout_tree = cKDTree(holdout_coordinates[["x", "y"]].values)
    distances, _ = holdout_tree.query(train_coordinates[["x", "y"]].values)

    too_close = distances < buffer_distance
    train_plot_ids_too_close = train_coordinates.loc[too_close, "identification"]

    return set(train_plot_ids_too_close)

from pathlib import Path

import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COORDINATES_PATH = PROJECT_ROOT / "data" / "interim" / "plot_coordinates.csv.gz"
COMPARTMENT_BOUNDARIES_PATH = PROJECT_ROOT / "data" / "interim" / "compartment_boundaries.parquet"


def load_plot_coordinates():
    # One row per plot: identification, x, y (metres, EPSG:27700).
    # Built once by models/common/export_coordinates.py.
    return pd.read_csv(COORDINATES_PATH)


def load_compartment_boundaries():
    # One dissolved boundary polygon per forestry compartment (`cpmt`),
    # metres, EPSG:27700 -- same coordinate system as load_plot_coordinates().
    # Built once by models/common/export_compartment_boundaries.py.
    # Presentation-only (plotting a map background) -- geopandas is imported
    # here, not at module level, so scripts that only need
    # load_plot_coordinates()/find_train_plots_near_holdout() still don't
    # need geopandas installed.
    import geopandas as gpd

    return gpd.read_parquet(COMPARTMENT_BOUNDARIES_PATH)


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

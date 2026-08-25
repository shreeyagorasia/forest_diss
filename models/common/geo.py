from pathlib import Path

import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COORDINATES_PATH = PROJECT_ROOT / "data" / "interim" / "plot_coordinates.csv.gz"
COMPARTMENT_BOUNDARIES_PATH = PROJECT_ROOT / "data" / "interim" / "compartment_boundaries.parquet"
SUBCOMPARTMENT_BOUNDARIES_PATH = PROJECT_ROOT / "data" / "interim" / "subcompartment_boundaries.parquet"
BLOCK_BOUNDARIES_PATH = PROJECT_ROOT / "data" / "interim" / "block_boundaries.parquet"

# Every distance-based statistic in this project (Moran's I, semivariograms, the spatial-block
# split buffer, TOPEX radius) silently depends on this being right. It used to only be a
# comment claim, never checked (2026-07-30 fix). EPSG:27700 = British National Grid, metres.
EXPECTED_CRS_EPSG = 27700

# plot_coordinates.csv.gz is a plain CSV. Once export_coordinates.py converts a GeoDataFrame's
# geometry into bare x/y float columns, the real CRS metadata is gone, so there's nothing to
# assert against here except plausibility. This is a generous bounding box around Aberfoyle
# forest in EPSG:27700 (actual data sits within roughly x=233,000-262,000 / y=692,000-710,000,
# see progress_notes.md). Wide enough to never false-positive on real data, tight enough to
# catch an obviously wrong CRS (e.g. lat/lon degrees, or a different UK grid).
PLAUSIBLE_X_RANGE = (150_000, 350_000)
PLAUSIBLE_Y_RANGE = (600_000, 800_000)


def load_plot_coordinates():
    # One row per plot: identification, x, y (metres, EPSG:27700).
    # Built once by models/common/export_coordinates.py.
    coordinates = pd.read_csv(COORDINATES_PATH)

    x_min, x_max = coordinates["x"].min(), coordinates["x"].max()
    y_min, y_max = coordinates["y"].min(), coordinates["y"].max()
    if not (PLAUSIBLE_X_RANGE[0] <= x_min and x_max <= PLAUSIBLE_X_RANGE[1]
            and PLAUSIBLE_Y_RANGE[0] <= y_min and y_max <= PLAUSIBLE_Y_RANGE[1]):
        raise ValueError(
            f"plot_coordinates.csv.gz's x/y range (x={x_min:.0f}-{x_max:.0f}, "
            f"y={y_min:.0f}-{y_max:.0f}) falls outside the plausible EPSG:27700 envelope for "
            f"Aberfoyle (x={PLAUSIBLE_X_RANGE}, y={PLAUSIBLE_Y_RANGE}). This file has no real "
            "CRS metadata (it's a plain CSV), so this range check is the only thing that would "
            "catch export_coordinates.py ever being regenerated from a differently-projected "
            "source. Every distance-based statistic in this project assumes EPSG:27700."
        )

    return coordinates


def _load_geoparquet_with_crs_check(path, label):
    # Shared by the three boundary loaders below. Unlike load_plot_coordinates() above, these
    # ARE geopandas GeoDataFrames read from GeoParquet, so they carry real CRS metadata that can
    # be checked directly, not just guessed at via a coordinate range.
    import geopandas as gpd

    gdf = gpd.read_parquet(path)
    if gdf.crs is None or gdf.crs.to_epsg() != EXPECTED_CRS_EPSG:
        raise ValueError(
            f"{label} has CRS={gdf.crs!r}, expected EPSG:{EXPECTED_CRS_EPSG}. Every "
            "distance-based statistic in this project assumes this file is in British National "
            "Grid metres."
        )
    return gdf


def load_compartment_boundaries():
    # One dissolved boundary polygon per forestry compartment (`cpmt`),
    # metres, EPSG:27700. Same coordinate system as load_plot_coordinates().
    # Built once by models/common/export_compartment_boundaries.py.
    # Presentation-only (plotting a map background). Geopandas is imported
    # here, not at module level, so scripts that only need
    # load_plot_coordinates()/find_train_plots_near_holdout() still don't
    # need geopandas installed.
    return _load_geoparquet_with_crs_check(COMPARTMENT_BOUNDARIES_PATH, "compartment_boundaries.parquet")


def load_subcompartment_boundaries():
    # Same as load_compartment_boundaries() but one level finer. Dissolved by cpmt+scpt
    # together (scpt labels like "A"/"B"/"C" repeat across different compartments, so scpt alone
    # would wrongly merge unrelated sub-compartments). Built by the same
    # export_compartment_boundaries.py script.
    return _load_geoparquet_with_crs_check(SUBCOMPARTMENT_BOUNDARIES_PATH, "subcompartment_boundaries.parquet")


def load_block_boundaries():
    # Same idea, one level coarser. Dissolved by `blk` (only 8 distinct blocks across the
    # whole forest). Built by the same export_compartment_boundaries.py script.
    return _load_geoparquet_with_crs_check(BLOCK_BOUNDARIES_PATH, "block_boundaries.parquet")


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

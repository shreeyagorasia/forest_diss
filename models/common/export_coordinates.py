# Run as: python -m models.common.export_coordinates
#
# One-off preprocessing step: reads the raw GeoPackage and saves one x/y
# coordinate (plus its forestry compartment, `cpmt`) per plot to a small
# CSV. This is the only place in models/ that needs geopandas — every other
# script just reads the CSV this produces.
#
# Each plot's geometry is a cell from an adaptive 20m/40m grid (see
# data_exploration_gpkg's grid generation notes), and that geometry is the
# same for a plot across every survey year it appears in. Most are regular
# grid cells, but border polygons can be irregular, so use one guaranteed
# interior point per plot rather than assuming every centroid lies inside.

from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPKG_PATH = PROJECT_ROOT / "data" / "raw" / "LiDAR_Years_All_7jul.gpkg"
LAYER_NAME = "LiDAR_Years"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "plot_coordinates.csv.gz"

# Every distance-based statistic downstream (Moran's I, semivariograms, the spatial-block split
# buffer, TOPEX radius) assumes this. This is the ONE place in the whole pipeline where a real
# CRS object still exists (the GeoPackage read below). Once x/y are split into plain float
# columns for the CSV, that metadata is gone for good, so this assertion is the only thing that
# can ever actually catch a wrong CRS, not just print it and hope someone reads the log
# (2026-07-30 fix. This used to only print gdf.crs, never check it).
EXPECTED_CRS_EPSG = 27700


def export_plot_coordinates():
    print(f"Reading {GPKG_PATH} ...")
    gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME, columns=["identification", "cpmt"])
    print(f"  Read {len(gdf):,} rows, CRS = {gdf.crs}")
    if gdf.crs is None or gdf.crs.to_epsg() != EXPECTED_CRS_EPSG:
        raise ValueError(
            f"{GPKG_PATH} has CRS={gdf.crs!r}, expected EPSG:{EXPECTED_CRS_EPSG}. Refusing to "
            "export coordinates in the wrong projection."
        )

    # Keep one row per plot — the geometry does not change across survey years.
    one_row_per_plot = gdf.drop_duplicates(subset="identification").copy()
    print(f"  {len(one_row_per_plot):,} unique plots")

    centroids = one_row_per_plot.geometry.centroid
    sample_points = centroids.where(
        centroids.within(one_row_per_plot.geometry),
        one_row_per_plot.geometry.representative_point(),
    )
    coordinates = one_row_per_plot[["identification", "cpmt"]].copy()
    coordinates["x"] = sample_points.x.values
    coordinates["y"] = sample_points.y.values

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    coordinates.to_csv(OUTPUT_PATH, index=False, compression="gzip")
    print(f"Saved {len(coordinates):,} plot coordinates -> {OUTPUT_PATH}")
    print(f"  CRS: {gdf.crs} (metres) — x range {coordinates['x'].min():.1f}-{coordinates['x'].max():.1f}, "
          f"y range {coordinates['y'].min():.1f}-{coordinates['y'].max():.1f}")


if __name__ == "__main__":
    export_plot_coordinates()

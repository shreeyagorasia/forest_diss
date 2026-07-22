# Run as: python -m models.common.export_compartment_boundaries
#
# One-off preprocessing step: reads the raw GeoPackage and saves one
# dissolved boundary polygon per forestry compartment (`cpmt`) to a small
# GeoParquet file. Presentation-only -- nothing under models/ reads this file
# except plotting code (e.g. results_notebooks/baseline_results.ipynb's
# spatial error maps), so it doesn't belong in export_coordinates.py's
# model-input pipeline.
#
# Each plot's geometry is the same across every survey year it appears in
# (see export_coordinates.py), so this drops to one row per plot BEFORE
# dissolving by compartment -- dissolving all ~795k raw rows directly would
# do the same union many times over for no benefit.
#
# Dissolving unions ~230k individual 20m/40m grid cells, so each compartment
# boundary inherits a "staircase" of tiny right-angle steps along its edge --
# real detail, but far finer than a map spanning the whole ~28km forest can
# ever show (checked: ~26m/pixel at the figure sizes actually used, so
# individual 20m steps are sub-pixel anyway). Simplifying to 30m cuts total
# vertices by ~88% (230k -> ~27k) with no visible change in shape at any
# zoom level tested, from forest-wide down to a 500m crop.

from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPKG_PATH = PROJECT_ROOT / "data" / "raw" / "LiDAR_Years_All_7jul.gpkg"
LAYER_NAME = "LiDAR_Years"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "compartment_boundaries.parquet"
SIMPLIFY_TOLERANCE_METRES = 30


def export_compartment_boundaries():
    print(f"Reading {GPKG_PATH} ...")
    gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME, columns=["identification", "cpmt"])
    print(f"  Read {len(gdf):,} rows, CRS = {gdf.crs}")

    one_row_per_plot = gdf.drop_duplicates(subset="identification").copy()
    print(f"  {len(one_row_per_plot):,} unique plots")

    boundaries = one_row_per_plot.dissolve(by="cpmt").reset_index()[["cpmt", "geometry"]]
    print(f"  Dissolved into {len(boundaries):,} compartment boundaries")

    boundaries["geometry"] = boundaries["geometry"].simplify(SIMPLIFY_TOLERANCE_METRES, preserve_topology=True)
    print(f"  Simplified at {SIMPLIFY_TOLERANCE_METRES}m tolerance")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    boundaries.to_parquet(OUTPUT_PATH)
    print(f"Saved -> {OUTPUT_PATH}")


if __name__ == "__main__":
    export_compartment_boundaries()

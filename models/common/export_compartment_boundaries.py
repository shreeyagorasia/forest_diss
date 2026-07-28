# Run as: python -m models.common.export_compartment_boundaries
#
# One-off preprocessing step: reads the raw GeoPackage and saves one dissolved boundary polygon
# per spatial grouping to a small GeoParquet file, at three different scales -- compartment
# (`cpmt`), sub-compartment (`cpmt`+`scpt` together), and block (`blk`). Used both for plotting
# (e.g. results_notebooks/baseline_results.ipynb's spatial error maps) and for the
# dist_to_*_boundary environmental features (notebooks/environmental_data/
# aux_data_resolution_check.ipynb).
#
# `scpt` values (e.g. "A", "B", "C") are reused across many different compartments -- they are a
# LOCAL label within a compartment, not a globally unique one (confirmed: every one of the 26
# distinct scpt values appears in more than one compartment). So a real sub-compartment boundary
# has to dissolve by the COMBINATION of cpmt+scpt, not scpt alone -- dissolving by scpt alone
# would wrongly merge unrelated sub-compartments from different compartments into one shape.
#
# Each plot's geometry is the same across every survey year it appears in (see
# export_coordinates.py), so this drops to one row per plot BEFORE dissolving -- dissolving all
# ~795k raw rows directly would do the same union many times over for no benefit.
#
# Dissolving unions many individual 20m/40m grid cells, so each boundary inherits a "staircase"
# of tiny right-angle steps along its edge -- real detail, but far finer than a map spanning the
# whole ~28km forest can ever show (checked for the compartment case: ~26m/pixel at the figure
# sizes actually used, so individual 20m steps are sub-pixel anyway). Simplifying to 30m cuts
# total vertices by ~88% with no visible change in shape at any zoom level tested.

from pathlib import Path

import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GPKG_PATH = PROJECT_ROOT / "data" / "raw" / "LiDAR_Years_All_7jul.gpkg"
LAYER_NAME = "LiDAR_Years"
SIMPLIFY_TOLERANCE_METRES = 30

COMPARTMENT_BOUNDARIES_PATH = PROJECT_ROOT / "data" / "interim" / "compartment_boundaries.parquet"
SUBCOMPARTMENT_BOUNDARIES_PATH = PROJECT_ROOT / "data" / "interim" / "subcompartment_boundaries.parquet"
BLOCK_BOUNDARIES_PATH = PROJECT_ROOT / "data" / "interim" / "block_boundaries.parquet"


def dissolve_boundaries(one_row_per_plot, group_columns, output_path, label):
    boundaries = one_row_per_plot.dissolve(by=group_columns).reset_index()[group_columns + ["geometry"]]
    boundaries["geometry"] = boundaries["geometry"].simplify(SIMPLIFY_TOLERANCE_METRES, preserve_topology=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    boundaries.to_parquet(output_path)
    print(f"  {label}: dissolved into {len(boundaries):,} boundaries -> {output_path}")


def export_compartment_boundaries():
    print(f"Reading {GPKG_PATH} ...")
    gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME, columns=["identification", "cpmt", "scpt", "blk"])
    print(f"  Read {len(gdf):,} rows, CRS = {gdf.crs}")

    one_row_per_plot = gdf.drop_duplicates(subset="identification").copy()
    print(f"  {len(one_row_per_plot):,} unique plots")

    dissolve_boundaries(one_row_per_plot, ["cpmt"], COMPARTMENT_BOUNDARIES_PATH, "compartment")
    dissolve_boundaries(one_row_per_plot, ["cpmt", "scpt"], SUBCOMPARTMENT_BOUNDARIES_PATH, "sub-compartment")
    dissolve_boundaries(one_row_per_plot, ["blk"], BLOCK_BOUNDARIES_PATH, "block")


if __name__ == "__main__":
    export_compartment_boundaries()

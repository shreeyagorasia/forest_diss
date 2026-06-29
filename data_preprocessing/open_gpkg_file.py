"""Read the LiDAR GeoPackage and save its attribute table as a CSV.

This is the simple GeoPandas approach. It loads the complete layer into memory.

Install GeoPandas once if it is not already installed:

    python -m pip install geopandas

Then run this file:

    python data_preprocessing/open_gpkg_file.py
"""

from pathlib import Path

import geopandas as gpd


# __file__ is the location of this Python file. Its parent directory is
# data_preprocessing, and parents[1] is the forest_diss project directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input GeoPackage and output CSV locations.
GPKG_PATH = PROJECT_ROOT / "LiDAR_Years_All" / "LiDAR_Years_All.gpkg"
CSV_PATH = PROJECT_ROOT / "LiDAR_Years_All" / "LiDAR_Years_All_attributes.csv"

# This GeoPackage contains one feature layer called LiDAR_Years.
LAYER_NAME = "LiDAR_Years"


# Read the layer into a GeoDataFrame. A GeoDataFrame is like a pandas
# DataFrame, but it also has a geometry column containing the polygons.
gdf = gpd.read_file(GPKG_PATH, layer=LAYER_NAME)

print(f"Rows: {len(gdf):,}")
print(f"Columns: {len(gdf.columns)}")
print(f"Coordinate reference system: {gdf.crs}")

# Create an ordinary pandas DataFrame containing only the attribute columns.
# The polygons are not deleted from the GeoPackage; they are only omitted from
# the CSV because CSV files are not spatial files.
attribute_table = gdf.drop(columns="geometry")

# Save the complete attribute table. index=False prevents pandas from adding an
# extra numbered column to the CSV.
attribute_table.to_csv(CSV_PATH, index=False)

print(f"Saved attribute table to: {CSV_PATH}")

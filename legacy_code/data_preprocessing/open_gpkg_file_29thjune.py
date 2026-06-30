"""Extract the archived 29 June GeoPackage attributes to CSV.

This script is retained only to reproduce the legacy notebook and dataset.
Run it from anywhere with:

    python legacy_code/data_preprocessing/open_gpkg_file_29thjune.py
"""

from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DATA_DIR = (
    PROJECT_ROOT / "legacy_code" / "data" / "LiDAR_Years_All_29thjune"
)
GPKG_PATH = LEGACY_DATA_DIR / "LiDAR_Years_All.gpkg"
CSV_PATH = LEGACY_DATA_DIR / "LiDAR_Years_All_attributes.csv"
LAYER_NAME = "LiDAR_Years"


def main() -> None:
    """Read the archived attributes and recreate the legacy CSV."""
    if not GPKG_PATH.exists():
        raise FileNotFoundError(f"Legacy GeoPackage not found: {GPKG_PATH}")

    attribute_table = gpd.read_file(
        GPKG_PATH,
        layer=LAYER_NAME,
        ignore_geometry=True,
    )
    attribute_table.to_csv(CSV_PATH, index=False)

    print(f"Rows: {len(attribute_table):,}")
    print(f"Columns: {len(attribute_table.columns)}")
    print(f"Saved legacy attribute table to: {CSV_PATH}")


if __name__ == "__main__":
    main()

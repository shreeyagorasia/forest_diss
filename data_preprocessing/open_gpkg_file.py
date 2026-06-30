"""Extract the active LiDAR GeoPackage attribute table to CSV.

Run from anywhere with:

    python data_preprocessing/open_gpkg_file.py
"""

from pathlib import Path

import geopandas as gpd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GPKG_PATH = PROJECT_ROOT / "data" / "raw" / "LiDAR_Years.gpkg"
CSV_PATH = PROJECT_ROOT / "data" / "interim" / "LiDAR_Years_attributes.csv"
LAYER_NAME = "LiDAR_Years"


def main() -> None:
    """Read attributes without geometry and save them as an interim CSV."""
    if not GPKG_PATH.exists():
        raise FileNotFoundError(f"Active GeoPackage not found: {GPKG_PATH}")

    available_layers = set(gpd.list_layers(GPKG_PATH)["name"])
    if LAYER_NAME not in available_layers:
        raise ValueError(
            f"Layer {LAYER_NAME!r} not found. Available layers: "
            f"{sorted(available_layers)}"
        )

    # Geometry stays safely in the GeoPackage; only attributes go into the CSV.
    attribute_table = gpd.read_file(
        GPKG_PATH,
        layer=LAYER_NAME,
        ignore_geometry=True,
    )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    attribute_table.to_csv(CSV_PATH, index=False)

    print(f"Rows: {len(attribute_table):,}")
    print(f"Columns: {len(attribute_table.columns)}")
    print(f"Saved active attribute table to: {CSV_PATH}")


if __name__ == "__main__":
    main()

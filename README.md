# Forest dissertation

Code and notebooks for investigating and preprocessing data for a forestry dissertation.

## Repository structure

- `data_preprocessing/`: reusable preprocessing scripts
- `notebooks/`: exploratory analysis notebooks

## Data

The source and generated datasets are not committed to Git. In particular, the local
`LiDAR_Years_All/` directory is intentionally ignored because its files are too large
for an ordinary source-code repository.

Place the source GeoPackage at:

```text
LiDAR_Years_All/LiDAR_Years_All.gpkg
```

## Environment

Create and activate a virtual environment, then install GeoPandas and Jupyter:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install geopandas jupyter
```

## Preprocessing

Extract the GeoPackage attribute table as CSV:

```bash
python data_preprocessing/open_gpkg_file.py
```

The output is written to `LiDAR_Years_All/LiDAR_Years_All_attributes.csv` and remains
local because the whole data directory is ignored.

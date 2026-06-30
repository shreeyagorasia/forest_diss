# Forest dissertation

Code and notebooks for investigating and preprocessing data for a forestry dissertation.

## Repository structure

- `data/raw/`: active source data, excluded from Git
- `data/interim/`: generated attribute tables, excluded from Git
- `data/processed/`: future analysis-ready data, excluded from Git
- `data_preprocessing/`: preprocessing for the active dataset
- `notebooks/`: notebooks for the active dataset
- `legacy_code/`: archived 29 June data workflow and notebooks

## Data

The source and generated datasets are not committed to Git because they are too
large for an ordinary source-code repository.

Place the source GeoPackage at:

```text
data/raw/LiDAR_Years.gpkg
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

The output is written to `data/interim/LiDAR_Years_attributes.csv` and remains
local because the data directory is ignored.

## Legacy workflow

The previous dataset, preprocessing script, and exploratory notebook are archived
under `legacy_code/`. The main legacy notebook is:

```text
legacy_code/notebooks/lidar_years_all_29thjun.ipynb
```

It reads the archived CSV from:

```text
legacy_code/data/LiDAR_Years_All_29thjune/LiDAR_Years_All_attributes.csv
```

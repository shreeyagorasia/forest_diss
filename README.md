# Forest Dissertation — Geospatial Data Pipeline

Code and notebooks from my MSc Artificial Intelligence dissertation work with forestry data.

This repository contains the **data investigation and preprocessing workflow** used to prepare geospatial forest datasets for downstream modelling. The wider dissertation explored forest-growth prediction using inventory and GIS-derived information; the modelling experiments are kept separately in [reuben_data_my_experiments](https://github.com/shreeyagorasia/reuben_data_my_experiments).

## What this repository does

The workflow focuses on preparing a large GeoPackage-based dataset for analysis:

- loads and inspects geospatial forestry data;
- extracts GeoPackage attribute tables into a lighter tabular format;
- keeps raw, interim and processed data separated;
- preserves an older workflow under `legacy_code/` for reproducibility;
- provides notebooks for exploration of the active dataset.

## Repository structure

```text
forest_diss/
├── data/
│   ├── raw/                 # source data (not committed)
│   ├── interim/             # generated intermediate tables
│   └── processed/           # analysis-ready outputs
├── data_preprocessing/      # preprocessing scripts
├── notebooks/               # exploratory notebooks
└── legacy_code/             # archived earlier workflow
```

## Tech

- Python
- GeoPandas
- Jupyter
- GeoPackage / GIS data
- tabular data preprocessing

## Data

The source and generated datasets are not committed because they are too large for a normal source-code repository.

Place the source GeoPackage at:

```text
data/raw/LiDAR_Years.gpkg
```

## Setup

Create and activate a virtual environment, then install the required geospatial tooling:

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

The generated table is written to:

```text
data/interim/LiDAR_Years_attributes.csv
```

and remains local because the data directory is ignored by Git.

## Legacy workflow

The previous dataset, preprocessing script and exploratory notebook are archived under `legacy_code/`.

The main legacy notebook is:

```text
legacy_code/notebooks/lidar_years_all_29thjun.ipynb
```

It reads the archived CSV from:

```text
legacy_code/data/LiDAR_Years_All_29thjune/LiDAR_Years_All_attributes.csv
```

## Related modelling work

The downstream forest-growth modelling work is in:

**[Forest height growth models — Chapman–Richards + PINN](https://github.com/shreeyagorasia/reuben_data_my_experiments)**

That repository contains a classical Chapman–Richards baseline and a physics-informed neural network implemented in Python/PyTorch, with reusable training code and SLURM support.

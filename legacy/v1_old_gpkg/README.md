# Version 1 — LiDAR Years GeoPackage (legacy)

This folder preserves the exploratory analysis built against the original `LiDAR_Years.gpkg`
(4-year dataset: 2008, 2012, 2021, 2023; 550 948 rows; 24 columns).
It is kept as a fallback reference only. New work should use `LiDAR_Years_All_7jul.gpkg`.

## Data

| File | Description |
|---|---|
| `data/LiDAR_Years.gpkg` | Original GeoPackage (EPSG:27700) |
| `data/LiDAR_Years_attributes.csv` | Pre-extracted attribute table (no geometry) |

## Notebooks

| Notebook | What it does |
|---|---|
| `notebooks/lidar_years_data_understanding.ipynb` | Dataset overview, spatial maps, variable distributions, Chapman-Richards curve fitting, cleaning stages |
| `notebooks/lidar_years_data_cleaning.ipynb` | Sequential cleaning funnel (4-year and 6-year cohorts), volume equations, variable dependency audit |

Both notebooks use a project-root discovery pattern to locate `data/LiDAR_Years.gpkg` via the
presence of that file — they do not require any environment variables or path configuration.

## Known data quality issues (v1)

- `plyr = 0` artefact: ~10% of rows had planting year coded as 0; these created spurious ages equal to the survey year.
- LAI capped at 5.5 (sensor ceiling), concentrated in the 2021 survey.
- Negative `Vol` values present in a small number of rows.
- Survey expansion between 2002/2006 and 2008–2023 means spatial coverage is not uniform.
- `SCDB` vintage concern: block/compartment identifiers may not reflect the current stand boundaries.

These issues are documented and partially handled in `lidar_years_data_cleaning.ipynb`.

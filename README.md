# Forest dissertation

Code and notebooks for a forestry dissertation using Aberfoyle LiDAR data,
working towards PINN-based growth modelling.

## Repository structure

- `data/raw/`: active source GeoPackage, excluded from Git
- `data/interim/`: generated attribute tables (e.g. plot coordinates), excluded from Git
- `data/processed/`: cleaned exports and model-ready tables, excluded from Git — see `data/processed/README.md`
- `data_exploration_gpkg/notebooks/`: active data-understanding and data-cleaning notebooks
- `data_processing/`: standalone scripts that derive model-ready tables from the cleaned master exports — no notebook dependency, see below
- `models/`: baseline model code, organised as one folder per model plus `models/common/` for shared utilities (metrics, splits, plotting, saving)
- `outputs/`: per-model fitted parameters/predictions/metrics, one folder per model per cohort, excluded from Git
- `results_notebooks/`: notebooks that summarise saved results and plots — read `outputs/` only, never refit anything
- `legacy/`: archived datasets, scripts, and notebooks from earlier data versions — preserved as-is
- `documentation/`: dissertation plan, citations, key-terms cheat sheet, draft LaTeX source

## Data

The source and generated datasets are not committed to Git because they are too
large for an ordinary source-code repository.

The modelling cohorts are balanced-panel survivor cohorts, not all Aberfoyle plots: a plot must
have every required survey, remain Sitka spruce, and pass the planting-year, age and height checks.
Results therefore describe this retained population rather than harvested, converted, damaged,
young or incompletely surveyed plots.

Place the source GeoPackage at:

```text
data/raw/LiDAR_Years_All_7jul.gpkg
```

## Environment

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Active workflow

Producing the model-ready data is a two-step, plain-Python pipeline (no notebook dependency):

1. `python -m data_processing.clean_master_data` does the actual cleaning funnel — species
   filtering, age/height validity checks, deduplication, cohort balancing — and writes the
   cleaned **master exports** to `data/processed/master/`.
2. `python -m data_processing.export_model_tables` reads those master exports and derives the
   consolidated per-cohort `model_table.parquet` (`current_state/`) and the transition tables.

The original exploratory notebooks this pipeline was converted from (`lidar_years_all_data_understanding.ipynb`,
`lidar_years_all_data_cleaning.ipynb`) are archived at `legacy/2026-07-28/` — their diagnostic/
plotting cells have standalone value but no longer drive the exported data.

See `data/processed/README.md` for the exact file layout, and
`data_processing/export_model_tables.py`'s module docstring for exactly which columns are
features vs. evaluation-only, and why.

## Legacy workflow

Earlier dataset versions, preprocessing scripts, and exploratory notebooks are
archived under `legacy/` and preserved for reference; they are not part of the
active workflow.

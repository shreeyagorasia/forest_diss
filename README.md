# Forest dissertation

Code and notebooks for a forestry dissertation using Aberfoyle LiDAR data, developing
physics-informed (PINN) growth models for Sitka spruce top-height prediction and spatial
attribution of growth-curve departure.

## Research questions

The dissertation's three questions, and the older internal code labels they replaced. Code,
job scripts, and `outputs/` folder names still use the old `rq1`/`rq2`/`rq3` labels below. This
table maps between the two:

| Current label | Question | Old code label |
| --- | --- | --- |
| **Q1** | Persistent curve-departure attribution, which environmental variables explain a plot's departure from the shared Chapman-Richards growth curve? | old **RQ2** |
| **Q2** | Spatial attribution, does a spatially-varying model (GNNWR) explain plot-level curve deviation better than a global model? | old **RQ3** |
| **Q3** | Prediction and physics guidance, does environment-conditioning and a physics (Chapman-Richards) constraint improve raw top-height prediction? | old **RQ1** + **RQ2a** |

A third label scheme also appears inside the code itself: `models/xgb_environmental/feature_set_builder.py`
(see "Building environmental feature sets" below) labels these **RSQ1/RSQ2/RSQ3**, mapping
1:1 onto old RQ1/RQ2/RQ3 above (i.e. RSQ1=old RQ1, RSQ2=old RQ2, RSQ3=old RQ3), not onto the
current Q1/Q2/Q3 numbering. Three different label sets for the same three questions is genuinely
confusing; this table is the single place that reconciles all of them.

## Pipeline stages, in order

1. **Clean data**: `data_processing/` (below). A plain two-step CLI, no notebook involved.
2. **Build environmental feature sets**: see "Building environmental feature sets" below. This
   step is notebook-driven, not a CLI script, which is easy to miss.
3. **Run models**: `jobs/` + `models/<family>/run_*.py` (see "Canonical entry points" and
   "Model variants" below). Writes to `outputs/`.
4. **Display/plot results**: `notebooks/results_q1/`, `results_q2/`, `results_q3/`. Read-only,
   these only read from `outputs/`, never fit/train/retrain anything.

## Repository structure

- `data_processing/`: standalone scripts that derive model-ready tables from the cleaned master
  exports, no notebook dependency, see "Active workflow" below.
- `models/`: model code, organised as one folder per model family plus `models/common/` for
  shared utilities (metrics, splits, data loading, plotting, saving). See "Model variants" below.
- `jobs/`: cluster (SLURM) job scripts. `jobs/rq123_methodology/` is the canonical orchestration
  layer for all three research questions; other subfolders are per-model fit/evaluate primitives
  it calls. `jobs/retrain/` is an unexecuted draft plan, kept for reference only. See "Canonical
  entry points" below.
- `temp_results_pinn/`: self-contained investigation that found and fixed a bug in the PINN
  forward pass (see "Model variants" below), kept as the full record of the bug discovery, the
  fix derivation, and the corrected reruns. Its own `PLAN.md` documents the 15 pitfalls hit along
  the way.
- `notebooks/`: notebooks that summarise saved results and plots, read `outputs/` only, never
  refit or retrain anything.
- `figures/`: curated, dissertation-cited figures.
- `documentation/`: dissertation plan, citations, key-terms cheat sheet, model design-decision
  notes (`model_instructions/`), the running project decision log (`progress_notes/`), and the
  draft LaTeX source (`refocus_draft/`).
- `TEMP_results/`, `TEMP_results_attribution/`: dated LaTeX table exports and results ledger.
  Each file's own header states what it captures and whether it has since been superseded by a
  later-dated file on the same topic.
- `outputs/` (not included in this submission, see below): per-model fitted parameters,
  predictions, and metrics, one folder per model per cohort/split.
- `data/` (not included in this submission, see below): source and derived datasets.
- `legacy/` (not included in this submission): archived datasets, scripts, and notebooks from
  earlier project stages, preserved as-is for provenance.

## What's included in this submission

This is a code/materials submission, not a full copy of the working repository. Excluded, and
why:

- **`data/`**: the source LiDAR GeoPackage and all derived tables. Excluded because of size and
  because this is licensed third-party forestry data, not redistributable. See "Data" below for
  how to obtain/place it.
- **`outputs/`**: all fitted model results. Only a small evidence subset (metrics, run metadata,
  and predictions for the canonical final runs) is included; checkpoints and every diagnostic/
  tuning run are excluded. Full results are regenerable from the code here plus the source data,
  see "Reproducing results" below for exactly which commands regenerate which parts of it.
- **`legacy/`**: archived material from earlier project stages, not needed to run or assess the
  current pipeline.

Everything needed to understand, re-run, and extend the actual modelling code is included.

## Data

The source and generated datasets are not committed here because they are too large and because
the source LiDAR data is not freely redistributable.

### Where the data came from

- **LiDAR-derived inventory (the core dataset)**: Aberfoyle Forest District survey data, processed
  from raw point clouds to per-plot inventory fields by Forest Research using their Area-Based
  Analysis and Hybrid Method (LiDAR-derived Top Height combined with species, age, and management
  records to estimate volume, yield class, and related fields). Supplied as
  `LiDAR_Years_All_7jul.gpkg`, one row per plot per survey year. Formula references and field
  definitions are in `documentation/data_exploration_gpkg/`.
- **Wind**: two sources. Long-term average wind speed/direction (Weibull parameters) from the
  Global Wind Atlas, sampled onto plot locations by `data_processing/add_environmental_candidates.py`.
  Observed storm-interval wind data from CEDA's MIDAS Open station network, downloaded by
  `data_processing/download_midas_wind_aberfoyle.py` (requires a CEDA account, see
  `data_processing/create_ceda_token.py`) and summarised by `data_processing/build_midas_wind_intervals.py`.
- **Terrain**: OS Terrain 50 tiles, used by `data_processing/add_environmental_candidates.py` to
  derive multiscale terrain measures (e.g. topographic position index, local relief).
- **Roads/rivers**: OS Open Roads and OS Open Rivers, used as terrain/access candidate features.

The modelling cohorts are balanced-panel survivor cohorts, not all Aberfoyle plots: a plot must
have every required survey, remain Sitka spruce, and pass the planting-year, age and height checks.
Results therefore describe this retained population rather than harvested, converted, damaged,
young or incompletely surveyed plots.

To re-run anything, place the source GeoPackage at:

```text
data/raw/LiDAR_Years_All_7jul.gpkg
```

The exact schema of every derived table (`master/`, `current_state/`, `transitions/`) this
pipeline produces from that GeoPackage is documented in
`documentation/data_schema/processed_data_layout.md`, including column meanings and cohort row
counts, kept here since `data/processed/README.md` itself is not included in this archive.

## Environment

Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Active workflow

Producing the model-ready data is a two-step, plain-Python pipeline (no notebook dependency):

1. `python -m data_processing.clean_master_data` does the cleaning funnel: species filtering,
   age/height validity checks, deduplication, cohort balancing, and writes the cleaned
   **master exports** to `data/processed/master/`.
2. `python -m data_processing.export_model_tables` reads those master exports and derives the
   consolidated per-cohort `model_table.parquet` (`current_state/`) and the transition tables.

`data_processing/add_environmental_candidates.py` and `data_processing/export_growth_curve_tables.py`
extend this with environmental features and the Q1/Q2 growth-curve table respectively, see each
script's own header for what it reads and produces.

See `data/processed/README.md` (present once `data/` is populated) for the exact file layout, and
`data_processing/export_model_tables.py`'s module docstring for exactly which columns are
features vs. evaluation-only, and why.

## Building environmental feature sets

Unlike data cleaning, this step is **notebook-driven, not a CLI script**:
`notebooks/environmental_data/multicollinearity_screen_set1_5.ipynb` is the actual entry point.
It calls into `models/xgb_environmental/feature_set_builder.py`, a shared library (despite living
in the `xgb_environmental` folder, it is imported by all three research questions' models, not
just XGBoost-environmental) that builds the same nested "Set1-5" feature hierarchy for each RSQ,
ranking/gating candidate environmental variables against that RSQ's own target column:

- Set1 = baseline only (5 stand/management columns every model already gets)
- Set2 = Set1 + top 5 candidates by |Spearman rho| against the target
- Set3 = Set1 + terrain/wind candidates clearing a signal-gate threshold
- Set4 = Set1 + every other category's candidates clearing the same gate
- Set5 = Set1 + every deduplicated candidate, no gate

The resulting sets are read back by name (e.g. `nested_set3_gated_terrain_wind_vif`) via
`ENV_TERRAIN_FEATURE_SETS` in `models/common/torch_data.py` (DNN/PINN) or
`FEATURE_SETS` in `models/xgb_environmental/xgb_environmental.py` (XGBoost/Elastic Net/NLME).
See `documentation/env_feature_sets_manifest.csv` for the manifest these are loaded from.

## Model variants

Each model family is one folder under `models/`, with a `run_<model>.py` (fit) and
`evaluate_<model>.py` (test-set evaluation) pair, following a strict fit-now/evaluate-later split.
See any `run_*.py`'s own header comment for the exact invocation.

- `models/pinn_noenv/`, `dnn_noenv/`: no environmental conditioning.
- `models/dnn_env_terrain/`: environment-conditioned DNN.
- `models/pinn_env_terrain/`, `models/pinn_env_terrain_k/`: environment-conditioned PINN
  (terrain/wind adjusts `y_max` only, or both `y_max` and `k`). **These contain a known bug**: the
  terrain/wind sub-network's output only ever reached the physics-loss target, never the actual
  height prediction, at any physics-weight setting. Left in place, untouched, because existing
  `outputs/` results were produced with this version and need it to be interpreted correctly.
- **`models/pinn_env_terrain_fix/`**: the corrected version. `forward()` now routes the
  terrain-conditioned `y_max`/`k` through to the prediction itself (a Chapman-Richards term plus
  a residual from the main network), not just the loss. Migrated from `temp_results_pinn/` where
  the fix was originally isolated and validated. See that file's own header for the exact bug and
  fix, and `temp_results_pinn/PLAN.md` for the full investigation.
- `models/baselines/`, `xgb_baseline/`, `xgb_environmental/`, `rf_baseline/`: non-neural
  baselines.
- `models/growth_curve_attribution/`: NLME, Elastic Net, XGBoost, and GNNWR models for Q1/Q2
  (growth-curve departure attribution and spatial attribution).
- `models/spatial_attribution/`: Q1 environmental-attribution models.

**Known limitation**: the PINN fix has only been rerun on the 4-survey cohort. The 6-survey
cohort's PINN-env numbers, if cited anywhere, are still from the buggy pre-fix version. A
6-survey rerun using `models/pinn_env_terrain_fix/` is still open work.

## Canonical entry points

`jobs/rq123_methodology/` is the orchestration layer for all three research questions (see its
own README for the exact script-per-step matrix). It calls the per-model scripts under
`jobs/<model_family>/`, which in turn invoke the corresponding `models.<family>.run_*` module.
Fit happens on the cluster; `evaluate_*`/plotting steps are meant to run locally.

## Safe commands (evaluation and plotting only, no retraining)

To re-check a metric or regenerate a figure without retraining anything, evaluate an existing
checkpoint and/or run a results notebook:

```bash
python -m models.pinn_env_terrain.evaluate_pinn_env_terrain --cohort 4survey --split-type spatial_block_kfold
```

Then open the relevant notebook under `notebooks/results_q1/`, `results_q2/`, or `results_q3/`.
These only read from `outputs/`, they never call `.fit()`/`.train()`. `models/pinn_env_terrain_fix/`
has no `evaluate_*.py` of its own yet (see its header); its own `run_full_rerun.py` fits and
evaluates in one step and is not cheap to re-run. Check
`temp_results_pinn/outputs/full_rerun/fold_*/summary.json` before re-running any fold, since
existing results there are skipped automatically, not overwritten.

## Reproducing results

Every `outputs/` subfolder is reproducible by running the matching `jobs/` script(s) against a
populated `data/`, see `jobs/rq123_methodology/README.md` for the canonical step-by-step matrix,
and each model family's own `run_*.py` for its exact CLI. Nothing under `outputs/` is required to
regenerate itself; it is a cache of prior runs, not an input to anything.

## Archived code

`legacy/` holds archived datasets, scripts, and notebooks from earlier project stages (not
included in this submission). The original exploratory data-understanding/cleaning notebooks the
current `data_processing/` pipeline was converted from are archived there too, their
diagnostic/plotting cells have standalone reference value but no longer drive the exported data.

# data/processed/

All files here are Parquet (not CSV) — chosen over CSV because it preserves dtypes exactly
(booleans stay boolean instead of round-tripping as the strings `"True"`/`"False"`, categorical
columns like `thinning_status` stay a proper string dtype), and produces smaller files. Not
committed to Git (see root `.gitignore`).

Two-step, notebook-independent pipeline:

1. **`master/`** is produced by `python -m data_processing.clean_master_data`. It runs the
   cleaning funnel, balances each cohort to exactly 4 or 6 surveys per plot, and derives the
   survey-specific thinning fields. The old cleaning notebook is retired.
2. **`current_state/`** and **`transitions/`** are produced by
   `python -m data_processing.export_model_tables`, a plain Python script with no dependency on
   the notebook. It only needs `master/` to already exist — selecting which columns go into which
   model's table is pure pandas column selection on already-cleaned data, so if the notebook is
   ever mid-edit or broken, every downstream table can still be regenerated from `master/` alone.
   See that script's module docstring for the full reasoning behind every column choice.

Both cohorts are drawn from the same active LiDAR dataset
(`data/raw/LiDAR_Years_All_7jul.gpkg`) — they are balanced survey-count subsets,
not separate raw-data versions. They are balanced-panel survivor cohorts: inclusion requires
complete survey coverage, Sitka spruce throughout, and valid planting-year, age and height data.
They should not be described as representing every Aberfoyle plot.

- **4survey**: 71,766 plots x 4 surveys = 287,064 rows
- **6survey**: 13,897 plots x 6 surveys = 83,382 rows

## master/

Broad research tables (not feature matrices) — one row per plot per survey,
retaining audit and downstream-forestry variables (volumes, GYC fields, formula
ingredients) that must never be used as model predictors — see the "feature vs.
evaluation-only" comment block at the top of `data_processing/export_model_tables.py`
for exactly which columns those are and why.

```
clean_master_4survey.parquet
clean_master_6survey.parquet
```

## current_state/

Current-state model tables, one row per plot per survey. The target is the raw
`elev_percentile_95th`. `Top_Height95` is retained for audit only and is never a predictor.
Cohort is encoded by directory, not filename.

```
current_state/4survey/model_table.parquet
current_state/6survey/model_table.parquet
```

## transitions/

Change between consecutive surveys per plot, including the raw 95th-percentile height increment
and the earlier/later survey features used by trajectory models.

```
transition_growth_4survey.parquet
transition_growth_6survey.parquet
```

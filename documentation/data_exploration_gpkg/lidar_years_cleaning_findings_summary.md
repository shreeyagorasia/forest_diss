# LiDAR Years cleaning findings summary

This summary is based on `lidar_years_all_data_cleaning.ipynb` (archived at `legacy/2026-07-28/`
once its cleaning logic was converted to `data_processing/clean_master_data.py`, 29 July 2026) and
the supplied **LiDAR_Years layer field descriptions**.

## What the new field documentation resolved

The field-description document answers several questions that were previously uncertain:

- `p1–p5` are Chapman-Richards growth-model parameters used in the `GYCspec95/99` calculations. They are lookup/model coefficients, not plot-level environmental predictors.
- `g1` and `g3` are volume-equation coefficients used in `Vol95/Vol99`.
- `Thin` is derived as `1` when `last_thinn != 0`, otherwise `0`. It means a last thinning year is recorded, not necessarily that the plot was recently thinned.
- `Age = LiDAR_year - plyr`.
- `Top_Height95 = elev_percentile_95th × 1.1`.
- `Top_Height99 = elev_percentile_99th`.
- `Vol95 = (g1 × Top_Height95^g3) × (area / 10000) × CanopyCover`.
- `Vol99 = (g1 × Top_Height99^g3) × (area / 10000) × CanopyCover`.
- `Vol_RM95 = (-35.8733 + 5.3486 × Top_Height95^1.5424) × (area / 10000) × CanopyCover`.
- `GYCspec95 = (Top_Height95 / ((1 - exp(-p4 × Age))^p5) - p1 - p3 × 2) / p2`.
- `GYCspec99 = (Top_Height99 / ((1 - exp(-p4 × Age))^p5) - p1 - p3 × 2) / p2`.

Important correction: `Vol_RM95` is not a running mean. It is a generic Robertson & Miller-style volume equation using top height, area, and canopy cover.

## What remains unresolved or still needs caution

- `Top_Height99 < Top_Height95` occurs in a substantial share of rows. The formulas are now known, but the inversion still needs explanation because a corrected 95th percentile can exceed the raw 99th percentile after applying the 1.1 multiplier.
- `GYCspec95/99` formulas are known, but the fields still contain infinite and extreme values. They also directly use height and age, so they should not be used as predictors when modelling `Top_Height95`.
- `spis` and `Species` disagree systematically. The notebook uses `spis == "SS"` because `spis` is the FR inventory species code from the grid polygon layer.
- `whcl` is confirmed as an FR inventory windthrow hazard class, not raw LiDAR. It is management-linked and should not be included in baseline models.

## Cleaned cohort outputs

The notebook builds two balanced Sitka spruce trajectory datasets:

| Cohort | Survey years | Plots | Rows |
|---|---:|---:|---:|
| Four-survey | 2008, 2012, 2021, 2023 | 71,766 | 287,064 |
| Six-survey | 2002, 2006, 2008, 2012, 2021, 2023 | 13,897 | 83,382 |

The cleaning logic is plot-level after the cohort-year step. If a plot fails a rule in any required survey year, the whole plot is removed so the retained dataset remains balanced.

Cleaning rules applied:

1. Keep required cohort years.
2. Keep plots present in every required survey year.
3. Keep Sitka spruce using `spis == "SS"`.
4. Keep valid planting years.
5. Keep plausible age.
6. Keep plausible `Top_Height95`.

Volume is not used as a general cleaning rule. Negative `Vol_RM95` values remain in the master height datasets and should only be filtered in volume-specific analyses.

## Main modelling decisions

- Main target: `Top_Height95`.
- Main biological time input: `Age`.
- Baseline predictors: `Age`, `CanopyCover`, `Thin`, `time_since_thinning`, and `yldc`, depending on model type.
- `blk` is retained only for spatial/grouped validation, not as a predictor.
- `whcl` is retained in the master export for auditing but excluded from all baseline and draft PINN model tables.
- `Vol95`, `Vol99`, and `Vol_RM95` are excluded from top-height predictors because they are derived from height.
- `GYCspec95/99` are excluded from top-height predictors because they are derived from height and age.
- `LAI` is retained in the master table, but the baseline feature sets use `CanopyCover` to avoid redundant canopy inputs.

### Why source columns are dropped from the master modelling export

The cleaned master exports are narrower than the original source table. The aim is
to keep columns that are useful for modelling, grouping, auditing, or later joins,
while dropping fields that duplicate information, leak the target, are unstable, or
are not needed for the planned baseline models.

| Dropped source column(s) | Main reason for dropping |
|---|---|
| `GapFraction` | Redundant canopy/gap information. The baseline keeps `CanopyCover` as the simpler canopy-structure variable. |
| `elev_percentile_95th` | Direct raw input to `Top_Height95 = elev_percentile_95th × 1.1`; keeping it would duplicate/leak the target. |
| `elev_percentile_99th` | Direct raw input to `Top_Height99`; too close to the height target and affected by the percentile inconsistency. |
| `AOI` | Filename/source label, not a biological plot predictor. |
| `block` | Parsed LiDAR filename block; `blk` from the inventory is retained instead for spatial grouping. |
| `SPIS_RF` | Random-forest species-raster code; species filtering uses the FR inventory `spis` field instead. |
| `Species` | Lookup species name from `SPIS_RF`; disagrees with `spis` in many rows, so it is not used for the cleaned Sitka spruce export. |
| `p1`, `p2`, `p3`, `p4`, `p5` | Documented Chapman-Richards lookup parameters. They are not plot-level measured predictors and are used inside `GYCspec95/99`. |
| `flyr` | Final/rotation year field, but unusable in this extract because it is all zeros. |
| `area` | Mainly grid-cell geometry/edge information. It affects volume scaling but is not needed for top-height baselines. |
| `g1`, `g3` | Volume-equation coefficients used in `Vol95/Vol99`; not top-height predictors. |
| `Top_Height99` | Alternative height outcome, not an independent predictor. It also has the `Top_Height99 < Top_Height95` issue. |
| `Vol95`, `Vol99` | Derived from height, area, canopy cover and coefficients; would leak target information in top-height models. |
| `GYCspec95`, `GYCspec99` | Derived from height and age using the Chapman-Richards formula; leakage-prone and contains infinite/extreme values. |

### Why columns are dropped for each model table

All model-specific tables keep `identification`, `LiDAR_year`, and `blk` as
metadata. These are used to identify rows and support temporal/spatial splitting.
They are not predictors. `Top_Height95` is the target for every top-height model.

| Model table | Predictors kept | Main columns deliberately excluded from predictors | Reason |
|---|---|---|---|
| `cr_age_*` | `Age` | `CanopyCover`, `Thin`, `time_since_thinning`, `yldc`, `LAI`, `whcl`, volume fields, `GYCspec` fields, identifiers | Pure age-only growth curve. This is the simplest Chapman-Richards/neural growth baseline. |
| `linear_baseline_*` | `Age`, `CanopyCover`, `Thin`, `yldc` | `time_since_thinning`, `LAI`, `whcl`, `Vol95/99`, `Vol_RM95`, `GYCspec95/99`, raw height percentiles, identifiers | Keeps the interpretable baseline small. Uses one canopy variable and avoids height-derived leakage. |
| `rf_baseline_*` | `Age`, `CanopyCover`, `Thin`, `time_since_thinning`, `yldc` | `LAI`, `whcl`, volume fields, `GYCspec` fields, raw height percentiles, identifiers | Adds time since thinning for non-linear modelling while still avoiding redundant canopy inputs and leakage. |
| `dnn_age_*` | `Age` | Same as `cr_age_*` | Pure neural age-to-height comparison. This checks whether a DNN adds value without contextual variables. |
| `dnn_context_*` | `Age`, `CanopyCover`, `Thin`, `time_since_thinning`, `yldc` | `LAI`, `whcl`, volume fields, `GYCspec` fields, raw height percentiles, identifiers | Uses the same defensible feature set as the random forest so the comparison is fair. |
| `pinn_draft_*` | `Age`, `CanopyCover`, `Thin`, `time_since_thinning`, `yldc` | `whcl`, `LAI`, volume fields, `GYCspec` fields, raw height percentiles, identifiers | Draft PINN table uses current non-wind variables only. Continuous terrain/climate/wind/soil variables should be joined later as independent environmental inputs. |

In short, the first model tables are intentionally conservative. The excluded
variables are not necessarily useless; they are excluded from the baseline because
they are redundant, derived from the target, management-linked, unstable, or better
used for validation/auditing rather than prediction.

## Thinning and wind-hazard findings

The thinning/wind-hazard analysis supports the management-rule interpretation:

- For records with `whcl > 4`, only 2.63% are marked as thinned.
- These exceptions represent 312 distinct plots.
- Class 6 is almost entirely unthinned.
- Within comparable strata, thinned records had about 0.72 m higher median `Top_Height95`, about 3.57% higher derived `Vol_RM95`, 0.018 lower canopy cover, and 0.046 lower LAI.

These are observational differences, not causal effects. `Vol_RM95` is derived from height, area, and canopy cover, so volume differences should not be treated as independent evidence.

## Export tables created in the notebook

Master exports:

- `clean_export_4`
- `clean_export_6`

Model-specific tables:

- `cr_age_4`, `cr_age_6`
- `linear_baseline_4`, `linear_baseline_6`
- `rf_baseline_4`, `rf_baseline_6`
- `dnn_age_4`, `dnn_age_6`
- `dnn_context_4`, `dnn_context_6`
- `pinn_draft_4`, `pinn_draft_6`

The notebook does not write CSV files by default. Set `EXPORT_FILES = True` in the optional export section when ready.

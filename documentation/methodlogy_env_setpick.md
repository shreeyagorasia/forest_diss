# Environmental Feature Methodology — Final Implementation Summary

Reconstructed from the implemented pipeline (`models/xgb_environmental/feature_set_builder.py`,
`models/xgb_environmental/multicollinearity_screen.py`, `notebooks/environmental_data/
multicollinearity_screen_set1_5.ipynb`), the generated manifest
(`documentation/env_feature_sets_manifest.csv`, 225 rows, verified 2026-08-10), and direct queries
against `data/processed/environmental/plot_environmental_features.parquet` (71,766 plots) and
`data/processed/current_state/4survey/model_table.parquet` (287,064 rows). Presented in pipeline
order, not development order. Three research questions, one shared recipe, different targets:

- **RSQ1** — predictive tiers feeding `dnn_env_terrain`/`pinn_env_terrain*`. Target:
  `elev_percentile_95th` (raw LiDAR height).
- **RSQ2** — Avenue 1 attribution, feeding Elastic Net/XGBoost/NLME. Target: `mean_cr_residual`.
- **RSQ3** — Avenue 2 attribution, feeding Elastic Net/XGBoost/GNNWR. Target:
  `local_y_max_difference`.

All statistics below are computed on the **4survey cohort** (this project's primary cohort) at
**plot level** (`plot_environmental_features.parquet`, one row per plot) unless stated otherwise.
Environmental/terrain variables are static per plot (do not vary by survey year); management/
stand-structure variables do vary by survey year, so RSQ1's row-level table
(`model_table.parquet`) is reported separately for those.

---

## 1. Data-source summary

### 1.1 Terrain (8 variables in the final widest RSQ1/RSQ2 sets)

| Variable | Source | Resolution | Period | Static/time-varying | Extraction method | Type | Valid | Missing | Distinct | Min | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `elevation` | External — OS Terrain 50 DTM | 50 m | current survey | Static | Direct raster read | Continuous | 71,766 (100.00%) | 0 (0.00%) | 4,028 | 18.2 | 561.1 |
| `slope_degrees` | Own calculation | 50 m DTM grid | — | Static | `arctan(√(dz_dx²+dz_dy²))`, `np.gradient` | Continuous | 71,766 (100.00%) | 0 | 15,326 | 0.0 | 42.84 |
| `northness` | Own calculation | 50 m DTM grid | — | Static | `cos(aspect_degrees→radians)` | Continuous | 71,766 (100.00%) | 0 | 15,387 | −1.0 | 1.0 |
| `eastness` | Own calculation | 50 m DTM grid | — | Static | `sin(aspect_degrees→radians)` | Continuous | 71,766 (100.00%) | 0 | 15,572 | −1.0 | 1.0 |
| `profile_curvature` | Own calculation | 50 m DTM grid | — | Static | 2nd derivative of DTM (Zevenbergen & Thorne 1987) | Continuous | 71,766 (100.00%) | 0 | 16,835 | −0.00725 | 0.00837 |
| `plan_curvature` | Own calculation | 50 m DTM grid | — | Static | Same method as `profile_curvature` | Continuous | 71,766 (100.00%) | 0 | 16,829 | −0.6667 | 0.4000 |
| `local_relief_500m` | Own calculation | 500 m window (21×21 cells, 50 m grid) | — | Static | Max − min elevation in window | Continuous | 71,766 (100.00%) | 0 | 6,086 | 7.50 | 461.90 |
| `solar_radiation_index` | Own calculation | 50 m DTM grid | Summer solstice, solar noon (fixed) | Static | `cos(slope)·cos(zenith) + sin(slope)·sin(zenith)·cos(aspect−azimuth)`, no horizon shading | Continuous | 71,766 (100.00%) | 0 | 16,012 | 0.3384 | 0.9996 |
| `frost_hollow_flag` | Own calculation | Derived from TPI/curvature | — | Static | Binary: TPI < own 15th percentile AND `plan_curvature` < 0 | Binary (float 0/1) | 71,766 (100.00%) | 0 | 2 | 0 (n=62,691) | 1 (n=9,075) |
| `ceh_twi` | External — CEH Topographic Wetness Index raster | 50 m | current | Static | Direct raster read | Continuous | 71,727 (99.95%) | 39 (0.05%) | 16,800 | 3.751 | 16.861 |

`tpi`, `tpi_500m`, `elevation_roughness` exist in the source parquet but did not survive VIF
screening into any final Set — see §2.7 and §5.

**Missing-value treatment**: `ceh_twi`'s 39 missing plots (shared with the other CEH-raster
variables, same source file) are dropped whole-plot downstream (`load_split_table_with_terrain`),
not imputed.

### 1.2 Wind (up to 8 variables)

| Variable | Source | Resolution | Period | Static/time-varying | Extraction method | Type | Valid | Missing | Distinct | Min | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `topex` | Own calculation | 1000 m radius horizon sampling | — | Static | Sum of horizon angles, 8 compass directions (Wilson 1984) | Continuous | 71,766 (100.00%) | 0 | 16,839 | −85.89 | 68.58 |
| `windward_topex` | Own calculation | Same, single bearing | — | Static | Same method, restricted to 225° (prevailing SW wind) | Continuous | 71,766 (100.00%) | 0 | 8,093 | −19.83 | 19.42 |
| `gwa_weibull_a_10m` | External — Global Wind Atlas national raster | national grid | long-term climatology | Static | Direct raster read, 10 m height | Continuous | 71,766 (100.00%) | 0 | 1,824 | 0.068 | 12.442 |
| `gwa_weibull_a_50m` | External — Global Wind Atlas national raster | national grid | long-term climatology | Static | Direct raster read, 50 m height | Continuous | 71,766 (100.00%) | 0 | 1,824 | 3.210 | 11.461 |
| `gwa_weibull_k_10m` | External — Global Wind Atlas national raster | national grid | long-term climatology | Static | Direct raster read, 10 m height | Continuous | 71,766 (100.00%) | 0 | 1,822 | 0.656 | 1.944 |
| `gwa_weibull_k_50m` | External — Global Wind Atlas national raster | national grid | long-term climatology | Static | Direct raster read, 50 m height | Continuous | 71,766 (100.00%) | 0 | 1,818 | 1.246 | 1.963 |
| `gwa_wind_speed_10m` | External — Global Wind Atlas REST API | ~140–155 m (E–W) × 250 m (N–S), anisotropic | long-term climatology | Static | Nearest-cell extraction | Continuous | 71,766 (100.00%) | 0 | 1,816 | 0.069 | 11.178 |
| `gwa_wind_speed_50m` | External — Global Wind Atlas national raster | national grid | long-term climatology | Static | Direct raster read, 50 m height | Continuous | 71,766 (100.00%) | 0 | 1,824 | 2.874 | 10.261 |
| `whcl` | External — raw GPKG survey field | plot polygon | current | Static (constant per plot across survey years) | Direct field read | Ordinal (int32) | 71,766 (100.00%) | 0 | **6 observed** | 0 | 6 |

**`whcl` — distinguish documented range from observed data**: the field is documented as a
0–6 ordinal windthrow hazard class (7 possible codes). In this dataset, **class 1 does not occur**
— observed level counts: `0`→396, `2`→13,375, `3`→9,558, `4`→36,135, `5`→4,847, `6`→7,455. So the
*source's* possible category range is 7 codes; the *plot-level output* here has 6 distinct values.

### 1.3 Soil / site (up to 7 variables, before one-hot expansion)

| Variable | Source | Resolution | Period | Static/time-varying | Extraction method | Type | Valid | Missing | Distinct | Min/Max or levels |
|---|---|---|---|---|---|---|---|---|---|---|
| `soilgrids_ph` | External — SoilGrids v2.0, ISRIC | 250 m | current | Static | Topsoil layer (0–5 cm), direct read | Continuous | 71,353 (99.42%) | 413 (0.58%) | 13 | 4.0 – 5.2 |
| `ceh_pedotope` | External — CEH natural capital raster | 50 m | current | Static | Categorical class ID, one-hot encoded | Categorical | 71,727 (99.95%) | 39 (0.05%) | 7 levels | `10.0`→38,027; `11.0`→26,376; `12.0`→6,887; `8.0`→215; `2.0`→108; `9.0`→103; `5.0`→11 |
| `ceh_subsurface_drainage` | External — CEH natural capital raster | 50 m | current | Static | Categorical class ID, one-hot encoded | Categorical | 71,727 (99.95%) | 39 (0.05%) | 3 levels | `1.0`→54,320; `3.0`→14,281; `2.0`→3,126 |
| `ceh_textural_composition` | External — CEH natural capital raster | 50 m | current | Static | Categorical class ID, one-hot encoded | Categorical | 71,727 (99.95%) | 39 (0.05%) | 4 levels | `4.0`→64,403; `5.0`→6,887; `2.0`→273; `1.0`→164 |
| `dist_to_watercourse` | Own calculation | OS Open Rivers vector, 5 km buffered bbox | current | Static | Distance to nearest line | Continuous | 71,766 (100.00%) | 0 | 71,758 | 0.0045 – 1850.86 |

**Extraction note**: the 39-plot CEH gap (`ceh_pedotope`/`ceh_subsurface_drainage`/
`ceh_textural_composition`/`ceh_twi`) is the *same* 39 plots across all four CEH-sourced
variables — one shared raster-boundary/edge-of-window gap, not four independent failures.

**Reference-level treatment**: three RSQ3 categoricals get one reference level dropped before
screening (`ceh_pedotope=10.0`, `ceh_subsurface_drainage=1.0`, `ceh_textural_composition=4.0` —
each the modal/most-common level) — see §2.4. RSQ1/RSQ2 exclude all three categoricals from their
candidate pool entirely (never one-hot encoded for those two RSQs).

### 1.4 Climate (2–4 variables depending on RSQ)

| Variable | Source | Resolution | Period | Static/time-varying | Extraction method | Type | Valid | Missing | Distinct | Min | Max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `chelsa_gdd5_degc` | External — CHELSA v2.1 BIOCLIM+ | 1 km | **1981–2010 climatology** (static, not survey-year-matched) | Static | Direct read | Continuous | 71,766 (100.00%) | 0 | 206 | 906.5 | 1580.8 |
| `chelsa_bio12_precip_mm` | External — CHELSA v2.1 | 1 km | **1981–2010 climatology** (static, not survey-year-matched) | Static | Direct read | Continuous | 71,766 (100.00%) | 0 | 269 | 1729.8 | 3004.8 |
| `tas_mean` | External — HadUK-Grid, CEDA | 1 km | **Averaged over each cohort's own real survey years** (4survey: 2008/2012/2021/2023) | Time-varying by cohort | Cohort-suffixed columns (`tas_mean_4survey`/`_6survey`) resolved to plain name per cohort | Continuous | 71,766 (100.00%) | 0 | **149 distinct extracted values** | 5.754 | 9.145 |
| `groundfrost_mean` | External — HadUK-Grid, CEDA | 1 km | Same multi-year cohort averaging as `tas_mean` | Time-varying by cohort | Same cohort-suffix resolution | Continuous | 71,766 (100.00%) | 0 | 149 | 8.756 | 9.473 |

**A genuine limitation, not resolved by this pipeline**: `chelsa_gdd5_degc`/`chelsa_bio12_precip_mm`
are a single static 1981–2010 climatology applied uniformly to every plot regardless of that
plot's actual survey year (2008–2023) — a real temporal mismatch. `tas_mean`/`groundfrost_mean`
are cohort-year-averaged and therefore closer to survey-year-matched, but still a mean across
multiple years, not a single matched year. Neither is corrected here; flagged as an existing,
disclosed limitation of the underlying data sources, not something this screening pipeline fixes.

### 1.5 Spatial position / edge effects (up to 6 variables)

| Variable | Source | Extraction method | Type | Valid | Missing | Distinct | Min | Max |
|---|---|---|---|---|---|---|---|---|
| `dist_to_cpmt_boundary` | Own calculation, GPKG compartment polygon | Distance to own compartment boundary | Continuous | 71,766 (100.00%) | 0 | not separately re-queried this pass — see `dist_to_scpt_boundary`/`dist_to_block_boundary` below for the same geometry family | — | — |
| `dist_to_scpt_boundary` | Own calculation, GPKG sub-compartment polygon | Distance to own sub-compartment boundary | Continuous | 71,766 (100.00%) | 0 | 68,188 | 8.1e-12 | 216.99 |
| `dist_to_block_boundary` | Own calculation, GPKG block polygon | Distance to own block boundary | Continuous | 71,766 (100.00%) | 0 | 53,974 | 6.2e-11 | 216.99 |
| `dist_to_forest_perimeter` | Own calculation, buffered union of compartments | Distance to forest exterior/interior boundary | Continuous | 71,766 (100.00%) | 0 | 71,701 | 9.8e-11 | 764.69 |
| `cpmt_compactness_ratio` | Own calculation, GPKG compartment polygon | Perimeter / area (per-compartment, not per-plot) | Continuous | 71,766 (100.00%) | 0 | 296 | 0.00974 | 0.23975 |
| `dist_to_road` | Own calculation, OS Open Roads vector | Distance to nearest line (incl. tracks) | Continuous | 71,766 (100.00%) | 0 | 71,580 | 0.056 | 2375.65 |

**Categorisation note (see §3)**: `dist_to_road` sits in `spatial_position_edge_effects` under
RSQ1/RSQ2's `CATEGORY_GROUPS`, and in `edge_position` under RSQ3's `FEATURE_GROUPS` — same
variable, two different category-scheme names for the same underlying grouping. Not an
inconsistency in the data, a naming difference between the two pre-existing group dictionaries
this pipeline reuses rather than redefines.

### 1.6 Stand structure / management (4 final variables, RSQ2/RSQ3 baseline)

Reported twice — plot-level (RSQ2/RSQ3's actual input) and row-level (RSQ1's actual input) — because
the two differ materially in shape, not just row count.

**Plot-level** (`plot_environmental_features.parquet`, 71,766 plots — mean/fraction aggregated
across each plot's own survey years):

| Variable | Type | Valid | Missing | Distinct | Min | Max |
|---|---|---|---|---|---|---|
| `CanopyCover` | Continuous (mean fraction) | 71,766 (100.00%) | 0 | 70,769 | 0.0081 | 0.9963 |
| `time_since_thinning` | Continuous (NaN-filled with 0 downstream) | 71,766 (100.00%) | 0 | 19 | 0.0 | 16.0 |
| `time_since_thinning_missing` | Continuous fraction | 71,766 (100.00%) | 0 | 5 | 0.0 (n=4,095) | 1.0 (n=50,831) |
| `recent_thinning_5yr` | Continuous fraction | 71,766 (100.00%) | 0 | 3 | 0.0 (n=54,354) | 0.5 (n=7,467) |

**Row-level** (`model_table.parquet`, 287,064 survey rows — RSQ1's actual input, one row per
plot per survey year):

| Variable | Type | Valid | Missing | Distinct |
|---|---|---|---|---|
| `CanopyCover` | float64 | 287,064 (100.00%) | 0 | 224,125 |
| `Thin` | int64 (binary) | 287,064 (100.00%) | 0 | 2 |
| `time_since_thinning` | float64 | 45,452 (**15.8%**) | 241,612 (**84.2%**) | 24 |
| `time_since_thinning_missing` | **bool** | 287,064 (100.00%) | 0 | 2 |
| `recent_thinning_5yr` | **bool** | 287,064 (100.00%) | 0 | 2 |

`time_since_thinning` is genuinely missing for 84.2% of individual survey rows at row level (any
row where the plot had not yet been thinned as of that survey) — filled with 0 downstream, with
`time_since_thinning_missing` carrying the "this 0 is a placeholder" signal (`torch_data.py::
fill_missing_time_since_thinning`). The **dtype difference (`bool` at row level vs. `float`
fraction at plot level) is not cosmetic** — it caused a real crash in VIF computation, see §5.

**`Thin`, dropped from the final baseline** (§2.3): confirmed directly, `Thin +
time_since_thinning_missing = 1.0` exactly for all 71,766 plots at plot level, zero exceptions —
a genuine deterministic duplicate, not an approximation.

`thinning_status` (5–6 level derived re-bucketing of the above) exists in the source data but was
never part of any candidate pool — excluded upstream, matching pre-existing project precedent
(`CATEGORY_GROUPS["stand_structure"]` and `MANAGEMENT_COLUMNS` both already excluded it before
this work began).

### 1.7 Excluded from every candidate pool, on record

- **`Age`**: circular for RSQ2/RSQ3 (their targets are built from `Age` via the Chapman-Richards
  curve). Not circular for RSQ1 — included there as a *control* variable in the reference model
  used for ranking (never exported as a feature, since RSQ1's real model already receives `Age`
  through a separate pathway).
- **`inverse_slope_proxy`, `gwa_wind_p95_10m`/`_50m`, `gwa_prob_above_critical_10m`/`_50m`**: dropped
  at dedup stage 1 as documented closed-form functions of other candidates already in the pool
  (formulas in `DETERMINISTIC_DUPLICATES`, `multicollinearity_screen.py`).
- **`chelsa_bio1_celsius`**: dropped at dedup stage 2 as a near-exact empirical duplicate of
  `tas_mean` (Spearman ρ ≈ 0.997).
- **`neighbour_mean_height`/`neighbour_height_differential`**: excluded project-wide, pre-dating
  this work — confirmed to leak test-set ground truth (documented separately, not re-verified
  here).
- **`era5_land_temp_k`**: excluded on the numbers alone, pre-dating this work — only 8 distinct
  values across 71,766 plots (documented separately, not re-verified here).

---

## 2. Final feature-selection methodology, in pipeline order

### 2.1 Candidate identification and grouping

RSQ1/RSQ2 share one candidate pool: `xgb_environmental.ALL_FEATURE_COLUMNS` (37 columns) minus
`CATEGORY_GROUPS["stand_structure"]` (5 columns, handled separately as baseline) minus the three
unordered categorical columns (never valid as continuous input) = **40 shared candidates**.
RSQ3 uses `broad_environmental_check.FEATURE_GROUPS` (`terrain_wind`/`climate`/`soil_site`/
`edge_position`, `management` excluded — handled as baseline) with categoricals one-hot expanded
= **44 candidates** before reference-level dropping (41 after — see §2.4).

**Automated.** Grouping is a fixed lookup against pre-existing dictionaries
(`CATEGORY_GROUPS`, `FEATURE_GROUPS`), not re-derived here.

### 2.2 Data-quality and coverage checks

Target-circularity check (§1.7's `Age` note) run once for all three targets, confirmed by reading
the actual target-construction code (`build_plot_level_table`), not assumed. **Judgement-based**
in the sense that "is this variable circular" required reading construction code, not a
threshold; automated in that once confirmed, `Age` is unconditionally excluded, never
re-evaluated per run.

### 2.3 Removal of unusable, duplicated or poorly supported variables

**Rule**: `drop_deterministic_duplicates()` — drop any candidate matching a documented closed-form
formula of another candidate already in the pool (§1.7). **Automated**, dictionary-driven, zero
free parameters.

**`Thin` removed from baseline**: confirmed empirically (not by formula) that `Thin = 1 −
time_since_thinning_missing` exactly, all 71,766 plots. Added as a documented entry alongside the
formula-based duplicates once confirmed — same treatment, different discovery method.
`time_since_thinning_missing` kept (paired missingness flag for `time_since_thinning`); `Thin`
dropped (redundant given the other two). **Baseline is now 4 columns for all three RSQs**:
`CanopyCover`, `time_since_thinning`, `time_since_thinning_missing`, `recent_thinning_5yr`.

### 2.4 Categorical variables and reference levels

**Rule, RSQ1/RSQ2**: exclude `ceh_pedotope`/`ceh_subsurface_drainage`/`ceh_textural_composition`
from the candidate pool entirely (their pipeline has no categorical-encoding path).

**Rule, RSQ3**: one-hot encode all three (`prepare_broad_table`, pre-existing), then — new in
this work — drop exactly one level per categorical (`drop_reference_level_per_category`) before
dedup/ranking/VIF run, so the design matrix used by every downstream signal has no structural
singularity. Levels dropped: `ceh_pedotope=10.0`, `ceh_subsurface_drainage=1.0`,
`ceh_textural_composition=4.0` (in each case, alphabetically/numerically first surviving level —
a deterministic tie-break, not chosen by frequency or significance). **Automated** once the rule
is fixed; the specific level dropped is a mechanical consequence of sort order, not a judgement
call.

**Why required**: confirmed directly — before this fix, RSQ3's widest set showed `VIF = inf` for
multiple `ceh_*` dummy columns (one categorical's dummies structurally summing to a constant,
perfectly collinear with the regression intercept). Not evidence of real redundancy between
meaningful variables.

### 2.5 Low-variation / near-constant features

No explicit separate low-variance filter is applied. `whcl` (6 observed levels, one theoretically
possible level absent) and `frost_hollow_flag` (binary, 12.6% positive) both passed through
unfiltered — neither is near-constant enough to trigger removal under any rule actually applied.
**Not implemented as a distinct stage** — flagged here as a genuine gap against the requested
structure, not silently folded into another step.

### 2.6 Pairwise correlation screening

**Rule**: `find_near_exact_duplicates()` — any pair at `|Spearman ρ| ≥ 0.95` is treated as a
near-duplicate; `choose_representative_loser()` keeps one (external-provenance preferred over
derived, then higher `|ρ|` with that RSQ's own target as tie-break). **Automated**, fixed
threshold, applied identically across all three RSQs.

**Confirmed pairs resolved this run**: `gwa_wind_speed_50m`/`gwa_weibull_a_50m` (ρ = +1.000,
RSQ1/RSQ2); `chelsa_bio1_celsius`/`tas_mean` (ρ ≈ 0.997, all three RSQs); two RSQ3-specific
categorical dummy pairs (`ceh_textural_composition=4.0`↔another level, ρ ≈ −0.965;
`ceh_pedotope=12.0`↔another level, ρ = 1.000).

**Checked, not assumed, to remain distinct**: `topex` vs. `windward_topex` — Spearman ρ = 0.6168
on the real RSQ3 table, well below the 0.95 threshold. Both are kept in the same sets deliberately
(direction-specific exposure genuinely differs from omnidirectional exposure here), not an
oversight.

### 2.7 VIF / multicollinearity checks

**Rule**: iterative VIF reduction, threshold 5.0 (Dormann et al. 2013) — compute VIF for every
remaining column, drop the single worst, recompute, repeat (`iterative_vif_reduction`/
`run_vif_pass`). Applied to **Set2 (with backfill) and Set3/Set4/Set5, all three RSQs** — extended
from RSQ2-only after direct evidence: RSQ1's Set2 was independently found 4-of-5 non-baseline
columns VIF ≥ 5; RSQ3's Set3 was independently found 5-of-9. **Automated**, fixed threshold and
procedure; baseline columns are structurally protected from removal (still included in the design
matrix, so they still affect every other column's VIF, but are never the one dropped) — a
judgement call (baseline must always be present, agreed design) implemented as a hard constraint,
not re-litigated per run.

**Set2's algorithm differs from Set3/4/5's**: Set2 needs a *fixed size* (10 candidates), so it
uses sequential add-with-skip-and-backfill (walk the importance ranking in order; skip a candidate
if its own VIF against baseline + already-kept candidates exceeds 5.0; try the next-ranked
candidate instead) rather than iterative reduction on a fixed starting set. This is a genuine
algorithmic difference, not an inconsistency — Set3/4/5 are defined as "everything that clears a
gate," which can legitimately shrink; Set2 is defined as "the top 10," which should not silently
shrink to 7 or 8 just because some higher-ranked candidates turned out to be collinear.

**VIF drop counts** (final, all three RSQs, after §2.4's reference-level fix):

| Set | RSQ1 | RSQ2 | RSQ3 |
|---|---|---|---|
| Set2 (skipped, not dropped — see above) | 3 skipped, backfilled | 1 skipped, backfilled | 3 skipped, backfilled |
| Set3 | 15→15 (0) | 15→14 (1) | 13→12 (1) |
| Set4 | 21→18 (3) | 21→19 (2) | 24→20 (4) |
| Set5 | 37→30 (7) | 37→28 (9) | 42→31 (11) |

`elevation` is the single most frequent VIF casualty across Set4/Set5 in all three RSQs —
consistently redundant with the wider terrain/wind pool once enough other variables are present.

### 2.8 Spatial diagnostics (Moran's I, ICC, variogram)

**Not part of this feature-selection pipeline.** Moran's I is used elsewhere in this project as a
*post-fit residual diagnostic* (does spatial structure remain in a model's errors), a different
question from "which variable belongs in the feature set," and is not computed anywhere inside
`feature_set_builder.py` or the Set1–5 notebook. Reported here as a confirmed absence, not an
oversight to be quietly filled in.

### 2.9 Relationship with the target

**Rule**: rank-aggregate of three signals against that RSQ's own target — Spearman `|ρ|`
(`rank_by_target_correlation`), XGBoost permutation importance (`permutation_importance_ranking`),
XGBoost drop-column ablation R² change (`drop_column_ablation_ranking`). Each candidate's rank
under each method is averaged (`combine_importance_ranks`); Set2 = top 10 by average rank
(subject to §2.7's VIF backfill); Set3/Set4 gate = top half of each candidate's own category by
average rank (`gate_columns_by_combined_rank`, ties round up via `ceil`).

Permutation/ablation both fit ONE reference XGBoost model per RSQ on a single
`spatial_block_split` (not the project's full pooled 5-fold CV — that evaluation is reserved for
whichever Set is eventually chosen, not for this screening step). RSQ1's reference model includes
`Age` as a control variable (never ranked, never exported) — without it, the reference model
scored R² = −0.168 (worse than predicting the mean); with it, R² = 0.255.

**These are association measures, not causal or even confirmed predictive-value evidence.** A
high combined rank means a variable is associated with the target across three different lenses
on one held-out split of this dataset — it does not establish that the variable causes the
observed growth pattern, nor does it guarantee the variable will matter once the chosen Set is
evaluated under the project's real 5-fold spatial CV.

**Known limitation of permutation importance specifically**, disclosed not hidden: biased under
correlated predictors (Strobl et al. 2008) — shuffling one column while its correlated partners
stay fixed creates combinations absent from the real data. Mitigated, not eliminated, by §2.6's
dedup running first.

### 2.10 Domain-based decisions and final retention

**Judgement-based, not automated**:
- `whcl` stays categorized as `wind`, never `stand_structure`/`management`, despite being read
  from the same raw GPKG survey record as the management columns — a domain call (it measures
  wind-damage susceptibility, not a management action).
- Baseline (4 columns) is unconditionally present in every Set, all three RSQs, including RSQ3
  (whose pre-existing `SCOPE_GROUPS` system deliberately has no fixed baseline). This was an
  explicit decision to prioritise a comparable nested hierarchy across RSQs over preserving
  `SCOPE_GROUPS`'s own management-isolation property — `SCOPE_GROUPS` itself is untouched and
  still available for that specific question.
- Set2's fixed size (10, not 5) was widened specifically to leave headroom once VIF screening
  removes collinear candidates from a fixed-size top-N pick.

### 2.11 Construction of final Set1–5

Set1 = baseline (4 columns, all RSQs). Set2 = baseline + 10 VIF-screened, backfilled candidates
by combined rank. Set3 = baseline + terrain/wind candidates in the top half of their category by
combined rank, VIF-screened. Set4 = Set3's terrain/wind members + every other category's
top-half candidates, VIF-screened. Set5 = baseline + every deduplicated candidate, VIF-screened,
no rank filter.

### 2.12 Full Set1–5 membership (exact columns, pulled fresh from the manifest, 2026-08-11)

Baseline (`CanopyCover`, `time_since_thinning`, `time_since_thinning_missing`,
`recent_thinning_5yr`) is included in every row below exactly as the manifest stores it. **RSQ1's
baseline is stripped before export** into `ENV_TERRAIN_FEATURE_SETS` (fed to the model through a
separate pathway — see §5) — the RSQ1 lists below are the manifest's own (baseline-included) form,
for direct comparison with RSQ2/RSQ3; subtract the 4 baseline columns to get what
`torch_data.py` actually receives.

**RSQ1** (target `elev_percentile_95th`):

| Set | n | Columns |
|---|--:|---|
| Set1 | 4 | `CanopyCover`, `time_since_thinning`, `time_since_thinning_missing`, `recent_thinning_5yr` |
| Set2 | 14 | Set1 + `gwa_weibull_k_50m`, `dist_to_scpt_boundary`, `elevation`, `tas_mean`, `eastness`, `windward_topex`, `gwa_weibull_a_10m`, `gwa_weibull_a_50m`, `slope_degrees`, `cpmt_compactness_ratio` |
| Set3 | 15 | Set1 + `gwa_weibull_k_50m`, `elevation`, `eastness`, `windward_topex`, `gwa_weibull_a_10m`, `gwa_weibull_a_50m`, `slope_degrees`, `gwa_weibull_k_10m`, `solar_radiation_index`, `ceh_twi`, `gwa_wind_speed_10m` |
| Set4 | 18 | Set1 + `gwa_weibull_k_50m`, `eastness`, `windward_topex`, `gwa_weibull_a_10m`, `gwa_weibull_a_50m`, `slope_degrees`, `gwa_weibull_k_10m`, `solar_radiation_index`, `ceh_twi`, `gwa_wind_speed_10m`, `dist_to_scpt_boundary`, `tas_mean`, `chelsa_gdd5_degc`, `cpmt_compactness_ratio` |
| Set5 | 30 | Set1 + `slope_degrees`, `northness`, `eastness`, `profile_curvature`, `plan_curvature`, `local_relief_500m`, `frost_hollow_flag`, `topex`, `windward_topex`, `gwa_weibull_a_10m`, `gwa_weibull_k_10m`, `gwa_weibull_a_50m`, `gwa_weibull_k_50m`, `dist_to_forest_perimeter`, `dist_to_scpt_boundary`, `cpmt_compactness_ratio`, `dist_to_road`, `dist_to_watercourse`, `gwa_wind_speed_10m`, `soilgrids_ph`, `ceh_twi`, `chelsa_gdd5_degc`, `chelsa_bio12_precip_mm`, `tas_mean`, `groundfrost_mean`, `whcl` |

**RSQ2** (target `mean_cr_residual`, baseline retained on export — no second pathway):

| Set | n | Columns |
|---|--:|---|
| Set1 | 4 | `CanopyCover`, `time_since_thinning`, `time_since_thinning_missing`, `recent_thinning_5yr` |
| Set2 | 14 | Set1 + `chelsa_bio12_precip_mm`, `chelsa_gdd5_degc`, `gwa_weibull_k_50m`, `slope_degrees`, `gwa_weibull_a_10m`, `local_relief_500m`, `eastness`, `dist_to_road`, `gwa_weibull_a_50m`, `tas_mean` |
| Set3 | 14 | Set1 + `gwa_weibull_k_50m`, `slope_degrees`, `gwa_weibull_a_10m`, `local_relief_500m`, `eastness`, `gwa_weibull_a_50m`, `ceh_twi`, `whcl`, `topex`, `gwa_wind_speed_10m` |
| Set4 | 19 | Set3 + `chelsa_bio12_precip_mm`, `dist_to_road`, `tas_mean`, `dist_to_scpt_boundary`, `soilgrids_ph` |
| Set5 | 28 | Set1 + `slope_degrees`, `northness`, `eastness`, `profile_curvature`, `plan_curvature`, `tpi_500m`, `local_relief_500m`, `frost_hollow_flag`, `windward_topex`, `gwa_weibull_a_10m`, `gwa_weibull_k_10m`, `gwa_weibull_a_50m`, `gwa_weibull_k_50m`, `dist_to_forest_perimeter`, `dist_to_block_boundary`, `cpmt_compactness_ratio`, `dist_to_road`, `dist_to_watercourse`, `gwa_wind_speed_10m`, `soilgrids_ph`, `ceh_twi`, `tas_mean`, `groundfrost_mean`, `whcl` |

**RSQ3** (target `local_y_max_difference`, baseline retained on export; `ceh_*` columns are
one-hot dummy levels, reference level already dropped per §2.4):

| Set | n | Columns |
|---|--:|---|
| Set1 | 4 | `CanopyCover`, `time_since_thinning`, `time_since_thinning_missing`, `recent_thinning_5yr` |
| Set2 | 14 | Set1 + `windward_topex`, `elevation`, `cpmt_compactness_ratio`, `gwa_wind_speed_50m`, `slope_degrees`, `tpi_500m`, `dist_to_road`, `topex`, `solar_radiation_index`, `soilgrids_ph` |
| Set3 | 12 | Set1 + `windward_topex`, `elevation`, `gwa_wind_speed_50m`, `slope_degrees`, `tpi_500m`, `topex`, `solar_radiation_index`, `northness` |
| Set4 | 20 | Set1 + `windward_topex`, `gwa_wind_speed_50m`, `slope_degrees`, `tpi_500m`, `topex`, `northness`, `cpmt_compactness_ratio`, `dist_to_road`, `chelsa_bio12_precip_mm`, `tas_mean`, `soilgrids_ph`, `ceh_subsurface_drainage=2.0`, `ceh_pedotope=2.0`, `ceh_pedotope=8.0`, `ceh_textural_composition=2.0`, `dist_to_forest_perimeter` |
| Set5 | 31 | Set1 + `slope_degrees`, `northness`, `eastness`, `profile_curvature`, `plan_curvature`, `ceh_twi`, `frost_hollow_flag`, `windward_topex`, `whcl`, `tpi_500m`, `local_relief_500m`, `tas_mean`, `groundfrost_mean`, `soilgrids_ph`, `dist_to_watercourse`, `dist_to_forest_perimeter`, `dist_to_scpt_boundary`, `cpmt_compactness_ratio`, `dist_to_road`, `ceh_pedotope=11.0`, `ceh_pedotope=2.0`, `ceh_pedotope=5.0`, `ceh_pedotope=8.0`, `ceh_pedotope=9.0`, `ceh_subsurface_drainage=2.0`, `ceh_subsurface_drainage=3.0`, `ceh_textural_composition=5.0` |

Note Set3's non-monotonic behaviour is expected, not an error: RSQ3 Set3 (12) is smaller than
RSQ3 Set2 (14) because Set2 and Set3 are built by two different rules (top-10 by rank vs.
top-half-of-category by rank) over overlapping but not identical candidate pools — Set3 is not
defined as a superset of Set2.

Machine-readable source: `documentation/env_feature_sets_manifest.csv` (225 rows) — the file
above is a snapshot for readability; the CSV is the definitive record if the two ever disagree.

---

## 3. Final variable-category table

**The specific 25-variable table supplied for audit could not be matched against any of the 15
sets currently in the manifest** — its variable membership (e.g. `elevation`, `profile_curvature`,
and `elevation_roughness` together in `terrain`; three CHELSA/HadUK columns together in `climate`)
does not correspond to any current RSQ1/RSQ2/RSQ3 Set2/3/4/5 after the VIF and reference-level
fixes in §2.4/§2.7. It most likely reflects an intermediate version of one RSQ3 Set, from before
those fixes were applied, that is no longer the current implementation. Per instruction, the table
below is reconstructed fresh from the current manifest rather than corrected in place.

**RSQ1, widest set (`nested_set5_all_ungated_vif`, 26 non-baseline candidates)**, categorised via
`CATEGORY_GROUPS`:

| Category | n | Variables |
|---|--:|---|
| terrain | 8 | `slope_degrees`, `northness`, `eastness`, `profile_curvature`, `plan_curvature`, `local_relief_500m`, `frost_hollow_flag`, `ceh_twi` |
| wind | 8 | `topex`, `windward_topex`, `gwa_weibull_a_10m`, `gwa_weibull_k_10m`, `gwa_weibull_a_50m`, `gwa_weibull_k_50m`, `gwa_wind_speed_10m`, `whcl` |
| spatial_position_edge_effects | 4 | `dist_to_forest_perimeter`, `dist_to_scpt_boundary`, `cpmt_compactness_ratio`, `dist_to_road` |
| climate | 4 | `chelsa_gdd5_degc`, `chelsa_bio12_precip_mm`, `tas_mean`, `groundfrost_mean` |
| soil_site | 2 | `dist_to_watercourse`, `soilgrids_ph` |
| **Total** | **26** | (baseline, 4, reported separately — never categorised as "environmental") |

**RSQ2, widest set (`nested_set5_all_ungated_vif`, 24 non-baseline candidates)**:

| Category | n | Variables |
|---|--:|---|
| terrain | 9 | `slope_degrees`, `northness`, `eastness`, `profile_curvature`, `plan_curvature`, `tpi_500m`, `local_relief_500m`, `frost_hollow_flag`, `ceh_twi` |
| wind | 7 | `windward_topex`, `gwa_weibull_a_10m`, `gwa_weibull_k_10m`, `gwa_weibull_a_50m`, `gwa_weibull_k_50m`, `gwa_wind_speed_10m`, `whcl` |
| spatial_position_edge_effects | 4 | `dist_to_forest_perimeter`, `dist_to_block_boundary`, `cpmt_compactness_ratio`, `dist_to_road` |
| climate | 2 | `tas_mean`, `groundfrost_mean` |
| soil_site | 2 | `dist_to_watercourse`, `soilgrids_ph` |
| **Total** | **24** | (+ 4 baseline = 28, matching the manifest's reported Set5 size) |

**RSQ3, widest set (`nested_set5_all_ungated_vif`, 27 non-baseline candidates)**, categorised via
`FEATURE_GROUPS` (RSQ3's own dictionary — note different category names from RSQ1/RSQ2's
`CATEGORY_GROUPS` for the same underlying groupings):

| Category | n | Variables |
|---|--:|---|
| terrain_wind | 11 | `slope_degrees`, `northness`, `eastness`, `profile_curvature`, `plan_curvature`, `ceh_twi`, `frost_hollow_flag`, `windward_topex`, `whcl`, `tpi_500m`, `local_relief_500m` |
| soil_site | 10 | `soilgrids_ph`, `dist_to_watercourse`, `ceh_pedotope=11.0`, `ceh_pedotope=2.0`, `ceh_pedotope=5.0`, `ceh_pedotope=8.0`, `ceh_pedotope=9.0`, `ceh_subsurface_drainage=2.0`, `ceh_subsurface_drainage=3.0`, `ceh_textural_composition=5.0` |
| edge_position | 4 | `dist_to_forest_perimeter`, `dist_to_scpt_boundary`, `cpmt_compactness_ratio`, `dist_to_road` |
| climate | 2 | `tas_mean`, `groundfrost_mean` |
| **Total** | **27** | (+ 4 baseline = 31, matching the manifest) |

**Answers to the specific audit questions**:
- These are current final model columns, per the manifest, verified 2026-08-10.
- No variable is counted twice within any one Set's table.
- One-hot columns are shown individually (each level a real, distinct binary column in the design
  matrix) under their parent categorical's category — showing them collapsed under one row would
  misstate the actual column count a model receives.
- `dist_to_road` belongs to `spatial_position_edge_effects` (RSQ1/RSQ2 naming) /
  `edge_position` (RSQ3 naming) — same grouping, different pre-existing dictionary names; not a
  miscategorisation.
- Management/stand-structure variables are reported **separately from "environmental"** in every
  table above (as baseline, not a category row) — this is deliberate (§2.10), not an omission.

---

## 4. Feature-set purpose

| Feature set | Included feature groups | Model comparison | Environmental attribution | Reason |
|---|---|---:|---:|---|
| RSQ1 Set1–Set5 | Baseline + terrain/wind/climate/soil/edge (VIF-screened) | Yes (primary) | No | Feeds `dnn_env_terrain`/`pinn_env_terrain*` — designed for a fair DNN-vs-PINN comparison under progressively richer environmental input, and to test whether environmental conditioning improves height prediction. No linear/coefficient model reads these; not designed for attribution. |
| RSQ2 Set1–Set5 | Baseline + terrain/wind/climate/soil/edge (VIF-screened) | Secondary (EN vs. XGBoost vs. NLME) | Yes (primary) | Feeds Elastic Net/XGBoost/NLME on `mean_cr_residual` — designed to identify which conditions align with persistent departure from the shared CR curve. Model comparison across the three fitted models is possible but secondary to the attribution question. |
| RSQ3 Set1–Set5 | Baseline + terrain_wind/climate/soil_site/edge_position (VIF-screened, one-hot, reference level dropped) | Secondary (EN vs. XGBoost vs. GNNWR) | Yes (primary) | Feeds Elastic Net/XGBoost/GNNWR on `local_y_max_difference` — per-plot attribution; GNNWR specifically tests whether attribution is spatially varying, not just globally linear. |
| RSQ3 `SCOPE_GROUPS` (pre-existing, untouched) | Named category combinations (`terrain_wind`, `+climate`, `+soil_site`, `+edge_position`, `+management`, `broad_environment`, `broad_environment_plus_management`) | No | Yes (category-level) | A *different*, coarser question — "does adding this whole category help" — answered by a pre-existing system this work did not modify. No RSQ2 equivalent exists. |
| 6survey informational check (RSQ2 Set2 only) | Same candidate pool, 6survey cohort | No | Diagnostic only | Not a primary result — checks whether Set2's top-10 changes on the smaller (7,467-row), noisier 6survey cohort. It does (only 1 of 10 candidates overlap with 4survey's Set2) — reported as a caveat on Set2's stability, not as a second deliverable. |

Widening a Set (Set2→Set3→Set4→Set5) is **not** claimed to necessarily improve prediction or
strengthen an attribution finding — each Set is a candidate for downstream evaluation, and which
one performs best is an empirical question for the project's real 5-fold spatial CV, not decided
by this screening pipeline.

---

## 5. Code corrections and downstream effects

| Code/file area | Previous problem | Correction | Why required | Outputs affected | Results-script risk |
|---|---|---|---|---|---|
| `torch_data.py`, `ENV_TERRAIN_FEATURE_SETS` (pre-existing `stage1–4` tiers) | RSQ1's tiers were gated using correlation against `mean_cr_residual` (RSQ2's target), not RSQ1's own `elev_percentile_95th` | New Set1–5 built with every ranking/gate function taking `target_column` as an explicit argument, never a shared constant | Confirmed empirically: old `stage3_terrain_wind_plus` kept 33 of 40 candidates (near-unfiltered); correctly targeted, the equivalent new Set3 keeps 11 | New `nested_set*` tiers only — old `stage1–4` tiers untouched, still present, not deleted | No effect on old tiers' already-computed results. New tiers require fresh training runs — none exist yet. |
| `feature_set_builder.SET1_BASELINE_COLUMNS` | `Thin`/`time_since_thinning_missing` are an exact deterministic duplicate (confirmed, not assumed), unknown to the pipeline | `Thin` excluded from baseline | Baseline changed from 5 to 4 columns for all three RSQs | Every Set for all three RSQs (baseline size) | Schema change (one fewer baseline column) — any code assuming 5-column baseline needs updating. |
| Candidate-pool construction (RSQ1/RSQ2) | Excluding only the 4-column reduced baseline (not the full 5-column management group) let `Thin` leak back in as an "environmental candidate" | Pool now excludes `CATEGORY_GROUPS["stand_structure"]` (all 5) | Hit directly: `torch_data.py`'s own disjointness guard rejected `Thin` at runtime | RSQ1/RSQ2 candidate pool (40, not 41) | Schema change — pool size affected. No effect on RSQ3 (never had this bug). |
| RSQ3 table construction (`prepare_broad_table` call site) | Baseline columns never merged into RSQ3's table at all (only requested environmental groups) | Baseline explicitly requested in the merge, then excluded again from the screening-only pool | Permutation/ablation need baseline present as real data in the design matrix | RSQ3 table columns | No downstream risk — table now has the correct columns; screening pool unaffected. |
| `multicollinearity_screen.compute_vif_table` | `.to_numpy()` on a DataFrame mixing `bool` and `float64` columns can yield an object-dtype array; `statsmodels.add_constant` then raises `TypeError` on `np.isfinite` | Explicit `.astype(float)` cast added | Hit directly running VIF on RSQ1's row-level table (`time_since_thinning_missing`/`recent_thinning_5yr` are `bool` there, `float` in the plot-level tables) | Every VIF computation in the whole project, not just this pipeline (shared function) | **No schema/value change** — same VIF numbers, just no longer crashes on bool-containing tables. Any other caller of `compute_vif_table` on a bool-containing table benefits silently. |
| RSQ3 categorical encoding (screening pool only, not `prepare_broad_table` itself) | Every level of `ceh_pedotope`/`ceh_subsurface_drainage`/`ceh_textural_composition` kept → design-matrix singularity → `VIF = inf` | `drop_reference_level_per_category` removes one level per categorical before dedup/ranking/VIF | Confirmed necessary: RSQ3 Set5 showed `VIF = inf` for multiple `ceh_*` dummies before the fix | RSQ3 candidate pool (44→41 before dedup), all downstream RSQ3 Sets | Column-count and naming change for RSQ3's `nested_set*` tiers only. `prepare_broad_table` and the pre-existing `SCOPE_GROUPS` system are untouched — no effect there. |
| `feature_set_builder.build_set2` | Set2 (top-N by importance) had no collinearity check at all — a collinear pair (`dist_to_scpt_boundary`/`dist_to_cpmt_boundary`, mutual VIF ≈ 9.14) could both survive since Set3/4/5's VIF pass never ran on Set2 | Rewritten to walk the ranked list in order, skip a candidate whose VIF (against baseline + already-kept) exceeds 5.0, and backfill from the next-ranked candidate | Found in review, before any downstream model was fit on the affected Set2 | RSQ1/RSQ2/RSQ3 Set2 membership (all three changed) | Set2's variable list and function signature changed (`build_set2` now takes `df` and returns a tuple `(columns, skip_log)`) — any code calling the old signature needs updating. No results exist yet on the old Set2, so no rerun is needed, only a fresh run. |
| `torch_data.py`, `ENV_TERRAIN_FEATURE_SETS` (new `nested_set*` entries) | Wired (earlier in this session) to read live from the manifest via `load_feature_set()`, using the *old* set names (`nested_set2_top5`, `nested_set3_gated_terrain_wind`, `nested_set4_gated_all`) | Updated to the current manifest names (`nested_set2_top10`, `nested_set3_gated_terrain_wind_vif`, `nested_set4_gated_all_vif`) | Renames elsewhere in the pipeline (Set2 top-5→top-10; `_vif` suffix added) broke this lookup — confirmed by hitting `ValueError: No rows found` at import time | `torch_data.py` module import itself | **Import-time failure if left unfixed** — any script importing `torch_data` (which is most of this project's DNN/PINN code) would crash at import, not just at feature-set lookup. Now fixed; import succeeds. |
| Manifest ↔ `torch_data.py` circular dependency | `torch_data.py` reads the manifest CSV at import time; the manifest is only regenerated by running the Set1–5 notebook, which itself imports `torch_data.py` first | One-time manual bridge: renamed the on-disk manifest's `set_name` values to the new names before the first post-rename notebook run, so import-time lookup succeeded; the notebook then overwrote the manifest with real recomputed data | Otherwise neither side can go first after any set-name change | Manifest CSV (bridged, then correctly regenerated) | **Structural risk, not yet resolved**: any future set-name change in the notebook will reproduce this exact circular failure at `torch_data.py` import time. Needs either a decoupling (e.g. `torch_data.py` falling back gracefully if a name is missing) or a documented manual-bridge step — flagged in §6, not fixed. |

---

## 6. Final checks and writing style

### 6.1 Confirmed from the evidence

- All statistics in §1 are computed directly against `plot_environmental_features.parquet`
  (71,766 plots) and `model_table.parquet` (287,064 rows), 2026-08-10 — not estimated or
  carried over from an earlier document.
- 225-row manifest (`documentation/env_feature_sets_manifest.csv`) matches the 15 expected
  (RSQ × Set) combinations, sizes as reported in §2.11/§3.
- `torch_data.py` imports successfully and its `ENV_TERRAIN_FEATURE_SETS` dict resolves against
  the current manifest (verified by re-running the notebook end to end, zero errors).
- Every VIF drop count in §2.7 and every skip/backfill in Set2 is read directly from this run's
  own printed drop logs, not recomputed independently for this document.

### 6.2 Remaining checks

- §2.5 (low-variation filter): confirmed **not implemented** as its own stage — worth deciding
  whether one is needed, given `whcl`/`frost_hollow_flag`'s low cardinality passed through
  unfiltered on other grounds.
- §5's manifest/`torch_data.py` circular dependency is a structural risk for any *future* rename,
  not just a one-off historical bug — needs a decision (documented manual step vs. code change)
  before the next set-name change.
- RSQ2 (`xgb_environmental.py`'s `FEATURE_SETS`) and RSQ3 (`broad_environmental_check.py`) are
  **not yet wired** to read from the manifest the way RSQ1 now is — confirmed by direct grep,
  not assumed.
- `dist_to_cpmt_boundary`'s own distinct-value count was not independently re-queried for this
  document (reported via its role in the `dist_to_scpt_boundary`/`dist_to_block_boundary` family
  instead) — needs checking if it is cited independently elsewhere.

### 6.3 Actions required before rerunning results

1. Add RSQ2's new entries to `xgb_environmental.py`'s `FEATURE_SETS` (or call `fit_with_columns`
   directly with a manifest-loaded list — no dict change is strictly required there).
2. Write an RSQ3 equivalent of `run_scope()` in `broad_environmental_check.py`, and a
   corresponding fix in `gnnwr_check.py`'s `build_scope_table()`, before RSQ3's new Sets can be
   fit at all.
3. Decide and implement a fix for the manifest/`torch_data.py` circular dependency (§5, §6.2)
   before the next set-name change, or document the manual-bridge step as a required procedure.
4. Any DNN/PINN run already using RSQ1's `nested_set2_top5`/`nested_set3_gated_terrain_wind`/
   `nested_set4_gated_all` (the pre-rename names) must be re-run — those names no longer exist in
   `ENV_TERRAIN_FEATURE_SETS`. No such runs are known to exist yet.

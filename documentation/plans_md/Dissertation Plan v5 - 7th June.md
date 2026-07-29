---
tags: []
parent: ""
collections:
    - 'Progress/summary notes'
$version: 3811
$libraryID: 1
$itemKey: A8S399ZC

---
# Dissertation Plan v5 - 7th June

## Environmental Drivers of Spatial Variation in Sitka Spruce Growth Trajectories

### A Wind and Terrain Attribution Study Using 21 Years of LiDAR Data from Aberfoyle Forest

***

**This is a living plan, not a fixed contract.** Specific data sources, methods, deliverables,
and even chapter structure below are the current best guess, made with the information
available at the time -- not commitments to defend unchanged. Several have already been revised
after real evidence contradicted them (e.g. HadUK-Grid's assumed "~2-3 cells" and James
Hutton's assumed "~3-5 polygons," both disproven by actual extraction; see
`documentation/experiment_log.md` and the environmental-data notebooks for the full record).
Treat any specific claim here as a hypothesis to check against real data/analysis, not settled
fact -- if evidence points to a different variable, method, or deliverable being more suitable,
the plan should change to match the evidence, not the other way around.

The most recent example (28-29 July 2026): this plan's own "resolved" decision to use
`Top_Height99` as the main response, and to keep `yldc` as a candidate feature, were both
overturned by real evidence found during routine review -- `Top_Height99` was retired in favour
of raw `elev_percentile_95th`, and `yldc` was removed as a feature everywhere after a real
held-out ablation showed it hurts generalisation in every model tested (not from a theoretical
concern -- see `documentation/progress_notes.md`'s "Systematic rebuild" entry for the full
reasoning and numbers). Every `Top_Height99` reference below this point describes the plan as it
stood before that decision -- read it as historical context for how the target was originally
chosen, not as the current target.

***

## One-Line Summary

> This dissertation investigates why some Sitka spruce plots in Aberfoyle consistently grow faster or slower than a standard biological growth model predicts, using up to 21 years of repeated LiDAR-derived plot attributes, and builds a physics-informed neural network that learns plot-specific growth ceilings from terrain and wind exposure data.

***

## Plain English Description

This dissertation investigates why some forest plots in Aberfoyle (Trossachs National Park, Scotland) consistently grow faster or slower than a standard biological growth model predicts, using repeated airborne LiDAR-derived plot attributes across six scan years (2002–2023).

The core idea is that the Chapman-Richards (CR) growth equation — a standard forestry formula describing how trees grow over time — uses a single global growth ceiling for all plots. In reality, an exposed ridge plot and a sheltered valley plot will have fundamentally different maximum heights. This dissertation builds a Physics-Informed Neural Network (PINN) that learns a plot-specific growth ceiling from terrain and wind data, replacing that single global value.

The analysis runs in three stages. First, CR residuals (the gap between observed growth and what the formula predicts) are computed for every plot and timestamp, and terrain features — elevation, slope, wetness index, wind exposure — are extracted from OS mapping data. Spatial analysis using Moran's I and trajectory classification identifies whether underperforming plots cluster in particular parts of the forest. Second, XGBoost with SHAP analysis identifies which terrain and wind features best explain persistent growth anomalies — this acts as an interpretable, independent check before any neural network is trained. Third, the environmentally-conditioned PINN (Env-PINN) is trained in progressively richer versions, with the sub-network's learned feature importance compared against the SHAP ranking as a convergent validity test.

The primary output is a spatial map of learned growth ceilings across Aberfoyle, showing which parts of the forest are environmentally constrained and why. This forms a practical attribution layer for a wider Digital Twin pipeline for forest management.

***

## Accuracy Review: Key Corrections to the 29 June Plan

This plan is broadly sound as a growth-attribution study, but several data details need tightening before they appear in the dissertation.

| Plan detail                                          | Accuracy status                                           | Correction / safer wording                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---------------------------------------------------- | --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| "Drone-based LiDAR"                                  | Too specific unless confirmed by Forest Research metadata | Use "airborne LiDAR-derived plot attributes" or "LiDAR-derived plot metrics supplied by Forest Research". Do not claim drone/UAV unless the acquisition platform is documented.                                                                                                                                                                                                                                                                                                                         |
| Raw row count                                        | Source-dependent                                          | The legacy 29 June CSV has 1,152,801 data rows and 23 columns. The current 7 July GeoPackage has 795,645 rows in`LiDAR_Years`and 376,080 rows in`LiDAR_Years_SS`. State which source each number refers to.                                                                                                                                                                                                                                                                                             |
| Main analysis uses all six timestamps                | Partly true, but only for a smaller balanced cohort       | The current cleaning audit gives two balanced Sitka cohorts: 2008/2012/2021/2023 with 71,766 plots, and 2002/2006/2008/2012/2021/2023 with 13,897 plots. Use the four-survey cohort as the main spatial analysis unless a six-survey-only question requires the smaller cohort.                                                                                                                                                                                                                         |
| "Species variation negligible"                       | Needs evidence                                            | Filter to Sitka spruce using`spis == "SS"`rather than assuming species variation is negligible. The audit notes that`spis`and`Species`disagree, and`spis`is the better Forest Research inventory code.                                                                                                                                                                                                                                                                                                  |
| "Duplicate plot/year pairs: keep higher CanopyCover" | Too arbitrary                                             | Prefer the cleaned GeoPackage / audited cohort logic. If duplicate handling is still needed for the legacy CSV, document a reproducible rule and run sensitivity checks; do not treat higher canopy cover as automatically authoritative.                                                                                                                                                                                                                                                               |
| `plyr = 0`and negative ages                          | Now better resolved                                       | `Age = LiDAR_year - plyr`. Valid planting-year and plausible-age filtering is already part of the cleaned cohort logic; do not frame`Age = LiDAR_year`as a modelling option.                                                                                                                                                                                                                                                                                                                            |
| Top-height variable                                  | Resolved (28-29 July 2026, supersedes the `Top_Height99` decision below)            | Main response is raw, unadjusted `elev_percentile_95th` (a 95th-percentile LiDAR return height, no correction applied). `Top_Height99` (`=elev_percentile_99th`) and its whole family (`Vol99`, `GYCspec99`) are retired entirely. `Top_Height95` (`=elev_percentile_95th × 1.1`) is kept only as the ingredient for the pre-computed `Vol95`/`GYCspec95` forestry-audit fields, never a target or feature. See `documentation/progress_notes.md`'s "Systematic rebuild" entry for the full reasoning. |
| Height-derived predictors                            | Important leakage risk                                    | Exclude`Vol95`,`Vol99`,`Vol_RM95`,`GYCspec95`,`GYCspec99`, raw height percentiles, and`Top_Height95`from top-height predictor sets. They are derived from height and/or age, or are the alternate height target.                                                                                                                                                                                                                                                                                        |
| `whcl`                                               | Not raw wind exposure                                     | `whcl`is a Forest Research inventory windthrow hazard class, likely management-linked. Keep for audit/stratification, not baseline environmental modelling.                                                                                                                                                                                                                                                                                                                                             |
| Soil and climate exclusion                           | Correct direction, but soften the claim                   | Coarse soil/climate products may still support broad covariate adjustment or temporal indices, but they should not be sold as plot-level spatial attribution unless unique within-forest variation is demonstrated after extraction.                                                                                                                                                                                                                                                                    |


**Recommended revised data strategy:** make the four-survey balanced Sitka cohort (2008, 2012, 2021, 2023; 71,766 plots) the main spatial-attribution dataset, and use the six-survey balanced cohort (13,897 plots) for sensitivity checks and questions that genuinely need the full 2002-2023 span.

***

## Aberfoyle Forest: Study Area Statistics

| Property                  | Value                                                                                                                                                                                                                             | Notes                                                                                                                                                                                                                                                                               |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Location                  | Aberfoyle, Stirlingshire, Scotland                                                                                                                                                                                                | Part of Queen Elizabeth Forest Park, Trossachs National Park                                                                                                                                                                                                                        |
| OS National Grid centre   | ~NN 520 010                                                                                                                                                                                                                       | X: 235,000–255,000, Y: 692,500–707,500 (from LiDAR data extent)                                                                                                                                                                                                                     |
| Approximate forest extent | ~20km × 15km                                                                                                                                                                                                                      | Non-uniform plot distribution; clustered by compartment                                                                                                                                                                                                                             |
| Elevation range           | ~25m (valley floor) to ~500m+ (ridges)                                                                                                                                                                                            | Forested plots typically 25–400m                                                                                                                                                                                                                                                    |
| Terrain character         | Steep-sided glens, exposed ridges, lochs                                                                                                                                                                                          | Highland Boundary Fault geology; glacially scoured U-shaped valleys                                                                                                                                                                                                                 |
| Plot/grid-cell size       | Confirmed: adaptive 20m/40m grid, clipped to sub-compartment boundaries via an R`sf`workflow (fine 20m grid intersected with the study area; full 20m cells kept, remainder filled by a coarser 40m grid; slivers <250m² dropped) | Plot geometry is a`MultiPolygon`grid cell per plot in`EPSG:27700`(OSGB36 / British National Grid); plot centroid = grid-cell centre, extracted once per plot in`models/common/export_coordinates.py`                                                                                |
| Legacy raw rows           | 1,152,801                                                                                                                                                                                                                         | 23 columns; legacy`LiDAR_Years_All_attributes.csv`from 29 June                                                                                                                                                                                                                      |
| Current GeoPackage rows   | 795,645 total; 376,080 Sitka spruce                                                                                                                                                                                               | `data/raw/LiDAR_Years_All_7jul.gpkg`; layers`LiDAR_Years`,`LiDAR_Years_SS`                                                                                                                                                                                                          |
| Cleaned balanced cohorts  | 71,766 plots / 287,064 rows; 13,897 plots / 83,382 rows                                                                                                                                                                           | Four-survey and six-survey Sitka cohorts from cleaning audit                                                                                                                                                                                                                        |
| Dominant species          | Sitka spruce (*Picea sitchensis*)                                                                                                                                                                                                 | Filter with`spis == "SS"`; do not rely on`Species`lookup field                                                                                                                                                                                                                      |
| LiDAR scan method         | Airborne LiDAR-derived plot attributes                                                                                                                                                                                            | Acquisition platform to confirm; Top Height derived from elevation percentiles                                                                                                                                                                                                      |
| Timestamps available      | 2002, 2006, 2008, 2012, 2021, 2023                                                                                                                                                                                                | Six timestamps, five inter-scan intervals                                                                                                                                                                                                                                           |
| Total temporal span       | 21 years (2002–2023)                                                                                                                                                                                                              |                                                                                                                                                                                                                                                                                     |
| Main response variable    | `elev_percentile_95th`                                                                                                                                                                                                                    | Raw, unadjusted 95th-percentile LiDAR height (28-29 July 2026 decision — `Top_Height99` retired). `Top_Height95` (`= elev_percentile_95th × 1.1`) kept only for the `Vol95`/`GYCspec95` audit fields; avoid height-derived predictors                                                                                                                     |
| CR parameters             | Fitted (bounded`curve_fit`,`y_max`≥ observed max height)                                                                                                                                                                          | Four-survey:`y_max=53.4909, k=0.010582, p=0.830202`. Six-survey:`y_max=46.5132, k=0.023444, p=1.146632`. Fitted on the training split only (`plot_level_split`, 60/20/20);`Age >= 20`and yield class 2–50 filters applied first. See`outputs/chapman_richards/<cohort>/params.json` |


*Note: use the 2008–2023 four-survey balanced Sitka cohort as the main spatial analysis unless later coverage checks show the six-survey cohort is large and spatially representative enough. Earlier timestamps (2002–2006) are valuable for sensitivity checks and temporal sub-questions, but the balanced six-survey cohort is much smaller.*

### Key spatial data resolutions over Aberfoyle

*(Numbers to be verified against actual data — approximate figures shown) 408681298.985 m² (max area likely alot less than this)*

| Dataset                            | Resolution / scale                                       | Cells covering Aberfoyle (~300km²) | Varies within forest?                                                              |
| ---------------------------------- | -------------------------------------------------------- | ---------------------------------- | ---------------------------------------------------------------------------------- |
| OS Terrain 50 DTM                  | 50m post spacing                                         | ~120,000 cells                     | **Yes — dramatically**                                                             |
| WASP wind speed                    | ~200m–1km                                                | ~300–7,500 cells                   | **Yes — meaningfully**                                                             |
| Global Wind Atlas                  | 250m output grid; heights include 10, 50, 100, 150, 200m | ~4,800 cells over 300km²           | **Yes — useful**, but climatological and wind-resource oriented                    |
| TOPEX (derived from DTM)           | 50m if derived from OS Terrain 50                        | ~120,000 cells                     | **Yes — meaningfully**                                                             |
| HadUK-Grid (temperature, rainfall) | 1km                                                      | ~300 cells over 300km²             | Moderate spatial coverage, but may be too smooth/coarse for plot-level attribution |
| ERA5-Land                          | 9km                                                      | ~3–4 cells                         | **No**— entire forest in ~1 cell                                                   |
| CEH / UK soil products             | Product-dependent; often coarse support                  | To verify after extraction         | Use only if extracted values show meaningful variation                             |
| James Hutton 1:250k soil map       | 1:250,000                                                | ~3–5 polygons total                | **No**                                                                             |


This table justifies the data selection: DTM derivatives and project-specific WASP are the strongest candidates for genuine within-forest spatial attribution. Global Wind Atlas is a useful public wind-climatology fallback/comparison layer, especially if WASP is delayed or undocumented, but it should not be treated as a stand-level canopy wind model. For any additional data source (soil, climate, pest/infection) to be useful for **plot-level spatial** attribution, it must produce enough unique, spatially credible values inside Aberfoyle after extraction. The practical test is empirical: map the extracted values, count unique grid/polygon values, and check whether neighbouring plots simply inherit the same covariate.

***

## What This Dissertation Argues

The dissertation is a **growth attribution study**, not a prediction accuracy competition. The central argument is:

> A standard CR growth model treats every plot in Aberfoyle identically. This dissertation identifies the environmental drivers of why plots differ — using terrain and wind data — and builds a PINN that learns a plot-specific growth ceiling to reflect those differences. Simpler models (XGBoost + SHAP) establish which features matter independently, and the PINN is validated by checking whether it agrees.

The "I didn't just implement one thing" evidence comes from:

1.  Three versions of the model, each given more environmental information — starting with terrain shape only (hills, valleys, slope), then adding wind exposure. Each version is compared to see how much the extra information helps.
2.  Testing how sensitive the model is to its settings — specifically how much weight it gives to the biological growth rules versus the raw data, and whether removing the CR constraint (reducing it to a standard neural network) hurts performance.
3.  Comparing the final model against simpler baselines (the CR formula alone, linear regression, and XGBoost) to show the physics constraint is doing something useful.
4.  A feature importance check using SHAP analysis — asking "which inputs is the model actually paying attention to?" — and verifying the answer matches what forestry science says should matter (wind exposure, terrain wetness, elevation).

***

## What This Dissertation Is NOT Claiming

*   **Not** a study of Sitka spruce growth at national or species scale — findings are Aberfoyle-specific
*   **Not** a prediction accuracy competition — predictive performance is reported as validation, not the goal
*   **Not** a complete Digital Twin — this is a predictive subcomponent for a future DT pipeline

***

## Research Questions

**Core question:** What environmental and terrain factors explain why some plots in Aberfoyle consistently grow faster or slower than the Chapman-Richards biological baseline predicts?

**Sub-Question 1 (Spatial — the main story)** Which permanent plot-level characteristics explain why some plots consistently over- or under-perform the CR growth curve?

*   **SQ1-A:** Do plots that grow faster or slower than expected cluster together spatially, or are they scattered randomly? (Moran's I on CR residuals)
*   **SQ1-B:** Can plots be grouped into persistent over-performers, conformant plots, and persistent under-performers based on their residuals across timestamps? (Trajectory classification)
*   **SQ1-C:** Which terrain and wind features — slope, elevation, wetness, wind exposure — best predict whether a plot over- or under-performs? (XGBoost + SHAP feature importance)
*   **SQ1-D:** Do some features have a tipping point — for example, does growth only suffer above a certain wind exposure level, rather than declining gradually? (SHAP dependence plots)
*   **SQ1-E:** Does the model's own internal sense of which features matter most agree with the independent SHAP ranking? (Convergent validity)

**Sub-Question 2 (Temporal — secondary, if time and data allow)** How does growth change between scan years, and can the model predict growth accurately across large time gaps?

*   **SQ2-A:** How much did plots grow in each inter-scan interval, and which interval showed the most variation across the forest? (Interval-level growth comparison — descriptive)
*   **SQ2-B:** Does the model predict growth better across short gaps (2 years: 2006→2008, 2021→2023) than long ones (9 years: 2012→2021)? (Prediction accuracy by gap length)
*   **SQ2-C:** Does knowing a plot's terrain help the model stay accurate over long gaps — does environmental conditioning reduce the drop-off in accuracy for the 2012→2021 interval? (Env-PINN vs baseline PINN split by gap length)

*Note: with only five inter-scan intervals, temporal attribution is limited. SQ2 is framed around model evaluation across gap lengths rather than climate causation, which the sparse timestamps cannot support.*

**What would actually strengthen your causal claims:**

**1. Partial regression / residual-on-residual plots**\
After removing the effect of spatial location (via coordinates), does terrain still predict residuals? If yes, that's stronger evidence the terrain feature itself matters rather than just proximity.

**2. Moran's I on model residuals (not just raw residuals)**\
After fitting XGBoost, run Moran's I on the *remaining* unexplained residuals. If spatial autocorrelation drops substantially, your features are genuinely capturing spatial structure. If it stays high, the model is missing something important.

**3. Geographically Weighted Regression (GWR)**\
Instead of one global model, GWR fits a separate regression at each plot location, allowing relationships to vary across space. This is specifically designed for small geographic areas with spatially varying processes. It would show *where* in Aberfoyle wind matters most versus where elevation matters most — much more interesting than a single global coefficient.

**4. Gaussian Process regression**\
A GP explicitly models spatial correlation as a function of distance. You can decompose variance into "explained by terrain features" versus "explained by spatial proximity alone." This directly addresses your worry — it separates genuine feature effects from mere spatial clustering.

### DAG to think about

Climate (interval-level)

       ↓

   \[unobserved]

Soil moisture ←── TWI (terrain) ──→ Wind exposure ←── Elevation

       ↓                                                                  ↓                                   ↓

       └────→ Growth rate ←─────┘                                    │

                                   ↓                                                                             │

                             Top Height ←───── Age ←────────┘

                                   ↓

                   LAI / Canopy Cover / Volume

                        (consequences)

1.  **LAI, canopy cover, volume are consequences of height, not causes** — they're downstream of the same growth process. Do not use them as predictors of height or residuals, or you'll have a circular model.
2.  **Age is a cause of height but a weak proxy** because it's confounded by stand history (felling, replanting) and environmental suppression
3.  **TWI, elevation, wind are upstream causes** — they determine the growth ceiling the plot can achieve
4.  **Thinning is an unobserved confounder** — it simultaneously affects age records, height, LAI, and canopy cover in ways that look like environmental effects

***

**Additional causal considerations worth flagging**

*   **Species composition** — your data is mostly Sitka but any broadleaf or mixed plots will break the age-height relationship in a different way
*   **Planting density** — denser planting produces competition, suppressing individual tree height even in good environments; this would show up as low height despite good terrain
*   **Previous land use** — agricultural land converted to forestry often has better soil nutrition than upland blanket planting, producing over-performance relative to CR predictions
*   **Aspect × elevation interaction** — a north-facing slope at high elevation is much worse than a south-facing slope at the same elevation; these interact rather than acting independently

***

## Data Inventory

### LiDAR Data (Primary)

Six timestamps: **2002, 2006, 2008, 2012, 2021, 2023**. There are two relevant local sources:

| Source                   | Scope                                                                                    |                        Rows / columns | Notes                                                                                                                 |
| ------------------------ | ---------------------------------------------------------------------------------------- | ------------------------------------: | --------------------------------------------------------------------------------------------------------------------- |
| Legacy CSV               | `legacy/legacy_code_30june/data/LiDAR_Years_All_29thjune/LiDAR_Years_All_attributes.csv` |            1,152,801 rows; 23 columns | Original 29 June working extract.                                                                                     |
| Current GeoPackage       | `data/raw/LiDAR_Years_All_7jul.gpkg`, layer`LiDAR_Years`                                 | 795,645 rows; 40 fields plus geometry | Updated working source with derived`Top_Height95`,`Top_Height99`, volume, GYC, age, thinning, and wind-hazard fields. |
| Current Sitka layer      | `data/raw/LiDAR_Years_All_7jul.gpkg`, layer`LiDAR_Years_SS`                              |                          376,080 rows | Filtered Sitka spruce layer.                                                                                          |
| Clean four-survey cohort | 2008, 2012, 2021, 2023                                                                   |            71,766 plots; 287,064 rows | Recommended main spatial-attribution cohort.                                                                          |
| Clean six-survey cohort  | 2002, 2006, 2008, 2012, 2021, 2023                                                       |             13,897 plots; 83,382 rows | Use for full-span sensitivity and temporal questions.                                                                 |


| Interval    | Gap  | Notes                                                 |
| ----------- | ---- | ----------------------------------------------------- |
| 2002 → 2006 | 4 yr | 2003 eastern Scotland drought in this window          |
| 2006 → 2008 | 2 yr | Short interval, good signal-to-noise baseline         |
| 2008 → 2012 | 4 yr | Medium gap, no major documented disturbance           |
| 2012 → 2021 | 9 yr | Storm Frank 2015–16; multiple warm years; longest gap |
| 2021 → 2023 | 2 yr | **Held-out test set — never seen during training**    |


### Static Spatial Data (Terrain and Wind Only)

| Dataset           | Variables                                                | Resolution       | Source                               | Access                  |
| ----------------- | -------------------------------------------------------- | ---------------- | ------------------------------------ | ----------------------- |
| OS Terrain 50 DTM | Elevation, slope, northness, eastness, TWI, TOPEX        | 50m              | Ordnance Survey OpenData             | Free                    |
| WASP wind atlas   | Mean wind speed / exposure layer                         | Confirm          | Via Dr. Suárez-Minguez               | Project contact         |
| Global Wind Atlas | Mean wind speed / wind power density at standard heights | 250m output grid | DTU Wind Energy / World Bank / ESMAP | Free web + GIS download |


**Why soil data is not in the core model by default:** available soil products may be too coarse, polygonal, or model-derived to support plot-level attribution inside one forest. This should be tested empirically rather than assumed: extract candidate soil layers, count unique values, map them, and include them only if they vary meaningfully across the cleaned Sitka cohort. Terrain-derived TWI remains the defensible default proxy for drainage/topographic moisture.

**Why climate spatial features are excluded from the core spatial model:** HadUK-Grid is a strong temporal climate source, but at 1km it is still coarse relative to plot/grid-cell scale and meteorological station interpolation may smooth local terrain effects. Use it for interval-level indices first. Include spatial climate covariates only after showing they add credible within-forest variation beyond elevation, aspect, TWI, and wind exposure.

**Global Wind Atlas and TOPEX as WASP fallback:** If project-specific WASP wind data is unavailable or poorly documented, use Global Wind Atlas as a public 250m wind-climatology layer and TOPEX as a terrain-derived shelter/exposure metric. GWA is valuable because it provides a spatially varying wind surface, but TOPEX remains useful because it is derived directly from the same terrain surface used for the plot features.

*Soil, climate, pest/infection data could be added if available at sufficient granularity. Replace the hard "\~100m" rule with an empirical inclusion rule: after joining to plot centroids, a source must show enough unique values and spatial structure inside Aberfoyle to explain differences between nearby plots.*

### Temporal Climate Data (Forest-Level Only)

| Source                    | Variables                                                       | Use                                                         |
| ------------------------- | --------------------------------------------------------------- | ----------------------------------------------------------- |
| HadUK-Grid                | Daily rainfall, temperature, frost days (1km)                   | Interval-level climate indices                              |
| CHESS-met / PET successor | Daily meteorology and PET/SMD inputs where temporally available | Soil Moisture Deficit (SMD); confirm coverage for 2021–2023 |


Derived indices per inter-scan interval (one value per interval, applied to all plots):

| Index                    | Ecological meaning                     |
| ------------------------ | -------------------------------------- |
| Summer SMD               | Drought stress — primary Sitka limiter |
| Annual GDD               | Growing season length                  |
| Spring frost count       | Shoot damage at bud burst              |
| Max consecutive dry days | Extreme drought proxy                  |
| Winter precipitation     | Soil recharge proxy                    |
| Temperature anomaly      | Warming signal vs 2002–2023 baseline   |


### Response Variable: CR Residual

\$\$\varepsilon\_{i,t} = y\_{i,t}^{\text{obs}} - y\_{\text{CR}}(t\_i \mid y\_{\max}, k, p)\$\$

Main observed response: `elev_percentile_95th` (raw, unadjusted — `Top_Height99` retired, `Top_Height95` kept only for the `Vol95`/`GYCspec95` audit fields). Parameters are optimised from the cleaned dataset via bounded least squares (`scipy.optimize.curve_fit`, `y_max` constrained strictly above the observed max training height by a 0.1% buffer — a pre-existing bug where the bound was exactly the observed max, letting the fit land precisely on it, was found and fixed 28-29 July 2026) using `Age` as the biological time input, fitted on the training split of `plot_level_split()` only.

Fitted values so far (four-survey and six-survey cohorts, filtered to `Age >= 20` and yield class 2–50):

| Cohort      | y\_max  | k        | p        |
| ----------- | ------- | -------- | -------- |
| Four-survey | 53.4909 | 0.010582 | 0.830202 |
| Six-survey  | 46.5132 | 0.023444 | 1.146632 |


For comparison, the prior dissertation's fit (Lynch, 2025) was `y_max=46.1126, k=0.01866979, p=1.0175`. Both cohorts here land at a noticeably higher `y_max`, tied to Top\_Height99 rather than Top\_Height95 and to the new bounded-fit constraint — worth a methods note rather than treating the values as directly comparable.

Positive residual = taller/faster than expected for age; negative = suppressed relative to the age-matched CR baseline. Avoid using `GYCspec95/99`, volumes, or raw height percentiles as predictors because they are derived from height and/or age.

***

## Data Audit: Issues to Resolve Before Modelling

From the 29 June legacy CSV and the 7 July GeoPackage audit. Send remaining question register to Dr. Suárez-Minguez / Forest Research before final modelling.

| Issue                                                         | Current decision                                                                                                                                                                                                                                                                                                                                                            |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Duplicate plot/year pairs in legacy CSV                       | Prefer current GeoPackage / cleaned cohort logic. If using legacy CSV, apply a reproducible rule and sensitivity check; do not assume higher canopy cover is automatically correct.                                                                                                                                                                                         |
| Planting year / age problems                                  | Use documented`Age = LiDAR_year - plyr`; keep only valid planting years and plausible ages in balanced modelling cohorts.                                                                                                                                                                                                                                                   |
| Species disagreement (`spis`vs`Species`)                      | Use`spis == "SS"`for Sitka spruce filtering because it is the Forest Research inventory species code.                                                                                                                                                                                                                                                                       |
| Negative volume                                               | Do not use volume for height-model cleaning. Volume is height-derived and should be filtered only in volume-specific analyses.                                                                                                                                                                                                                                              |
| Height-derived/leakage-prone variables                        | Exclude`Vol95`,`Vol99`,`Vol_RM95`,`GYCspec95`,`GYCspec99`, raw height percentiles, and`Top_Height95`from top-height predictors.                                                                                                                                                                                                                                             |
| `whcl`windthrow hazard class                                  | Keep for audit/stratification; exclude from baseline environmental predictors because it is inventory/management-linked, not raw wind exposure.                                                                                                                                                                                                                             |
| Management fields (`Thin`,`last_thinn`,`time_since_thinning`) | Useful contextual covariates and confounder flags, but not substitutes for terrain/wind attribution.                                                                                                                                                                                                                                                                        |
| Non-standard plot areas / edge polygons                       | Confirm geometry interpretation; avoid deleting solely for area unless the cleaned cohort logic or Forest Research metadata requires it.                                                                                                                                                                                                                                    |


**Strategy:** use the audited balanced Sitka cohorts for main modelling, retain wider master exports for sensitivity and diagnostics, and document any exclusions at plot level so temporal trajectories remain internally consistent.

***

## Implementation Status (as of 13 July 2026)

A standing status marker, updated as work lands — reflects what has actually been built in the `forest_diss` repository, ahead of the Work Plan tiers below.

**Repository structure.** `models/` package with one folder per model (`chapman_richards/`, `average_by_age/` implemented; `linear_baseline/`, `rf_baseline/` scaffolded, not yet implemented) plus `models/common/` for shared, reusable infrastructure (`metrics.py`, `splits.py`, `plotting.py`, `saving.py`, `data.py`, `geo.py`). `models/baselines/` holds the cross-model fit/evaluate orchestration scripts (`run_baselines.py`, `evaluate_baselines.py`). Outputs land in `outputs/<model_name>/<cohort>/` (fitted params or lookup table, `metrics.json`, `predictions.csv`); split assignments in `outputs/splits/<cohort>/`.

**Data pipeline (Tier 1, done).** `data_processing/clean_master_data.py` (a plain script, replacing the original cleaning notebook's export logic 28-29 July 2026 — the notebook is retired for this step) exports the balanced four-survey (71,766 plots) and six-survey (13,897 plots) Sitka cohorts to `data/processed/{master,current_state,transitions}/`, with raw `elev_percentile_95th` as the primary target (`Top_Height99` retired, `Top_Height95` kept only for `Vol95`/`GYCspec95` audit fields — see the Accuracy Review row above). `data_processing/export_model_tables.py` then builds ONE consolidated `model_table.parquet` per cohort (replacing 5 near-duplicate per-model files, two of which — `dnn_noenv`/`pinn_noenv` — were confirmed byte-for-byte identical). Plot centroids (one per plot, `EPSG:27700`) are extracted separately from the raw GeoPackage's grid-cell polygons via `models/common/export_coordinates.py`, confirming the plot/grid-cell size question in the Study Area Statistics table above.

**Split infrastructure (ahead of Tier 2 / section 4.4).** `models/common/splits.py` already implements all three split types the later chapters need: `plot_level_split()` (60/20/20, currently used by the CR/average-by-age baselines), `spatial_block_split()` (compartment-based, size-aware block assignment, with a programmatically-verified `buffer_distance` exclusion zone — see `data_exploration_gpkg/notebooks/spatial_temporal_split_visualisation.ipynb` for maps and a pros/cons discussion), and `temporal_split()` (year-based, for the SQ2 gap-length questions). Only `plot_level_split()` is wired into a model so far; the other two are implemented and tested against real 4survey/6survey data (`models/common/test_splits.py`) but not yet used for training, since no model uses terrain/wind features or does temporal generalisation testing yet.

**Baselines (Tier 2, partial).** CR and average-by-age are fitted and evaluated on both cohorts (filters: `Age >= 20`, yield class 2–50). Fitted CR parameters and test-set metrics are folded into the relevant sections above (Study Area Statistics, Response Variable, section 4.3). Linear regression, RF, XGBoost, and all PINN work have not started.

***

## Work Plan

Work is ordered from safest to most ambitious. Each tier produces results worth submitting even if later tiers are not completed.

**Tier 1 — Data preparation and exploratory analysis (Days 1–9).** Clean the raw dataset (flag duplicates, bad ages, extreme heights), match plots across timestamps in QGIS, compute CR residuals for every plot-timestamp pair, and extract terrain and wind features (elevation, slope, TWI, TOPEX, WASP or Global Wind Atlas) via rasterio. Run Moran's I to check whether residuals cluster spatially, and classify plots into persistent under-performers, conformant, and over-performers based on their residual history.

**Tier 2 — Simple models and baselines (Days 10–15).** Fit CR and linear regression baselines first as a sanity check. Then run XGBoost in two versions (terrain-only, then terrain + wind) with SHAP analysis to identify which features best explain growth residuals — this ranking directly determines what goes into the Env-PINN sub-network. Finally train PINN Version 1 (no environmental conditioning) on 2002–2012, tested on 2021–2023.

**Tier 3 — Env-PINN (Days 16–26).** Add an environmental sub-network that learns a plot-specific growth ceiling \$\hat{y}\_{\max}(\mathbf{e})\$ from terrain and wind features, replacing the single global value. Train two versions (terrain-only inputs, then full SHAP-selected inputs). Run ablation experiments varying the physics weight and anchor regularisation strength. Produce a spatial map of learned growth ceilings and compare the sub-network's feature importance against the independent SHAP ranking as a convergent validity check.

**Tier 4 — Optional extensions (Days 24+).** Dynamic growth rate conditioning on climate indices, or a transformer/sequence model — only if Tier 3 is complete and a clear research question justifies the added complexity. A transformer treating each plot's growth history as a sequence is only worth pursuing if enough plots have ≥4 clean timestamps; otherwise relegate to Future Work.

***

## Timeline

| Days  | Tasks                                                                | Writing             |
| ----- | -------------------------------------------------------------------- | ------------------- |
| 1–2   | Data audit resolution. Email question register to Dr. Suárez-Minguez | —                   |
| 2–4   | Cleaning pipeline. Flag-and-retain. Produce clean dataset            | Ch. 1 draft         |
| 4–5   | Plot matching across timestamps in QGIS. Noise removal               | Ch. 2 background    |
| 5–7   | CR residuals + EDA. Box plots, correlations, VIF                     | Ch. 3 data sections |
| 7–8   | Feature extraction: TWI, slope, TOPEX, WASP/GWA via rasterio         | Ch. 3 data sections |
| 8–9   | Moran's I + trajectory classification. Spatial map                   | Ch. 3 complete      |
| 10–12 | CR + linear regression baselines. Sanity check before proceeding     | Ch. 4 methodology   |
| 12–13 | XGBoost A and B + SHAP. Feature selection for Env-PINN               | Ch. 4 methodology   |
| 13–15 | PINN Version 1 (CR-PINN baseline, 6 timestamps)                      | Ch. 4 PINN sections |
| 16–20 | Env-PINN Version 2 (terrain-conditioned$y\_{\max}$)                  | Ch. 5 drafting      |
| 20–23 | Env-PINN Version 3 (full SHAP features) + ablations                  | Ch. 5 drafting      |
| 23–26 | Attribution outputs, spatial map, convergent validity test           | Ch. 5 complete      |
| 26–28 | Ch. 6 Discussion. Ch. 7 Conclusions                                  | Priority writing    |
| 28–30 | Proofreading, figures, appendices, references                        | Final polish        |


*Start writing Chapters 1–3 from Day 3 in parallel — don't wait for experiments to finish.*

***

## Full Chapter Structure

### Chapter 1: Introduction

**1.1 Motivation** Prior work has established that a CR-PINN can outperform benchmark ML models for temporal growth prediction across large data gaps in Aberfoyle (Lynch, 2025), but spatially coherent anomalies — neighbouring plots with dramatically different growth rates — remain unexplained. The global CR parameter assumption (a single \$y\_{\max}\$ for every plot) cannot represent the reality that exposed ridge plots and sheltered valley-floor plots have fundamentally different growth ceilings. This dissertation investigates the environmental drivers of that spatial variation using the full six-timestamp LiDAR dataset.

The Forestry 4.0 shift toward smart forestry and Digital Twin technologies (Buonocore et al., 2022; Jiang et al., 2022) demands prediction tools that practitioners can interrogate — not just accurate predictions but explanations of *why* growth differs between plots (Ercanli & Senyurt, 2024). An environmentally-conditioned PINN that learns plot-specific growth ceilings from terrain and wind exposure provides both.

**1.2 Research questions** (as stated above)

**1.3 Contributions**

1.  First six-timestamp trajectory analysis of Sitka spruce growth in Aberfoyle (2002–2023)
2.  Environmental attribution of persistent growth anomalies using XGBoost + SHAP across two progressively richer feature sets
3.  An environmentally-conditioned CR-PINN (Env-PINN) in which \$y\_{\max}\$ is a learned function of terrain and wind exposure
4.  A systematic PINN ablation study demonstrating that improvements are attributable to environmental conditioning
5.  A convergent validity test between Env-PINN sub-network attribution and independent SHAP feature rankings

**1.4 Scope** Findings are specific to Aberfoyle. Attribution, not accuracy optimisation, is the primary objective. Full Digital Twin implementation is outside scope.

***

### Chapter 2: Background

**2.1 Sitka spruce ecology and growth requirements**

Sitka spruce requires ≥900–1000mm annual rainfall and is primarily limited by soil water deficit rather than atmospheric dryness in Scottish conditions (Morison et al., 2010; Forest Research, 2025). Wind exposure causes shorter, stockier growth through thigmomorphogenesis (Telewski, 2006), and Worrell (1987) found TOPEX — a terrain-derived wind shelter index — was among the strongest predictors of Sitka Yield Class variation across Scottish upland sites, with Yield Class declining 3.2–4.0 m³/ha/yr per 100m elevation gain. Waterlogged soils suppress rooting depth and nutrient uptake (Forest Research, 2025), though within Aberfoyle this is largely terrain-driven and captured by TWI. Spring frost frequency adds a further growth limiter at high elevations (Worrell, 1987).

**2.2 Growth and yield modelling**

The Chapman-Richards (CR) equation (Pienaar & Turnbull, 1973) — a three-parameter sigmoid rooted in Von Bertalanffy (1949) growth theory — is widely used in UK forestry and is the biological baseline for this study. Its key limitation is that fixed global parameters cannot represent plot-level environmental variation, which is the central problem this dissertation addresses. The more complex 3PG process-based model (Landsberg & Waring, 1997; Forrester et al., 2021) incorporates climate explicitly but requires monthly inputs incompatible with Aberfoyle's irregular scan timestamps, making CR the practical choice here.

**2.3 Factors affecting forest growth**

Climate and water availability are the dominant drivers of growth variation in conifer forests (Toledo et al., 2011; Boulanger et al., 2024). Within a single forest like Aberfoyle, terrain mediates these effects: elevation controls temperature and frost exposure, slope and aspect affect radiation receipt, and topographic position determines soil drainage and waterlogging. Wind exposure directly suppresses height growth (Telewski, 2006; Worrell, 1987). The primary unobserved confounder is forest management — thinning events produce abrupt changes in stand metrics that can resemble environmental signals.

**2.4 Machine learning for growth prediction and attribution**

PINNs (Raissi et al., 2019) improve data efficiency in small-data regimes by embedding physical constraints directly into the loss function (Karniadakis et al., 2021), making them well-suited to Aberfoyle's sparse timestamps. CR-PINNs have been applied to Sitka spruce height prediction (Lynch, 2025). For tabular ecological data, XGBoost consistently outperforms neural networks (Everingham et al., 2016), and SHAP analysis (Lundberg & Lee, 2017; Lundberg et al., 2020) provides interpretable feature importance rankings — demonstrated in forest ecology by Brugger et al. (2023). Sequence models (LSTM, transformer) are not appropriate here given only five inter-scan intervals per plot. The Env-PINN combines knowledge-guided network structure with a knowledge-guided loss function, two of the four integration strategies identified in the Global Change Biology review (2025) on theory-guided ML for ecology.

**2.5 Digital Twins in forestry**

DT frameworks for forestry have been proposed by Buonocore et al. (2022) and demonstrated using LSTM-based prediction by Jiang et al. (2022). This dissertation contributes the environmental attribution and growth prediction layer, complementing a companion management interface (Arthur, 2025) and RL-based thinning optimiser (Clark, 2025).

***

### Chapter 3: Data and Feature Engineering

**3.1 LiDAR data** Six timestamps (2002, 2006, 2008, 2012, 2021, 2023) covering Aberfoyle's predominantly Sitka spruce forest. Report both the legacy CSV and current GeoPackage sources so row counts are not confused. The main response is `elev_percentile_95th` (raw, unadjusted 95th-percentile height); `Top_Height95`, `Vol95`, and `GYCspec95` are audit variables, not predictors for height (`Top_Height99` and its `Vol99`/`GYCspec99` family are retired entirely — see the Accuracy Review row above). Top Height distributions are visualised across all six scans to show the 21-year growth evolution, alongside a gap structure table noting key climate events per interval.

**3.2 Plot matching and cleaned cohorts** Use the audited plot-level cleaning logic. The main spatial cohort is the balanced Sitka spruce four-survey set (2008, 2012, 2021, 2023; 71,766 plots). The six-survey balanced set (13,897 plots) is used for full-span sensitivity analysis. Missing observations are left as structurally missing rather than imputed in any unbalanced exploratory analysis.

**3.3 CR residuals** CR parameters are optimised from the dataset via least squares and applied to all plot-timestamp pairs to compute residuals \$\varepsilon\_{i,t}\$. Box plots check whether residuals are centred near zero at each timestamp; a histogram of per-plot mean residuals \$\bar{\varepsilon}\_i\$ checks for systematic under-performance. Moran's I on \$\bar{\varepsilon}\_i\$ tests whether anomalies cluster spatially.

**3.4 Static spatial features** Extracted at each plot centroid using rasterio and geopandas from the OS Terrain 50 DTM (50m post spacing): elevation, slope, northness, eastness, TWI (\$\ln(A/\tan\beta)\$), and TOPEX wind shelter index. WASP wind exposure is added where available and after resolution/height definition is confirmed. Global Wind Atlas is added as a public 250m wind-climatology fallback or comparison layer, using the lowest suitable height above ground rather than turbine-hub-height layers unless justified. Soil or climate spatial data is included only if extraction shows meaningful within-forest variation and no obvious leakage or circularity.

**3.5 Temporal climate features** Seven interval-level indices (including summer soil moisture deficit, growing degree days, and spring frost count) are derived from HadUK-Grid daily data and, if temporal coverage is sufficient, CHESS-met / PET products. These are primarily forest-wide temporal signals; they should not be over-interpreted as plot-level spatial drivers unless extracted values demonstrate useful within-forest variation.

**3.6 Multicollinearity** A correlation matrix and VIF analysis identify expected collinear groups (elevation, TOPEX, and WASP all capture aspects of exposure). XGBoost handles this implicitly; the panel regression drops the most correlated variable in any pair with VIF > 10.

***

### Chapter 4: Methodology

*(Ordered simplest to most complex.)*

**4.1 Trajectory classification** Plots with ≥2 timestamps are classified into three groups based on mean CR residual \$\bar{\varepsilon}\_i\$: persistent under-performers (\$\bar{\varepsilon}\_i < -\delta\$), CR-conformant (\$|\bar{\varepsilon}\_i| < \delta\$), and persistent over-performers (\$\bar{\varepsilon}\_i > +\delta\$), where \$\delta \approx 2\$m (tuned from data). Spatial map of classes is the most visually striking figure in the dissertation.

**4.2 Moran's I** Global Moran's I on \$\bar{\varepsilon}\_i\$ using PySAL (k=8 neighbours). Significant positive result justifies spatial attribution. Local Moran's I (LISA) identifies specific clusters.

**4.3 Simple baselines** Implemented so far: CR (Chapman-Richards) and average-by-age, both fitted on a 60/20/20 plot-level split (`plot_level_split()`; `val` is saved for schema consistency with later models but unused by either, since neither has anything to tune), with `Age >= 20` and yield class 2–50 filters applied first. Linear regression is scaffolded (`models/linear_baseline/`) but not yet implemented; RF is scaffolded too. This deliberately differs from the 2002–2012 train / 2021–2023 test framing below — that temporal holdout is reserved for `temporal_split()` and the SQ2-B/C gap-length questions, kept separate from this plot-level sanity-check pass so a spatial-vs-temporal failure cannot be conflated. Baseline test-set metrics (MAE, RMSE, MSE, R², MRE, Accuracy, Bias, plus an age-banded breakdown) are already computed per cohort — see `outputs/{chapman_richards,average_by_age}/<cohort>/metrics.json`. CR shows a large one-signed bias in the oldest age bands (e.g. six-survey 60-80yr band: bias ≈ -11.2m), which is likely a genuine limitation of a 3-parameter curve with a small old-growth sample, not a bug — worth reporting as a baseline weakness the PINN work can improve on.

**Age filter, updated after forester consultation (15 July 2026).** The `Age >= 20` filter used for the baselines above is superseded going forward by a forester-informed rule: young LiDAR-measured top heights are unreliable (top height is "unrelated to age and competition" before the age at which surveyors switch to estimating Yield Class, ≈30), while old-age (70-80+) spread is expected allometric heteroskedasticity, not a data problem, so no upper cap is applied. Implemented as a plot-level condition — every plot must be ≥30 years old by 2023 (`plyr <= 1993`) — rather than filtering every row's own age, since that removes far less data (19.0%/0.9% of rows for four-/six-survey vs. 36.7%/44.8% under a strict row-level `Age >= 30`) and preserves full plot trajectories for the PINN's trajectory-consistency loss. This is a deliberate trade-off, not a free win: it keeps some young-age rows (age <30) that the forester's own reasoning would flag as unreliable, as long as the parent plot is resurveyed for long enough to mature by 2023 — full pros/cons and numbers in `documentation/progress_notes.md` ("Age filter — resolved 15 July 2026").

**4.4 XGBoost spatial attribution**

| Version | Features                  | Purpose                                                     |
| ------- | ------------------------- | ----------------------------------------------------------- |
| XGB-A   | Terrain only              | What does terrain alone explain?                            |
| XGB-B   | Terrain + WASP wind speed | Does measured wind add beyond TOPEX proxy?                  |
| XGB-C   | Terrain + GWA wind speed  | Public-data fallback if WASP is unavailable or undocumented |


Spatial block holdout split (not random — spatially autocorrelated data violates standard CV independence; Roberts et al., 2017), already implemented as `spatial_block_split()`: whole forestry compartments (`cpmt`, 296 units) — not `blk`, which only has 8 wildly uneven-sized units (81 to 29,182 plots) — are assigned to train/val/test via a size-aware greedy assignment that keeps proportions close to the requested 60/20/20 despite compartment sizes ranging from 1 to 1,336 plots. A `buffer_distance` (currently 50m, roughly one grid-cell ring out from a split boundary; tuned down from an initial 100m to reduce data loss) excludes plots within that distance of a plot in a different split, verified programmatically via a KDTree nearest-neighbour search rather than assumed — see `data_exploration_gpkg/notebooks/spatial_temporal_split_visualisation.ipynb` for maps and a pros/cons discussion of the buffer trade-off. Not yet wired into a model, since no model uses terrain/wind features yet. SHAP outputs from XGB-B determine which features enter the Env-PINN sub-network.

**4.5 PINN Version 1 (baseline)** CR-PINN architecture trained on 2002–2012, tested on 2021 and 2023. Global \$y\_{\max}\$, \$k\$, \$p\$ from dataset optimisation. This is the baseline every subsequent version must beat. Confirmed hyperparameters from Lynch (2025): physics weight \$\lambda\_{\text{ph}} = 1.0\$ (the peak from their Fig 4.8 sweep), L1 weight \$\lambda\_1 = 1\text{e-}5\$, batch size 32 for the PINN (smaller batches gave the physics loss more gradient update steps per epoch, which helped convergence — their DNN comparison used a larger batch, not explicitly stated as a single number in the thesis).

**4.6 Env-PINN Versions 2 and 3** An environmental sub-network \$g\_\phi\$ replaces the fixed global \$y\_{\max}\$:

\$\$\hat{y}*{\max}(\mathbf{e}) = g*\phi(\mathbf{e}), \quad g\_\phi: \mathbb{R}^{n\_e} \to \mathbb{R}\_{>0}\$\$

*   **Version 2:** terrain features only (elevation, TWI, TOPEX, northness, eastness)
*   **Version 3:** full SHAP-selected feature set (expected to include WASP wind speed)

Sub-network architecture: 2 × 32 neurons, ReLU, softplus output (ensures \$\hat{y}*{\max} > 0\$). Anchor regularisation \$\lambda*{\text{anc}} \cdot (\hat{y}*{\max}(\mathbf{e}) - \bar{y}*{\max})^2\$ prevents degenerate solutions. Pre-trained on XGB-B residuals for stable joint training.

Full loss: \$\$\mathcal{L}*{\text{total}} = \mathcal{L}*{\text{data}} + \lambda\_{\text{ph}} \cdot \mathcal{L}*{\text{physics}}(\hat{y}*{\max}(\mathbf{e})) + \lambda\_1|\theta|*1 + \lambda\_2|\phi|1 + \lambda{\text{anc}} \cdot \frac{1}{N}\sum\_i(\hat{y}*{\max}(\mathbf{e}*i) - \bar{y}*{\max})^2\$\$

**4.7 Ablation experiments** Physics weight sweep (\$\lambda\_{\text{ph}}\$ across six values), anchor ablation (\$\lambda\_{\text{anc}} \in {0, 0.001, 0.1}\$), and batch size comparison (32 vs 128).

**4.8 Convergent validity test** Sub-network gradient attribution \$\partial\hat{y}\_{\max}/\partial e\_j\$ ranked and compared against SHAP global importance from XGB-B via Spearman correlation.

***

## Loss Function

Both start from the same physical setup: a neural network branch and the CR branch run in parallel on the same inputs, and CR's parameters (`y_max`, `k`, `p` from your Step 2 global fit) stay **frozen** — gradients only ever update the neural network, never the process model. From there they diverge:

**Parallel physics — the two predictions are summed, then compared to the truth once:**

```
ŷ_total = ŷ_NN(x) + ŷ_CR(Age)
Loss = MSE(y_true, ŷ_total)
```

There's a single loss term. Because CR's contribution is fixed, minimising this is mathematically equivalent to training the NN to predict `(y_true − ŷ_CR)` directly — the NN is implicitly learning the residual, without ever being told "residual" is the target. This is the cleanest possible version of "explain what CR misses using covariates."

**Physics regularisation — the two predictions are kept separate and compared to *each other*, not summed:**

```
ŷ_final = ŷ_NN(x)                    ← this is the actual prediction, on its own
Loss = MSE(y_true, ŷ_NN) + λ · Δ(ŷ_NN, ŷ_CR)
```

The NN's own output *is* the final prediction — CR never gets added in. Instead, CR's prediction enters only as a second loss term, pulling the NN's output toward what CR would say, evaluated as what the paper calls a joint likelihood (effectively a soft consistency penalty rather than an additive component). It's a prior on the answer, not part of the answer.

**One thing worth flagging about your existing CR-PINN specifically:** it's actually a *third*, more specific variant of this — your physics loss compares **derivatives** (`d(Height)/d(Age)` from the network vs. from the CR PDE), not raw predicted heights like the paper's generic "process regularisation" does. That's the classic Raissi-style PINN formulation and it's a perfectly legitimate — arguably stricter — regularisation, but it's not literally identical to what Wesselkamp et al. call "physics regularisation" in their taxonomy. Worth a one-line clarification in your methodology chapter so a marker familiar with that paper doesn't think you've mislabelled it.

**What the physics loss actually constrains — not the curve's shape, its speed.** A common misreading is that this penalty pins down the sigmoid's overall shape, especially its flat top (near \$y\_{\max}\$) and flat tail (young trees). It doesn't. At every training step, for every plot, it only checks one thing: *at this plot's current age, is the network's growth rate (how fast predicted height is rising with age) close to what the CR formula's growth rate is at that same age?* It never looks at the predicted height itself, only the slope.

The CR growth-rate curve itself has three regions across age: low near the start (age \~10–20, trees still establishing), a peak in the middle (age \~30–50), and flattening back down as trees approach \$y\_{\max}\$ (age 70+). Since Lynch (2025) filtered to ages 20–80 and Aberfoyle's Sitka spruce is mostly middle-aged, the constraint mostly bites in that fast-growth middle stretch, not at the sigmoid's top or tail. Lynch's own thesis notes the growth-rate curve "looks more linear than non-linear" across that middle age range — which is also the likely reason their physics-weight sweep (\$\lambda\_{\text{ph}}\$) made so little difference to results: a near-linear rate constraint is easy for the network to satisfy regardless of how strongly it's enforced.

**Practically, for parallel physics:** you can build this today with zero new machinery — take your existing frozen CR fit, take your existing DNN, sum their two outputs, train the DNN against that combined MSE. No derivative computation, no PDE term, no change to the DNN's architecture at all.

***

### Chapter 5: Results

**5.1 Trajectory analysis and spatial structure** CR residual box plots per timestamp, Moran's I result, LISA cluster map, and trajectory class spatial map.

**5.2 Spatial attribution** Feature–residual correlation table, XGB-A vs XGB-B comparison (RMSE, R²), SHAP beeswarm and dependence plots for top features.

**5.3 Temporal analysis** Interval-level growth comparison (SQ2-A). Prediction accuracy by gap length — does the Env-PINN handle the 9-year gap better than the baseline PINN? (SQ2-B, SQ2-C).

**5.4 Env-PINN results**

| Model                                 | MAE | MSE | R² | MRE | Accuracy |
| ------------------------------------- | --- | --- | -- | --- | -------- |
| CR (global params)                    |     |     |    |     |          |
| Linear regression                     |     |     |    |     |          |
| XGB-B (terrain + wind)                |     |     |    |     |          |
| PINN v1 (baseline, global$y\_{\max}$) |     |     |    |     |          |
| Env-PINN v2 (terrain-conditioned)     |     |     |    |     |          |
| Env-PINN v3 (full-conditioned)        |     |     |    |     |          |


Reported separately for: all plots, persistent under-performers, CR-conformant plots.

**5.4.2 Ablation results.** Physics weight sweep checks whether environmental conditioning reduces the regularisation burden previously carried by the physics loss alone. Anchor ablation checks whether learned growth ceilings collapse without constraint. Batch size comparison (32 vs 128) confirms PINN preference for smaller batches.

**5.4.3 Learned \$\hat{y}\_{\max}(\mathbf{e})\$ spatial map.** Plot-level map of learned growth ceilings — the central interpretable output. Expected: lower ceilings on exposed ridges and at high elevation, higher in sheltered valley floors. Overlaid on trajectory class map; range checked against known Sitka site index variation in Scotland.

**5.4.4 Convergent validity test.** Sub-network gradient importance vs SHAP from XGB-B — bar chart and Spearman correlation. Agreement between two independent methods constitutes stronger evidence than either alone (Karniadakis et al., 2021; Lundberg & Lee, 2017).

**5.5 Synthesis.** Variance decomposition: what fraction of between-plot differences does terrain + wind explain (XGB-B R²)? What fraction of temporal variation do climate indices explain? What remains unexplained — likely thinning and management events unrecorded in the data. The section closes by identifying management records from Forest Research as the highest-value addition for future work.

***

### Chapter 6: Discussion

**6.1** Interpreting the spatial attribution — mechanistic links between learned \$\hat{y}\_{\max}\$ and ecology (thigmomorphogenesis, waterlogging, elevation lapse rate). Does the map contradict any known Sitka ecology?

**6.2** Interpreting the temporal results — what gap-length analysis reveals about model robustness, and what the 9-year 2012→2021 interval compresses.

**6.3** Env-PINN vs baseline — what the global-parameter assumption gets wrong, and what the convergent validity test means for confidence in the attribution.

**6.4** The unobserved confounder: forest management. Thinning and felling cannot be separated from environmental effects without management records. This is the most important honest acknowledgement.

**6.5** Limitations: HadUK-Grid is coarse relative to plot-level attribution and may smooth within-forest microclimate; OS Terrain 50 DTM at 50m is coarser than the plot/grid-cell scale; WASP resolution over complex terrain needs verification; only six timestamps.

**6.6 Future work**

*   More timestamps — the single highest-value improvement
*   3PG physics loss — replace CR constraint with process-based model using monthly climate
*   Dynamic \$k(\mathbf{c})\$ conditioning on interval-level climate (Tier 4, not completed)
*   CHESS-SCAPE projections — run Env-PINN forward under climate scenarios to 2050–2080
*   Full DT integration with companion management interface and RL thinning optimiser (Arthur, 2025; Clark, 2025)
*   Individual tree level — removing the plot-aggregation artefact

***

### Chapter 7: Conclusions

Restate research questions and summarise: trajectory analysis found spatially autocorrelated CR residuals (Moran's I); XGBoost + SHAP identified the dominant environmental drivers of growth anomalies; the Env-PINN improved on the baseline PINN by learning plot-specific growth ceilings, with the improvement concentrated in persistent under-performers; the convergent validity test either confirmed or constructively challenged the attribution. Together these results provide the first environmental attribution of Sitka spruce growth variability in Aberfoyle using the full six-timestamp dataset.

***

## Risk Register

| Risk                                           | Likelihood  | Impact | Mitigation                                                                                                      |
| ---------------------------------------------- | ----------- | ------ | --------------------------------------------------------------------------------------------------------------- |
| Data audit takes longer than 2 days            | Medium      | High   | Email Dr. Suárez-Minguez Day 1; flag-and-retain for anything unclear                                            |
| Duplicate deduplication rule unclear           | Medium      | Medium | Prefer audited GeoPackage/cohort logic; if using legacy CSV duplicates, document rule and run sensitivity check |
| Many plots lack ≥2 timestamps after cleaning   | Medium      | Medium | Check coverage by year before committing to trajectory classification                                           |
| Simple baselines look wrong on new data        | Low–Medium  | High   | Don't proceed to XGBoost/PINN until CR baseline makes sense; trace back to cleaning                             |
| Moran's I not significant                      | Low–Medium  | High   | SQ1-C (XGBoost + SHAP) still stands independently without spatial autocorrelation                               |
| Env-PINN sub-network fails to converge         | Medium      | High   | Pre-train on XGB-B residuals; use anchor loss; report as negative result if needed                              |
| WASP unavailable or wrong resolution           | Low         | Low    | Use Global Wind Atlas as public wind layer and TOPEX from DTM as terrain-derived fallback                       |
| Thinning events dominate under-performer class | Medium–High | Medium | Flag temporal-spike plots; request management records; acknowledge as limitation                                |
| Writing takes longer than expected             | Medium      | High   | Start Ch. 1–3 from Day 3 in parallel with code                                                                  |


***

## Data Access Checklist

*   \[ ] HadUK-Grid (1km UK gridded observations; daily rainfall and temperature, plus monthly/seasonal/annual variables depending on variable) — CEDA: <https://catalogue.ceda.ac.uk/uuid/4dc8450d889a491ebb20e724debe2dfb> — Open Government Licence
*   \[ ] CHESS-met / successor PET source — EIDC catalogue. Note: the Robinson et al. (2020) CHESS-met record covers 1961–2017 and is superseded by a 1961–2019 version, so it does not by itself cover 2021–2023.
*   \[ ] OS Terrain 50 DTM — osdatahub.os.uk/downloads/open/Terrain50 — free
*   \[ ] WASP wind speed data over Aberfoyle — via Dr. Suárez-Minguez (confirm resolution)
*   \[ ] Global Wind Atlas — <https://globalwindatlas.info/> — free; download GIS wind speed / wind power density layers, confirm selected height and citation/licence
*   \[x] LiDAR timestamps 2002, 2006, 2008, 2012, 2021, 2023 — received as `LiDAR_Years_All_attributes.csv`

***

## Source Quality and Additions

### High-priority sources to cite

These are strong enough to build the dissertation argument around.

| Claim supported                                                                                       | Recommended source                                                                                                                                 | How to use                                                                                                                                                       |
| ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| HadUK-Grid is a 1km UK gridded climate observation product from station interpolation                 | Met Office; Hollis, McCarthy, Kendon, Legg & Simpson, HadUK-Grid CEDA record; Hollis et al. (2019),*Geoscience Data Journal*, DOI: 10.1002/gdj3.78 | Data chapter and climate-index limitations. The CEDA record says the latest release is v1.3.2.ceda, released June 2026 and containing data to the end of 2025.   |
| CHESS-met is 1km daily meteorology for GB, but published records may not cover the newest LiDAR years | Robinson, Blyth, Clark, Comyn-Platt & Rudd (2020), NERC EIDC, DOI: 10.5285/2ab15bf0-ad08-415c-ba64-831168be7293                                    | Use cautiously for PET/SMD. Confirm the latest accessible successor dataset before deriving 2021–2023 indices.                                                   |
| OS Terrain 50 is 50m post-spacing DTM, broad-scale terrain product                                    | Ordnance Survey OS Terrain 50 product documentation                                                                                                | Data chapter; justify 50m terrain derivatives and acknowledge it is coarser than the plot/grid-cell scale.                                                       |
| Global Wind Atlas provides public modelled wind climatology useful as a fallback/comparison layer     | Global Wind Atlas, developed by DTU Wind Energy with World Bank / ESMAP support; use accompanying methodology/citation notes when downloading data | Data chapter and sensitivity analysis. Use as a 250m wind-resource layer, not as a direct canopy wind measurement.                                               |
| Site-specific growth ceilings are established in forestry growth modelling                            | Socha et al. (2021),*Scientific Reports*; Pienaar & Turnbull (1973); CR-H / ADA growth-curve literature                                            | Background and Env-PINN design justification. This is the strongest support for letting`y_max`vary by site/environment.                                          |
| Sitka spruce growth responds to climate/site effects in GB                                            | Manso, Davidson & McLean (2022),*Forestry*, DOI: 10.1093/forestry/cpab049                                                                          | Background. Their unexplained site effect is a clean motivation for your terrain/wind attribution.                                                               |
| Topographic wind exposure matters for Sitka productivity in upland Britain                            | Worrell (1987),*Forestry Commission Bulletin 72*                                                                                                   | Ecology background and interpretation of TOPEX/WASP effects.                                                                                                     |
| SHAP is the correct citation for feature attribution                                                  | Lundberg & Lee (2017); Lundberg et al. (2020)                                                                                                      | Methods and results. Use TreeSHAP citation specifically for XGBoost.                                                                                             |
| Spatial validation matters for environmental ML                                                       | Roberts et al. (2017); Ploton et al. (2020); Wadoux et al. / spatial CV critique                                                                   | Methods. Cite both the need for spatial blocking and the limitation that spatial CV can be pessimistic depending on deployment target.                           |
| PINNs/process-informed ML provide small-data regularisation                                           | Raissi et al. (2019); Karniadakis et al. (2021); Wesselkamp et al. (2024); Zhang et al. (2023)                                                     | Methods framing. Be precise that your CR-PINN physics loss may be derivative-consistency, residual/parallel, or process-regularised depending on implementation. |


### Sources to verify before citing

These may be useful, but should not be treated as firm references until bibliographic details and relevance are checked.

| Candidate                                                     | Why it may help                                                                                           | Check before use                                                                                                                                   |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| Frontiers in Remote Sensing 2025 LiDAR forest-attribute paper | Recent context for LiDAR-derived forest attributes                                                        | Confirm title, authors, study system, and whether it is about plot-level forest inventory rather than unrelated remote-sensing tasks.              |
| MDPI Remote Sensing 2019 paper linked in citations file       | The pasted notes suggest it supports environmental covariates modifying growth-curve intercept/site index | Confirm exact paper title and whether the model really modifies the intercept/asymptote.                                                           |
| bioRxiv 2020 LiDAR time-series paper                          | Potential precedent for repeated ALS growth monitoring                                                    | Check whether a peer-reviewed version exists; cite published version if available.                                                                 |
| arXiv 2025 / Hybrid-FINN source in pasted notes               | Could be very relevant if it is genuinely forest demography + neural network + spatial blocked CV         | Confirm title/authors. The current pasted note appears to conflate this with AgriPINN; do not cite until verified.                                 |
| Forest Research Sitka spruce species page                     | Useful practical ecology source                                                                           | Use as supporting/grey literature, not as the only source for physiological claims.                                                                |
| Soil datasets from James Hutton / UKCEH / SoilGrids           | Possible covariates                                                                                       | Extract and map first. Include only if values vary credibly inside the cleaned Aberfoyle cohort.                                                   |
| Global Wind Atlas exact product metadata                      | Need exact selected layer, height, version, licence, and download format                                  | Verify during download. Prefer the lowest above-ground layer relevant to forest exposure; avoid interpreting 100m/150m wind as canopy-height wind. |


### Carry-over from the April proposal

The April proposal is now a windthrow/PINN proposal, not the current dissertation. Carry forward only the pieces that still support this growth-attribution dissertation: ForestGALES/wind exposure context if discussing wind hazard, Wesselkamp et al. for process-informed architecture taxonomy, Buonocore/Jiang for Digital Twin framing, and general terrain-wind citations. Do not carry forward claims about 10m grids, NFI ground truth, Harwood transfer, Storm Arwen labels, or binary windthrow classification unless the project explicitly returns to that scope.

***

## Key References by Section

| Section                  | Key citations                                                                                            |
| ------------------------ | -------------------------------------------------------------------------------------------------------- |
| Sitka spruce ecology     | Forest Research (2025); Morison et al. (2010); Worrell (1987); Telewski (2006)                           |
| CR and growth models     | Pienaar & Turnbull (1973); Von Bertalanffy (1949); Landsberg & Waring (1997); Forrester et al. (2021)    |
| Factors affecting growth | Toledo et al. (2011); Boulanger et al. (2024); Gardiner et al. (2000); Worrell (1987)                    |
| PINN framework           | Raissi et al. (2019); Karniadakis et al. (2021); Batuwatta-Gamage et al. (2022); Nathaniel et al. (2023) |
| SHAP methodology         | Lundberg & Lee (2017); Lundberg et al. (2020); Brugger et al. (2023)                                     |
| Knowledge-guided ML      | GCB review (2025); Cuomo et al. (2022); Jia (2025)                                                       |
| XGBoost and spatial CV   | Everingham et al. (2016); Freeman et al. (2015); Roberts et al. (2017)                                   |
| HadUK-Grid               | Hollis et al. (2019)                                                                                     |
| CHESS-met                | Robinson et al. (2020)                                                                                   |
| Trajectory analysis      | Irauschek et al. (2021)                                                                                  |
| DT context               | Buonocore et al. (2022); Jiang et al. (2022); Arthur (2025); Clark (2025)                                |


***

## Appendices

**Appendix A: Abbreviations**

| Abbreviation | Expansion                                                             |
| ------------ | --------------------------------------------------------------------- |
| CR           | Chapman-Richards                                                      |
| PINN         | Physics-Informed Neural Network                                       |
| Env-PINN     | Environmentally-conditioned PINN (this work)                          |
| TWI          | Topographic Wetness Index                                             |
| TOPEX        | Topographic Exposure index (wind shelter)                             |
| SMD          | Soil Moisture Deficit                                                 |
| PET          | Potential Evapotranspiration                                          |
| GDD          | Growing Degree Days                                                   |
| SHAP         | SHapley Additive exPlanations                                         |
| WASP         | Wind Atlas Analysis and Application Program                           |
| HadUK-Grid   | Met Office 1km gridded UK climate observations                        |
| CHESS-met    | CEH Climate Hydrology and Ecology research Support System meteorology |
| OS           | Ordnance Survey                                                       |
| DT           | Digital Twin                                                          |
| VIF          | Variance Inflation Factor                                             |
| LISA         | Local Indicators of Spatial Association                               |


**Appendix B: Mathematical derivations**

*   CR scaling correction — \$\hat{y}\_{\max}(\mathbf{e})\$ enters the PDE after the scaling correction; chain rule derivation unchanged
*   TWI formula derivation
*   TOPEX computation from DTM

**Appendix C: Additional figures**

*   Full (unclipped) SHAP spatial maps
*   Env-PINN training loss curves across all ablation configurations
*   \$\hat{y}\_{\max}(\mathbf{e})\$ surface at high resolution
*   Trajectory class map with all plots
*   Full feature correlation matrix

**Appendix D: Data input tables**

*   All static features with source, resolution, DOI, licence
*   All temporal climate indices with formula, data source, and interval values

***

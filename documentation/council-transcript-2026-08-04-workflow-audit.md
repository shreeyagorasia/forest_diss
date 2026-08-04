# LLM Council Transcript — Dissertation Workflow Audit

Date: 2026-08-04

## Original question

Are there other errors throughout the general dissertation workflow, beyond the known polygon-centroid problem?

## Framed question

Audit the end-to-end Aberfoyle LiDAR dissertation workflow: raw-data selection, geometry, spatial/environmental extraction, temporal feature availability, target derivation, model splits, tuning, evaluation, uncertainty, reproducibility, and claims. Distinguish confirmed defects from design limitations and disclosure issues. Prioritise fixes that materially affect conclusions without reopening unnecessary architecture searches.

The requested council skill was not installed, so its linked public SKILL.md was used as the framework. The environment allowed three sub-agents concurrently; three advisors ran independently and the First-Principles and Executor lenses were conducted locally before anonymous peer review.

## Advisor responses

### Contrarian

The centroid issue is not the largest threat. The temporal models use covariates measured at the same LiDAR survey as the outcome: 2023 height is predicted using 2023 `CanopyCover` and management fields. This supports contemporaneous height estimation, not genuine forecasting from information available before 2023. Canopy cover comes from the same LiDAR scene and is tightly coupled to canopy structure.

The nominal test set has also influenced modelling decisions. Repository comments record removing `yldc` because test R2 improved, and earlier environmental ablations repeatedly reused test results. A later validation split cannot restore a test set already inspected. A fresh locked spatial holdout or nested spatial CV is required for a confirmatory headline estimate.

Finally, plot-specific growth-curve quantities fitted using all survey years, including 2023, support retrospective attribution rather than prospective prediction or causal claims.

### Expansionist

The highest-value check is target-provenance reconstruction. `mean_cr_residual` is computed with a pooled Chapman–Richards fit before the environmental train/validation/test split, while the repository later confirms that the pooled anchor overlaps held-out populations. Reconstruct residual targets within each training fold for predictive attribution, or explicitly label the existing analysis retrospective/descriptive.

Turn coarse-resolution reuse into a robustness result: report unique source cells and use compartment- or raster-cell-clustered uncertainty, cell aggregation, or inverse-occupancy weighting. Prioritise compartment spatial CV because the documented residual range is about 3,956 m while the 60 m buffer only protects against near neighbours. Rename 11x11 and 21x21 terrain operations as square-window measures, or run a circular-kernel sensitivity check.

### Outsider

Several statements will confuse an examiner. GWA is called “observed wind speed” although it is modelled climatology. The advertised install is not reproducible because active dependencies such as PyTorch, Rasterio, PyProj, XGBoost and SHAP are absent from `requirements.txt`. The population is a balanced-panel survivor cohort, not all Aberfoyle plots: complete repeated coverage, persistent Sitka spruce, valid measurements, and observed 2023 maturity are required.

The 60 m buffer must not be described as removing long-range dependence; compartment holdout does most of that work. Coarse cells repeated across many plots create multilevel dependence. Test reuse should separate exploratory from confirmatory findings.

### First-Principles Thinker

The organising problem is that the project mixes three estimands: contemporaneous height reconstruction, future forecasting, and retrospective environmental attribution. Same-year canopy cover is legitimate for the first and invalid for the second. All-year fitted growth parameters can describe the third but cannot establish prospective performance or causality.

Before interpreting metrics, create a feature-provenance audit with source time, measurement process, spatial support, derivation, prediction-time availability, and fold provenance. Every feature and transformed target should pass the relevant estimand’s availability rule.

### Executor

Freeze architectures and hyperparameters. Correct only invalid centroids using centroid-if-inside, otherwise `representative_point()`. Reconstruct `Thin` as a survey-time status from `last_thinn`, then regenerate affected tables and run paired fixed-model sensitivity. Create a machine-readable provenance table. Use a fresh locked spatial holdout or nested spatial CV only for final headline claims. Do not rerun unrelated no-environment models for the geometry correction unless their management fields change.

## Anonymous peer reviews

### Review 1

Response D was strongest because defining the estimand determines whether a feature is legitimate or leaking. Response E’s management correction needed source-semantics verification. All responses underemphasised survivor selection and long-range between-compartment dependence.

### Review 2

Response D was strongest conceptually and E operationally. The shared blind spot was LiDAR campaign comparability: sensor, season, point density, processing and coverage changes can masquerade as growth or temporal failure. Uncertainty is nested across surveys, plots, raster cells and compartments; thinning importance is not causal because treatment is endogenous.

### Review 3

Response E was strongest as an actionable workflow; D best framed the problem and B identified a potentially consequential target-provenance defect. All responses needed a complete temporal audit of management fields and LiDAR acquisition comparability. Geometry validity and multipart structure should be checked before replacing centroids.

### Review 4

Response A was strongest on immediate threats to headline predictive claims. Its blind spot was that same-survey covariates are not errors if the stated task is contemporaneous estimation. All responses needed quantified prevalence and impact before ordering reruns.

### Review 5

Response B was strongest for environmental inference because fold-derived residuals and clustered uncertainty directly affect attribution. Its blind spot was cost and the distinction between descriptive and predictive residual analysis. All responses missed documentation drift between the active Python pipeline and `data/processed/README.md`.

## Chairman synthesis

### Where the Council Agrees

The project must define separate estimands and align feature availability and claims with each. Test reuse prevents the repeatedly inspected split from serving as a pristine confirmatory test. The centroid correction should be narrow. Coarse raster support, survivor-cohort selection, and long-range spatial dependence need explicit treatment. Architecture retuning is not the priority.

### Where the Council Clashes

The council differs on whether same-year canopy cover and pooled residuals are “leaks” or valid retrospective inputs. The resolution is task-dependent: they are inappropriate for prospective forecasting and clean held-out predictive attribution, but can be valid for explicitly contemporaneous or descriptive analyses.

### Blind Spots the Council Caught

The most important new blind spots were survey-time leakage in `Thin`, LiDAR campaign measurement invariance, multilevel dependence, thinning endogeneity, stale installation/data documentation, and square-versus-circular terrain windows.

### Recommendation

Run a staged audit, not a wholesale rebuild. First lock the estimands and feature provenance. Then fix confirmed data errors (`Thin`, invalid centroids, misleading documentation), regenerate affected data, and rerun existing fixed models. Use fresh/nested spatial evaluation for claims intended to be confirmatory. Treat acquisition comparability, survivor selection and clustered uncertainty as required robustness work.

### The One Thing to Do First

Create and approve a feature/target provenance table that declares, for every modelling column, when it was observed, how it was derived, its spatial support, and whether it is available for each claimed prediction task.

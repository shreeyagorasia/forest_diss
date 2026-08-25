import pandas as pd
import shap
import xgboost as xgb

# One short note per column: "own calculation" means it was built directly in this repo (from
# the OS Terrain 50 DTM, the raw GPKG's plot geometry, or another already-listed variable) --
# nothing was downloaded for it specifically. "external" means the raw value comes straight from
# a downloaded dataset. Where a variable is derived FROM another variable already in this dict,
# that's named explicitly, so it's easy to see which pairs might be redundant with each other.
FEATURE_PROVENANCE = {
    "elevation": "external (OS Terrain 50 DTM, read directly)",
    "slope_degrees": (
        "own calculation: elevation gradient on the 50m DTM grid (np.gradient, east/north "
        "components), slope = arctan(sqrt(dz_dx^2 + dz_dy^2)), converted to degrees"
    ),
    "northness": "own calculation: cos(aspect_degrees in radians)",
    "eastness": "own calculation: sin(aspect_degrees in radians)",
    "profile_curvature": (
        "own calculation (second derivative of the DTM, Zevenbergen & Thorne 1987). Sign "
        "convention validated against a synthetic bowl/dome before use on real data"
    ),
    "plan_curvature": (
        "own calculation (second derivative of the DTM, Zevenbergen & Thorne 1987). Same "
        "validation as profile_curvature"
    ),
    "tpi": (
        "own calculation: elevation minus a 100m-window mean elevation (scipy.ndimage."
        "uniform_filter over the DTM grid, ~3x3 cells at 50m resolution)"
    ),
    "elevation_roughness": (
        "own calculation: std of elevation within a 100m-diameter window, via the moving-window "
        "variance identity Var = E[X^2] - E[X]^2 (scipy.ndimage.uniform_filter, much faster than "
        "a literal per-cell std loop)"
    ),
    # tpi_250m deliberately NOT in this dict (removed 2026-08-06, was briefly included
    # 2026-08-04). Avenue 2's own correlation_screen.py measured it directly: ρ=0.841 with the
    # existing `tpi` (100m window), ρ=0.879 with `tpi_500m` below. Genuinely redundant with its
    # two multiscale neighbours, unlike `tpi`/`tpi_500m` themselves (only ρ=0.619 apart, real
    # distinct extremes worth keeping both). Never used by any DNN/PINN model (not in any
    # ENV_TERRAIN_FEATURE_SETS entry in torch_data.py), so removing it here doesn't invalidate
    # any existing model result. Only this notebook's own multiscale-terrain analysis is
    # affected. See documentation/variable_registry_av1_av2.csv for the full cross-avenue check.
    "tpi_500m": (
        "own calculation (data_processing/add_environmental_candidates.py, 2026-08-04): elevation "
        "minus a 21x21-cell square moving-window mean, same OS Terrain 50 tiles and method as the "
        "existing `tpi` (100m window) above. Kept alongside `tpi` (not `tpi_250m`, see the removal "
        "note above) as a genuinely distinct scale. ρ=0.619 vs native `tpi`, confirmed via "
        "Avenue 2's correlation_screen.py, not a redundant intermediate step."
    ),
    "local_relief_500m": (
        "own calculation (data_processing/add_environmental_candidates.py, 2026-08-04): maximum "
        "minus minimum elevation within the same 21x21-cell square window used for tpi_500m. A local "
        "ruggedness measure, not a directional slope/aspect one."
    ),
    "solar_radiation_index": (
        "own calculation: cos(slope)*cos(zenith) + sin(slope)*sin(zenith)*cos(aspect - solar "
        "azimuth), evaluated at solar noon on the summer solstice (declination 23.44 deg, site "
        "latitude 56.15 deg). No horizon shading by surrounding terrain, a known simplification"
    ),
    "inverse_slope_proxy": (
        "own calculation, derived directly from slope_degrees (= -slope_degrees, a heuristic, "
        "not new information). An EXACT linear duplicate, not just correlated. For XGBoost "
        "this is mostly a SHAP-attribution question (which of the two gets credited). For "
        "Elastic Net it's more consequential: with two perfectly collinear columns, the "
        "regularisation path can split the same underlying signal's coefficient between them "
        "in a way that depends on solver internals, not on anything about the data. Treat "
        "inverse_slope_proxy's and slope_degrees's individual Elastic Net coefficients as not "
        "separately meaningful, only their sum is (2026-07-30 clarification)."
    ),
    "frost_hollow_flag": (
        "own calculation: binary flag = (tpi below its own 15th percentile) AND (plan_curvature "
        "< 0, i.e. concave). A documented heuristic threshold, not calibrated against any real "
        "frost/damage record"
    ),
    "topex": (
        "own calculation (Wilson 1984 method): sum of horizon angles sampled at 8 compass "
        "directions (N/NE/E/SE/S/SW/W/NW), 1000m radius, each angle = arctan2(remote elevation - "
        "own elevation, distance) in degrees. Positive total = more sheltered"
    ),
    "windward_topex": (
        "own calculation: reuses the same topex() function as topex, restricted to a single "
        "bearing (225 degrees / south-west, Scotland's prevailing wind direction), same 1000m radius"
    ),
    "gwa_weibull_a_10m": (
        "external (Global Wind Atlas GBR national Weibull-A raster, 10m height, cropped to a 5km "
        "buffer around Aberfoyle by data_processing/add_environmental_candidates.py, 2026-08-04). "
        "A DIFFERENT source/product to the existing gwa_wind_speed_10m above (that one is the GWA "
        "REST API's mean-wind endpoint; this is the raw national Weibull-scale raster). Both "
        "describe 10m wind but are not guaranteed to agree exactly, worth checking their "
        "correlation before treating both as independent evidence. CAVEAT (2026-08-04, flagged by "
        "a parallel session's council review of the same 10m/50m GWA family for a different "
        "target): 10m wind is a SUB-CANOPY measurement. Using it to explain a MATURE canopy's "
        "wind response is conceptually backwards for this stand. Prefer the 50m columns below (or "
        "topex/windward_topex/whcl) as the primary wind-exposure variables; keep the 10m columns "
        "as a comparison, not the headline evidence, until checked."
    ),
    "gwa_weibull_k_10m": (
        "external, same source/script as gwa_weibull_a_10m. The Weibull shape parameter at 10m."
    ),
    "gwa_wind_p95_10m": (
        "own calculation, DERIVED from gwa_weibull_a_10m and gwa_weibull_k_10m "
        "(A * [-ln(1-0.95)]^(1/k), data_processing/add_environmental_candidates.py). Not a third "
        "independent measurement. Per this notebook's own redundancy rule (grouped_category_"
        "importance.ipynb cell 45/48), do not read this alongside both Weibull parameters as if "
        "all three carry separate information."
    ),
    "gwa_prob_above_critical_10m": (
        "own calculation, DERIVED from gwa_weibull_a_10m and gwa_weibull_k_10m "
        "(exp[-(20 m/s / A)^k], data_processing/add_environmental_candidates.py). Modelled "
        "probability of exceeding a 20 m/s screening threshold, not a validated tree-failure "
        "threshold. Same derived-not-independent caveat as gwa_wind_p95_10m."
    ),
    "gwa_wind_speed_50m": (
        "external (Global Wind Atlas GBR national mean-wind raster, 50m height, same crop/script "
        "as gwa_weibull_a_10m). A different height level to gwa_wind_speed_10m, not a duplicate, "
        "but plausibly highly correlated with it. Check before treating both as independent."
    ),
    "gwa_weibull_a_50m": "external, same source/script as gwa_weibull_a_10m. Weibull scale parameter at 50m.",
    "gwa_weibull_k_50m": "external, same source/script as gwa_weibull_a_10m. Weibull shape parameter at 50m.",
    "gwa_wind_p95_50m": (
        "own calculation, DERIVED from gwa_weibull_a_50m and gwa_weibull_k_50m. Same derivation "
        "and same not-independent caveat as gwa_wind_p95_10m."
    ),
    "gwa_prob_above_critical_50m": (
        "own calculation, DERIVED from gwa_weibull_a_50m and gwa_weibull_k_50m. Same derivation "
        "and same not-independent caveat as gwa_prob_above_critical_10m."
    ),
    "dist_to_cpmt_boundary": (
        "own calculation: geometric distance from the plot point to its own compartment "
        "polygon's boundary (GPKG compartment geometry, one boundary per cpmt)"
    ),
    "dist_to_forest_perimeter": (
        "own calculation: distance to the forest's real exterior/interior boundary. Compartment "
        "polygons unioned then buffered +60m/-60m (morphological closing, merges slivers and "
        "narrow gaps between adjacent compartments), closed regions under 1ha dropped as "
        "closing artefacts, boundary includes both exterior and interior (internal clearing) rings"
    ),
    "dist_to_scpt_boundary": (
        "own calculation: geometric distance from the plot point to its own sub-compartment "
        "polygon's boundary (GPKG sub-compartment geometry, one boundary per (cpmt, scpt) pair)"
    ),
    "dist_to_block_boundary": (
        "own calculation: geometric distance from the plot point to its own block polygon's "
        "boundary (GPKG block geometry, one boundary per blk)"
    ),
    "cpmt_compactness_ratio": (
        "own calculation: compartment polygon perimeter / area. A per-compartment shape "
        "property, not a per-plot distance (every plot in the same compartment shares one value). "
        "NOT a leak (2026-07-31 clarification, to distinguish this from the real "
        "neighbour_mean_height leak above). But a real interpretive caveat: spatial_block_split() "
        "holds out whole compartments, so a held-out compartment's own value here may never have "
        "appeared in training at all. That makes generalisation for this specific variable a "
        "stricter test than for a smoothly-varying continuous one, not a data problem to fix. "
        "the split is doing exactly what it's supposed to. Read this variable's importance/effect "
        "numbers with that in mind, don't average it against continuous variables as if the test "
        "were equally hard for both."
    ),
    "dist_to_road": (
        "own calculation: distance from the plot point to the nearest OS Open Roads line, same "
        "distance-to-nearest-line method as dist_to_watercourse. Includes unclassified tracks, "
        "not just A/B roads"
    ),
    "dist_to_watercourse": (
        "own calculation: distance from the plot point to the nearest OS Open Rivers "
        "watercourse line (real vector survey data, not a flow-accumulation derivation), "
        "watercourses read within a 5km-buffered bounding box around the study area"
    ),
    "gwa_wind_speed_10m": (
        "external (Global Wind Atlas public REST API, GBR/wind-speed/10. 10m height, no auth). "
        "Nominal 250m grid, but that's a longitude/latitude spacing, not a real ground distance "
        "at this latitude. Confirmed real resolution over Aberfoyle is ~140-155m (E-W) x 250m "
        "(N-S), anisotropic. Modelled climatology (long-term average, not tied to any survey "
        "year), same nearest-cell extraction as every other raster here. See "
        "notebooks/environmental_data/environmental_data_sources_survey_SUPERSEDED_2026-07-30.ipynb's resolution check."
    ),
    "soilgrids_ph": (
        "external (SoilGrids v2.0, ISRIC, public Cloud-Optimised GeoTIFF via rasterio's "
        "/vsicurl/, no auth). 250m resolution. Topsoil pH specifically. SoilGrids' pH product "
        "is depth-stratified (0-5/5-15/15-30/30-60/60-100/100-200cm); this uses the shallowest "
        "layer, phh2o_0-5cm_mean. See notebooks/environmental_data/aux_data_resolution_check.ipynb."
    ),
    "ceh_pedotope": (
        "external (CEH natural capital dataset, natural_capital_pedotopes.tif, one of 9 layers "
        "in the same downloaded zip. 50m resolution, 7 classes over Aberfoyle). A categorical "
        "class ID, not an ordinal scale."
    ),
    "ceh_twi": (
        "external (CEH Topographic Wetness Index, twi.tif, 50m resolution, from the same CEH "
        "zip/product as ceh_pedotope)."
    ),
    "ceh_subsurface_drainage": (
        "external (CEH subsurface drainage raster, 50m resolution, from the same CEH zip/product "
        "as ceh_pedotope). A categorical class ID."
    ),
    "ceh_textural_composition": (
        "external (CEH soil textural composition raster, 50m resolution, from the same CEH "
        "zip/product as ceh_pedotope). A categorical class ID."
    ),
    "chelsa_bio1_celsius": (
        "external (CHELSA v2.1, direct public download, no auth). 1km nominal resolution (same "
        "grid as HadUK-Grid), but terrain-conditioned downscaling, not a plain interpolation. "
        "1981-2010 climatology (static, not tied to any survey year). Bio1 = mean annual air "
        "temperature."
    ),
    "chelsa_gdd5_degc": (
        "external (CHELSA v2.1 BIOCLIM+ extension, same access/resolution/period as "
        "chelsa_bio1_celsius: 1km, 1981-2010 climatology). Growing degree days above 5C."
    ),
    "chelsa_bio12_precip_mm": (
        "external (CHELSA v2.1, same access/resolution/period as chelsa_bio1_celsius: 1km, "
        "1981-2010 climatology). Bio12 = annual precipitation."
    ),
    "tas_mean": (
        "external (HadUK-Grid, CEDA, mean annual temperature, 1km). Genuinely multi-year "
        "(2026-07-30 fix, models/common/download_haduk_multi_year.py): averaged across each "
        "cohort's own survey years (2008/2012/2021/2023 for 4survey, all 6 "
        "2002/2006/2008/2012/2021/2023 for 6survey) via load_plots_for_cohort()'s cohort-suffixed "
        "columns (tas_mean_4survey/_6survey), not a single reused snapshot year applied "
        "uniformly. See notebooks/environmental_data/aux_data_resolution_check.ipynb for the "
        "screening (rho=0.273 vs 4survey CR residual) and why rainfall/tasmax/tasmin/sfcWind "
        "were also extracted but not kept (redundant with existing CHELSA/GWA variables, or. "
        "for sfcWind. Misleadingly labelled: rho=-0.96 with tas itself, only rho=0.25 with the "
        "existing gwa_wind_speed_10m)."
    ),
    "groundfrost_mean": (
        "external (HadUK-Grid, CEDA, days of ground frost per month, 1km, added 2026-07-30). "
        "Same multi-year, cohort-averaged extraction as tas_mean above. Checked as a potential "
        "complement/validation for the existing frost_hollow_flag terrain heuristic. Real "
        "signal (rho=0.172 vs 4survey CR residual), fairly distinct from the other climate "
        "variables already in the model."
    ),
    "whcl": (
        "external (raw GPKG windthrow hazard class field, integer 0-6, ordinal severity. "
        "constant per plot across survey years, confirmed by direct check, not assumed). Same "
        "compartment-constant caveat as cpmt_compactness_ratio above (2026-07-31). Assigned per "
        "forestry compartment, so a held-out compartment under spatial_block_split() may carry a "
        "class value never seen in training. Not a leak, a stricter generalisation test; read its "
        "importance/effect numbers with that in mind."
    ),
    # Silvicultural (stand-structure), not environmental. Kept in this same dict/set, not split
    # out separately (see progress_notes.md for why). Mean-aggregated per plot across survey years.
    # Age itself is deliberately NOT included here. It's circular with mean_cr_residual's own
    # construction (the Chapman-Richards curve's only input), see progress_notes.md for the
    # age-binned residual check that found a real, non-monotonic fit bias XGBoost was exploiting.
    "CanopyCover": "external (raw survey field, mean canopy cover fraction across every survey year)",
    "Thin": "external (raw survey field, fraction of surveys where this plot had been thinned)",
    "time_since_thinning": "external (raw survey field, NaN for never-thinned filled with 0 before averaging. Time_since_thinning_missing carries the real signal for those plots)",
    "time_since_thinning_missing": "external (raw survey field, fraction of surveys where this plot had never yet been thinned)",
    "recent_thinning_5yr": "external (raw survey field, fraction of surveys with a thinning in the preceding 5 years)",
}
# era5_land_temp_k is deliberately NOT in this dict, and so not in any feature set below --
# excluded on the numbers alone (only 8 distinct values across all 71,766 plots, confirmed
# ~11.1km native resolution), not grouped with tas_mean by category. tas_mean has 149 distinct
# values and one of the strongest raw correlations with the CR residual of anything tested
# (rho=0.27). Kept, on its own numbers, despite sharing the same "climate snapshot" shape as
# the excluded ERA5-Land variable. See
# notebooks/environmental_data/aux_data_resolution_check.ipynb's climate correlation section for
# the full comparison.

# neighbour_mean_height/neighbour_height_differential are deliberately NOT in this dict either
# (2026-07-31 fix. Previously included, tagged "SPATIAL-LAG, not exogenous" as a caveat, not
# excluded). Removed after confirming a real leak, not just a trust concern: both are built from
# every OTHER plot's own real 2023 height within a 75m radius, computed ONCE on the full
# 71,766-plot set BEFORE any train/val/test split exists
# (notebooks/environmental_data/aux_data_resolution_check.ipynb). Checked directly: because
# spatial_block_split() holds out whole compartments, 95.9% of a TEST-set plot's within-75m
# neighbours are ALSO test-set plots (82.3% of test plots have ZERO train-set neighbours in
# range). So for the large majority of test rows, this "feature" is built almost entirely from
# other test-set plots' real ground-truth heights, not learned from training data at all. Real
# consequence, already on disk: removing it drops XGBoost's spatial_block test R2 from 0.598 to
# 0.321 (4survey), and from 0.327 to -0.337 (6survey. Worse than predicting the mean). See
# experiment_log.md's 2026-07-31 entry for the full check and progress_notes.md for where this
# sits in the wider Tier-2 attribution story. If a genuine spatial-lag feature is wanted again in
# future, it needs to be computed split-aware (e.g. only ever averaging TRAINING-set neighbours'
# heights, even for val/test rows), not the global pre-split construction used here.

# aspect_degrees is deliberately excluded (a raw 0-360 bearing is a bad model input. See
# progress_notes.md for the full reasoning); northness/eastness are its fixed replacement.
# Redundant-looking pairs otherwise stay in raw. SHAP evidence decides what's redundant, not a
# pre-judgement here.
ALL_FEATURE_COLUMNS = list(FEATURE_PROVENANCE.keys())

TERRAIN_AND_WIND_COLUMNS = [
    # Genuinely terrain + wind only (2026-07-30 fix). This used to also include 6
    # spatial-position/edge-effect columns (dist_to_cpmt_boundary, dist_to_forest_perimeter,
    # dist_to_scpt_boundary, dist_to_block_boundary, cpmt_compactness_ratio, dist_to_road) plus
    # dist_to_watercourse (a soil/site variable), which meant any "terrain and wind explain X%"
    # claim built on this constant overstated what was actually isolated. The claim that the old
    # list "matches the dissertation plan's original XGB-A/B framing" didn't hold up either --
    # the plan's own XGB-A/B/C table (`Dissertation Plan v5 - 7th June.md`) defines those as
    # "terrain only" / "terrain + WASP/GWA wind speed", nothing about edge-effects or hydrology.
    # This list is now exactly grouped_analysis.py's CATEGORY_GROUPS["terrain"] +
    # CATEGORY_GROUPS["wind"]. The categorization already used (and already correct) for the
    # grouped permutation importance / Moran's-I-before-after checks, so there's now one
    # consistent definition of "terrain" and "wind" across every analysis in this project, not two.
    "elevation", "slope_degrees", "northness", "eastness",
    "profile_curvature", "plan_curvature", "tpi", "elevation_roughness", "ceh_twi",
    "solar_radiation_index", "inverse_slope_proxy", "frost_hollow_flag",
    "topex", "windward_topex", "whcl", "gwa_wind_speed_10m",
]

# Two named feature sets to fit and compare:
#   - all_environmental: every candidate variable, the main deliverable. Used to include
#     neighbour_mean_height/neighbour_height_differential (spatial-lag features). Removed
#     2026-07-31 after confirming they leak test-set ground truth into themselves (see the
#     FEATURE_PROVENANCE comment above and experiment_log.md's 2026-07-31 entry). There used to
#     be a separate "all_environmental_no_neighbour" entry here specifically to answer that
#     caveat. Redundant now that the leaky columns are gone from ALL_FEATURE_COLUMNS entirely,
#     so it's been removed rather than kept as a duplicate of this one.
#   - terrain_and_wind_only: genuinely terrain+wind (see TERRAIN_AND_WIND_COLUMNS above), the
#     closest equivalent to the dissertation plan's original XGB-A/B/C staging.
FEATURE_SETS = {
    "all_environmental": ALL_FEATURE_COLUMNS,
    "terrain_and_wind_only": TERRAIN_AND_WIND_COLUMNS,
}


def fit_with_columns(train_df, feature_columns, val_df=None, target_col="mean_cr_residual", seed=42, **xgb_params):
    # The actual fitting logic, taking a raw column list directly rather than a name looked up
    # in FEATURE_SETS. Lets ad-hoc analysis (e.g. the Tier-2 notebook's ablation tests, which
    # need many one-off column combinations for cluster testing) fit a model without adding a
    # new named entry to FEATURE_SETS for every combination tried.
    #
    # Two optional additions (2026-07-31, see TUNED_HYPERPARAMETERS below for why):
    #   - **xgb_params: any XGBRegressor keyword can be overridden (max_depth, reg_lambda, etc).
    #     Left empty, every call keeps behaving exactly as before (sklearn defaults). This is
    #     what the Tier-2 notebook's calls still do, since they never pass xgb_params.
    #   - val_df: if given, XGBoost stops adding trees once val performance stops improving
    #     (early stopping), instead of always building the full n_estimators regardless of
    #     whether the extra trees are still helping. Left as None, behaviour is unchanged
    #     (fits a plain, fixed number of trees, no validation set touched at fit time).
    features_train = train_df[feature_columns]

    if val_df is not None:
        # early_stopping_rounds is a CONSTRUCTOR argument in this XGBoost version (3.x), not a
        # fit()-time argument. Older XGBoost releases used to accept it in .fit() instead.
        model = xgb.XGBRegressor(random_state=seed, early_stopping_rounds=20, **xgb_params)
        features_val = val_df[feature_columns]
        model.fit(
            features_train, train_df[target_col],
            eval_set=[(features_val, val_df[target_col])], verbose=False,
        )
    else:
        model = xgb.XGBRegressor(random_state=seed, **xgb_params)
        model.fit(features_train, train_df[target_col])

    return model


# Small, deliberately simple hyperparameter grid (2026-07-31). Not an exhaustive search. Only
# two knobs swept, both aimed squarely at the specific failure this was built to fix: XGBoost's
# default settings overfitting 6survey's small train set (7,467 rows), landing at a NEGATIVE
# test R2 (worse than predicting the mean). Random Forest doesn't have this failure mode the
# same way (it bags/averages independently-grown trees rather than sequentially correcting
# residuals. See experiment_log.md's 2026-07-31 entry for the full reasoning), so this grid is
# specifically about taming XGBoost's boosting, not a generic "try lots of things" sweep.
#   - max_depth: shallower trees (2/3/4) vs. the default 6. Less capacity to memorise noise.
#   - reg_lambda: L2 regularisation strength (1 is XGBoost's own default, 5/20 are stronger).
# n_estimators is deliberately NOT swept here. It's set high (500) and early stopping (see
# fit_with_columns above) picks the right number automatically for whichever (max_depth,
# reg_lambda) pair is being tried, so there's no need for a third loop over it.
HYPERPARAMETER_GRID = [
    {"max_depth": max_depth, "reg_lambda": reg_lambda, "n_estimators": 500, "learning_rate": 0.1}
    for max_depth in [2, 3, 4, 6]
    for reg_lambda in [1, 5, 20]
]


def tune_hyperparameters(train_df, val_df, feature_columns, target_col="mean_cr_residual", seed=42):
    # Tries every combination in HYPERPARAMETER_GRID, fits on train with early stopping against
    # val, and picks whichever config gets the best VAL R2. Test is never touched here, same
    # val-decides/test-reads-once discipline used everywhere else in this project (see this
    # notebook's own "how val and test are used" note, and the PINN physics-weight sweep in
    # experiment_log.md for the same pattern applied to a different model).
    from models.common.metrics import compute_metrics

    results = []
    for params in HYPERPARAMETER_GRID:
        model = fit_with_columns(train_df, feature_columns, val_df=val_df, target_col=target_col, seed=seed, **params)
        val_predictions = predict_with_columns(val_df, model, feature_columns)
        val_r2 = compute_metrics(val_df[target_col].values, val_predictions)["r2"]
        results.append({**params, "val_r2": val_r2, "best_iteration": model.best_iteration})

    results_sorted = sorted(results, key=lambda row: row["val_r2"], reverse=True)
    best_params = {k: v for k, v in results_sorted[0].items() if k not in ("val_r2", "best_iteration")}
    return best_params, results_sorted


def fit(train_df, feature_set_name, val_df=None, target_col="mean_cr_residual", seed=42, **xgb_params):
    # Thin wrapper for the two named, permanent feature sets above. Everything else
    # (run_xgb_environmental.py) keeps calling this exactly as before, just with val_df/
    # xgb_params now optionally passed through for the tuned fit.
    feature_columns = FEATURE_SETS[feature_set_name]
    return fit_with_columns(train_df, feature_columns, val_df=val_df, target_col=target_col, seed=seed, **xgb_params)


def predict_with_columns(df, model, feature_columns):
    return model.predict(df[feature_columns])


def predict(df, model, feature_set_name):
    feature_columns = FEATURE_SETS[feature_set_name]
    return predict_with_columns(df, model, feature_columns)


def compute_shap_values(model, df, feature_set_name):
    # TreeExplainer works directly on a fitted tree model. No PINN, no neural network needed
    # anywhere in this chain. Returns one row per plot, one column per feature: exactly the
    # per-plot, per-variable "how much did this push the prediction up or down here" data a
    # future interactive attribution map would need.
    feature_columns = FEATURE_SETS[feature_set_name]
    features = df[feature_columns]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Built as a fresh float DataFrame from the raw shap_values array, rather than copying
    # `features` and overwriting its values in place. `whcl` is an integer column, and SHAP
    # values are floats, so writing floats into a copy of an int column raised
    # "Invalid value ... for dtype 'int32'" the first time a non-float feature was added here.
    shap_df = pd.DataFrame(shap_values, columns=feature_columns, index=features.index)
    shap_df.insert(0, "identification", df["identification"].values)
    return shap_df


def compute_shap_values_for_columns(model, df, feature_columns):
    # Raw-column-list sibling of compute_shap_values(). Same TreeExplainer call, but takes an
    # explicit column list directly instead of a name looked up in FEATURE_SETS. Built for RQ3's
    # nested_set* tiers, which aren't named FEATURE_SETS entries. A new function, not a
    # parameter added to compute_shap_values() itself, so that function's one existing caller
    # can't have its behaviour change.
    features = df[feature_columns]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    shap_df = pd.DataFrame(shap_values, columns=feature_columns, index=features.index)
    shap_df.insert(0, "identification", df["identification"].values)
    return shap_df

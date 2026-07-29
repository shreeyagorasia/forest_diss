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
        "own calculation (second derivative of the DTM, Zevenbergen & Thorne 1987) -- sign "
        "convention validated against a synthetic bowl/dome before use on real data"
    ),
    "plan_curvature": (
        "own calculation (second derivative of the DTM, Zevenbergen & Thorne 1987) -- same "
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
    "solar_radiation_index": (
        "own calculation: cos(slope)*cos(zenith) + sin(slope)*sin(zenith)*cos(aspect - solar "
        "azimuth), evaluated at solar noon on the summer solstice (declination 23.44 deg, site "
        "latitude 56.15 deg) -- no horizon shading by surrounding terrain, a known simplification"
    ),
    "soil_depth_proxy": "own calculation, derived directly from slope_degrees (= -slope_degrees, a heuristic, not new information)",
    "frost_hollow_flag": (
        "own calculation: binary flag = (tpi below its own 15th percentile) AND (plan_curvature "
        "< 0, i.e. concave) -- a documented heuristic threshold, not calibrated against any real "
        "frost/damage record"
    ),
    "topex": (
        "own calculation (Wilson 1984 method): sum of horizon angles sampled at 8 compass "
        "directions (N/NE/E/SE/S/SW/W/NW), 1000m radius, each angle = arctan2(remote elevation - "
        "own elevation, distance) in degrees -- positive total = more sheltered"
    ),
    "windward_topex": (
        "own calculation: reuses the same topex() function as topex, restricted to a single "
        "bearing (225 degrees / south-west, Scotland's prevailing wind direction), same 1000m radius"
    ),
    "dist_to_cpmt_boundary": (
        "own calculation: geometric distance from the plot point to its own compartment "
        "polygon's boundary (GPKG compartment geometry, one boundary per cpmt)"
    ),
    "dist_to_forest_perimeter": (
        "own calculation: distance to the forest's real exterior/interior boundary -- compartment "
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
        "own calculation: compartment polygon perimeter / area -- a per-compartment shape "
        "property, not a per-plot distance (every plot in the same compartment shares one value)"
    ),
    "dist_to_road": (
        "own calculation: distance from the plot point to the nearest OS Open Roads line, same "
        "distance-to-nearest-line method as dist_to_watercourse -- includes unclassified tracks, "
        "not just A/B roads"
    ),
    "dist_to_watercourse": (
        "own calculation: distance from the plot point to the nearest OS Open Rivers "
        "watercourse line (real vector survey data, not a flow-accumulation derivation), "
        "watercourses read within a 5km-buffered bounding box around the study area"
    ),
    "gwa_wind_speed_10m": "external (Global Wind Atlas API)",
    "soilgrids_ph": "external (SoilGrids COG, ISRIC)",
    "ceh_pedotope": "external (CEH natural-capital pedotope raster -- a categorical class ID, not an ordinal scale)",
    "ceh_twi": "external (CEH Topographic Wetness Index raster, from the same CEH zip as ceh_pedotope)",
    "ceh_subsurface_drainage": "external (CEH subsurface drainage raster -- a categorical class ID, from the same CEH zip)",
    "ceh_textural_composition": "external (CEH soil textural composition raster -- a categorical class ID, from the same CEH zip)",
    "chelsa_bio1_celsius": "external (CHELSA bioclimatic layer)",
    "chelsa_gdd5_degc": "external (CHELSA growing degree days above 5C, BIOCLIM+ layer)",
    "chelsa_bio12_precip_mm": "external (CHELSA annual precipitation, bio12)",
    "haduk_tas_2021_mean": (
        "external (HadUK-Grid, CEDA, manually downloaded -- 2021 only, not every survey year). "
        "KNOWN LIMITATION, not yet fixed: every survey year currently reuses this same single "
        "2021 raster rather than its own year's climate -- downloading one HadUK-Grid tas raster "
        "per survey year (2002/2006/2008/2012/2021/2023) and joining each plot-year to its own "
        "year's value would remove this approximation"
    ),
    "neighbour_mean_height": (
        "own calculation: mean of every OTHER plot's 2023 elev_percentile_95th within a 75m "
        "radius (scipy.spatial.cKDTree query_ball_point) -- SPATIAL-LAG, not exogenous"
    ),
    "neighbour_height_differential": (
        "own calculation: own 2023 elev_percentile_95th minus neighbour_mean_height -- "
        "SPATIAL-LAG, not exogenous"
    ),
    "whcl": "external (raw GPKG windthrow hazard class field, integer 0-6, ordinal severity -- constant per plot across survey years, confirmed by direct check, not assumed)",
    # Silvicultural (stand-structure), not environmental -- kept in this same dict/set, not split
    # out separately (see progress_notes.md for why). Mean-aggregated per plot across survey years.
    # Age itself is deliberately NOT included here -- it's circular with mean_cr_residual's own
    # construction (the Chapman-Richards curve's only input), see progress_notes.md for the
    # age-binned residual check that found a real, non-monotonic fit bias XGBoost was exploiting.
    "CanopyCover": "external (raw survey field, mean canopy cover fraction across every survey year)",
    "Thin": "external (raw survey field, fraction of surveys where this plot had been thinned)",
    "time_since_thinning": "external (raw survey field, NaN for never-thinned filled with 0 before averaging -- time_since_thinning_missing carries the real signal for those plots)",
    "time_since_thinning_missing": "external (raw survey field, fraction of surveys where this plot had never yet been thinned)",
    "recent_thinning_5yr": "external (raw survey field, fraction of surveys with a thinning in the preceding 5 years)",
}
# era5_land_temp_k is deliberately NOT in this dict, and so not in any feature set below --
# excluded on the numbers alone (only 8 distinct values across all 71,766 plots, confirmed
# ~11.1km native resolution), not grouped with haduk_tas_2021_mean by category. haduk_tas_2021_mean
# has 149 distinct values and one of the strongest raw correlations with the CR residual of
# anything tested (rho=0.29) -- kept, on its own numbers, despite sharing the same "single-year
# climate snapshot" shape as the excluded ERA5-Land variable. See
# notebooks/environmental_data/aux_data_resolution_check.ipynb's climate correlation section for
# the full comparison.

# aspect_degrees is deliberately excluded (a raw 0-360 bearing is a bad model input -- see
# progress_notes.md for the full reasoning); northness/eastness are its fixed replacement.
# Redundant-looking pairs otherwise stay in raw -- SHAP evidence decides what's redundant, not a
# pre-judgement here.
ALL_FEATURE_COLUMNS = list(FEATURE_PROVENANCE.keys())

NEIGHBOUR_COLUMNS = ["neighbour_mean_height", "neighbour_height_differential"]

TERRAIN_AND_WIND_COLUMNS = [
    "elevation", "slope_degrees", "northness", "eastness",
    "profile_curvature", "plan_curvature", "tpi", "elevation_roughness", "ceh_twi",
    "solar_radiation_index", "soil_depth_proxy", "frost_hollow_flag",
    "topex", "windward_topex", "whcl",
    "dist_to_cpmt_boundary", "dist_to_forest_perimeter", "dist_to_watercourse",
    "dist_to_scpt_boundary", "dist_to_block_boundary", "cpmt_compactness_ratio", "dist_to_road",
    "gwa_wind_speed_10m",
]

# Three named feature sets to fit and compare:
#   - all_environmental: every candidate variable, the main deliverable.
#   - all_environmental_no_neighbour: same, minus the two spatial-lag features -- answers the
#     standing caveat ("test the terrain+wind model with and without this feature") directly.
#   - terrain_and_wind_only: matches this repo's dissertation plan's original XGB-A/B framing,
#     kept as a continuity check against that earlier design.
FEATURE_SETS = {
    "all_environmental": ALL_FEATURE_COLUMNS,
    "all_environmental_no_neighbour": [c for c in ALL_FEATURE_COLUMNS if c not in NEIGHBOUR_COLUMNS],
    "terrain_and_wind_only": TERRAIN_AND_WIND_COLUMNS,
}


def fit_with_columns(train_df, feature_columns, target_col="mean_cr_residual", seed=42):
    # The actual fitting logic, taking a raw column list directly rather than a name looked up
    # in FEATURE_SETS -- lets ad-hoc analysis (e.g. the Tier-2 notebook's ablation tests, which
    # need many one-off column combinations for cluster testing) fit a model without adding a
    # new named entry to FEATURE_SETS for every combination tried. Plain XGBoost regressor,
    # sklearn defaults otherwise (no tuning yet) -- this is a baseline reference point for
    # feature importance, not a tuned model, same "not tuned yet" honesty as
    # models/rf_baseline/rf_baseline.py.
    features_train = train_df[feature_columns]

    model = xgb.XGBRegressor(random_state=seed)
    model.fit(features_train, train_df[target_col])

    return model


def fit(train_df, feature_set_name, target_col="mean_cr_residual", seed=42):
    # Thin wrapper for the three named, permanent feature sets above -- everything else
    # (run_xgb_environmental.py) keeps calling this exactly as before.
    feature_columns = FEATURE_SETS[feature_set_name]
    return fit_with_columns(train_df, feature_columns, target_col=target_col, seed=seed)


def predict_with_columns(df, model, feature_columns):
    return model.predict(df[feature_columns])


def predict(df, model, feature_set_name):
    feature_columns = FEATURE_SETS[feature_set_name]
    return predict_with_columns(df, model, feature_columns)


def compute_shap_values(model, df, feature_set_name):
    # TreeExplainer works directly on a fitted tree model -- no PINN, no neural network needed
    # anywhere in this chain. Returns one row per plot, one column per feature: exactly the
    # per-plot, per-variable "how much did this push the prediction up or down here" data a
    # future interactive attribution map would need.
    feature_columns = FEATURE_SETS[feature_set_name]
    features = df[feature_columns]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Built as a fresh float DataFrame from the raw shap_values array, rather than copying
    # `features` and overwriting its values in place -- `whcl` is an integer column, and SHAP
    # values are floats, so writing floats into a copy of an int column raised
    # "Invalid value ... for dtype 'int32'" the first time a non-float feature was added here.
    shap_df = pd.DataFrame(shap_values, columns=feature_columns, index=features.index)
    shap_df.insert(0, "identification", df["identification"].values)
    return shap_df

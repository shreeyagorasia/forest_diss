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
    "slope_degrees": "own calculation (gradient of the OS Terrain 50 DTM)",
    "northness": "own calculation, derived from aspect_degrees (cos)",
    "eastness": "own calculation, derived from aspect_degrees (sin)",
    "profile_curvature": "own calculation (second derivative of the DTM, Zevenbergen & Thorne 1987)",
    "plan_curvature": "own calculation (second derivative of the DTM, Zevenbergen & Thorne 1987)",
    "tpi": "own calculation, derived from elevation minus its own local neighbourhood mean",
    "elevation_roughness": "own calculation (moving-window std of the DTM)",
    "solar_radiation_index": "own calculation, derived from slope_degrees and aspect_degrees",
    "soil_depth_proxy": "own calculation, derived directly from slope_degrees (= -slope_degrees, a heuristic, not new information)",
    "frost_hollow_flag": "own calculation, derived from tpi and plan_curvature",
    "topex": "own calculation (horizon-angle sum from the OS Terrain 50 DTM, Wilson 1984 method)",
    "windward_topex": "own calculation, reuses the same topex() function as topex, restricted to one compass direction",
    "dist_to_cpmt_boundary": "own calculation (GPKG compartment polygon geometry)",
    "dist_to_forest_perimeter": "own calculation (GPKG compartment polygons, dissolved and morphologically closed)",
    "dist_to_scpt_boundary": "own calculation (GPKG sub-compartment polygon geometry, dissolved by cpmt+scpt together)",
    "dist_to_block_boundary": "own calculation (GPKG block polygon geometry, dissolved by blk)",
    "cpmt_compactness_ratio": "own calculation (compartment polygon perimeter / area -- a per-compartment shape property, not a per-plot distance)",
    "dist_to_watercourse": "own calculation, combines external vector data (OS Open Rivers) with plot geometry",
    "dist_to_road": "own calculation, combines external vector data (OS Open Roads) with plot geometry",
    "gwa_wind_speed_10m": "external (Global Wind Atlas API)",
    "soilgrids_ph": "external (SoilGrids COG, ISRIC)",
    "ceh_pedotope": "external (CEH natural-capital pedotope raster -- a categorical class ID, not an ordinal scale)",
    "ceh_twi": "external (CEH Topographic Wetness Index raster, from the same CEH zip as ceh_pedotope)",
    "ceh_subsurface_drainage": "external (CEH subsurface drainage raster -- a categorical class ID, from the same CEH zip)",
    "ceh_textural_composition": "external (CEH soil textural composition raster -- a categorical class ID, from the same CEH zip)",
    "chelsa_bio1_celsius": "external (CHELSA bioclimatic layer)",
    "chelsa_gdd5_degc": "external (CHELSA growing degree days above 5C, BIOCLIM+ layer)",
    "chelsa_bio12_precip_mm": "external (CHELSA annual precipitation, bio12)",
    "haduk_tas_2021_mean": "external (HadUK-Grid, CEDA, manually downloaded -- 2021 only, not every survey year)",
    "neighbour_mean_height": "own calculation, derived from plot geometry + measured height -- SPATIAL-LAG, not exogenous",
    "neighbour_height_differential": "own calculation, derived from own height minus neighbour_mean_height -- SPATIAL-LAG, not exogenous",
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
    # yldc deliberately NOT included -- removed as a feature everywhere in this rebuild (29 July
    # 2026): real held-out ablation showed it hurts generalisation in every model tested (RF,
    # DNN, this xgb_environmental model included), not just theory. See progress_notes.md.
}
# era5_land_temp_k is deliberately NOT in this dict, and so not in any feature set below --
# excluded on the numbers alone (only 8 distinct values across all 71,766 plots, confirmed
# ~11.1km native resolution), not grouped with haduk_tas_2021_mean by category. haduk_tas_2021_mean
# has 149 distinct values and one of the strongest raw correlations with the CR residual of
# anything tested (rho=0.29) -- kept, on its own numbers, despite sharing the same "single-year
# climate snapshot" shape as the excluded ERA5-Land variable. See
# notebooks/environmental_data/aux_data_resolution_check.ipynb's climate correlation section for
# the full comparison.

# aspect_degrees itself is deliberately left OUT of every feature set below: it's a raw compass
# bearing (0-360), so 359 and 1 are almost the same direction but numerically far apart -- a bad
# input for any model to split on directly. northness/eastness (already in FEATURE_PROVENANCE
# above) are its fixed replacement and are used instead. This is the only column excluded here
# for a data-validity reason -- everything else, including soil_depth_proxy (an exact transform
# of slope_degrees), goes in raw. Deciding what's redundant is the new Tier-2 notebook's job,
# using real SHAP evidence -- not pre-judged in this file.
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

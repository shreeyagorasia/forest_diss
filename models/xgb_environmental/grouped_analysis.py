import contextlib
import io

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from models.xgb_environmental.xgb_environmental import ALL_FEATURE_COLUMNS

# Domain categories -- what KIND of variable each one is (terrain/wind/soil/climate/etc), based
# on forestry domain knowledge of what actually drives tree growth. A different grouping to
# FEATURE_SETS (which group by "which variables are included at all") or the correlated-cluster
# grouping used elsewhere (which groups by measured correlation, not domain meaning). Every
# column in ALL_FEATURE_COLUMNS must appear in exactly one category here -- checked below.
CATEGORY_GROUPS = {
    "terrain": [
        "elevation", "slope_degrees", "northness", "eastness", "profile_curvature",
        "plan_curvature", "tpi", "elevation_roughness", "solar_radiation_index",
        "soil_depth_proxy", "frost_hollow_flag", "ceh_twi",
    ],
    "wind": ["gwa_wind_speed_10m", "topex", "windward_topex", "whcl"],
    "soil_site": [
        "soilgrids_ph", "ceh_pedotope", "ceh_subsurface_drainage",
        "ceh_textural_composition", "dist_to_watercourse",
    ],
    "climate": ["haduk_tas_2021_mean", "chelsa_bio1_celsius", "chelsa_gdd5_degc", "chelsa_bio12_precip_mm"],
    # A real, distinct forestry mechanism -- edge trees get more wind exposure and more light
    # than interior trees. Kept separate from "wind" (a different, GPKG-geometry-derived signal,
    # not a wind measurement) and separate from "neighbour_spatial_lag" below (a much less
    # trustworthy category, see its own note). Three spatial scales of "distance to an edge"
    # (sub-compartment, compartment, block/forest), one compartment-shape property
    # (`cpmt_compactness_ratio` -- perimeter/area, not a per-plot distance), and one real
    # physical opening in the canopy (`dist_to_road`, OS Open Roads).
    "spatial_position_edge_effects": [
        "dist_to_cpmt_boundary", "dist_to_forest_perimeter", "dist_to_scpt_boundary",
        "dist_to_block_boundary", "cpmt_compactness_ratio", "dist_to_road",
    ],
    "stand_structure": [
        "CanopyCover", "Thin", "time_since_thinning",
        "time_since_thinning_missing", "recent_thinning_5yr",
    ],
    # Kept separate from spatial_position_edge_effects on purpose: these two describe a NEARBY
    # PLOT'S OWN HEIGHT, not a real environmental measurement -- most likely reflecting shared
    # planting-year/management history between nearby plots, not a real wind-shielding or
    # competition effect (see progress_notes.md). A genuinely different trust level to the
    # edge-effect variables above, so grouping them together would hide that distinction.
    "neighbour_spatial_lag": ["neighbour_mean_height", "neighbour_height_differential"],
}


def check_category_groups_complete():
    # Runs at import time so a forgotten new variable (e.g. a future addition to
    # FEATURE_PROVENANCE) is caught immediately, not silently missing from every
    # category-level analysis in the new notebook.
    assigned_columns = [column for columns in CATEGORY_GROUPS.values() for column in columns]
    missing_columns = [c for c in ALL_FEATURE_COLUMNS if c not in assigned_columns]
    duplicated_columns = [c for c in assigned_columns if assigned_columns.count(c) > 1]
    if missing_columns:
        raise ValueError(f"Columns in ALL_FEATURE_COLUMNS but not in any CATEGORY_GROUPS: {missing_columns}")
    if duplicated_columns:
        raise ValueError(f"Columns assigned to more than one category: {set(duplicated_columns)}")


check_category_groups_complete()


def cross_category_correlations(df, category_groups=None, threshold=0.3):
    # Flags any pair of variables from TWO DIFFERENT categories whose Spearman correlation is at
    # least `threshold` -- context for reading any category-level importance number. A category
    # can look unimportant in an ablation/permutation test purely because a correlated variable
    # in a DIFFERENT category is compensating for it (e.g. elevation standing in for wind
    # exposure) -- this table makes that risk visible instead of silent. threshold=0.3 is
    # deliberately looser than the 0.9 used elsewhere for within-model "near-duplicate" pairs --
    # cross-category coupling is expected to be weaker but still real enough to matter.
    if category_groups is None:
        category_groups = CATEGORY_GROUPS

    category_names = list(category_groups.keys())
    flagged_pairs = []
    for i, category_a in enumerate(category_names):
        for category_b in category_names[i + 1:]:
            for column_a in category_groups[category_a]:
                for column_b in category_groups[category_b]:
                    paired_rows = df[[column_a, column_b]].dropna()
                    rho, p_value = spearmanr(paired_rows[column_a], paired_rows[column_b])
                    if abs(rho) >= threshold:
                        flagged_pairs.append({
                            "category_a": category_a, "variable_a": column_a,
                            "category_b": category_b, "variable_b": column_b,
                            "rho": rho, "p_value": p_value,
                        })

    columns = ["category_a", "variable_a", "category_b", "variable_b", "rho", "p_value"]
    if not flagged_pairs:
        return pd.DataFrame(columns=columns)
    flagged_df = pd.DataFrame(flagged_pairs)
    return flagged_df.reindex(flagged_df["rho"].abs().sort_values(ascending=False).index)


def grouped_permutation_importance(model, eval_df, feature_columns, group_columns, predict_fn, target_col="mean_cr_residual", n_repeats=10, seed=42):
    # Shuffles every column in group_columns TOGETHER, using the SAME random row order for all of
    # them at once -- this breaks the group's relationship to the target (and to every other
    # feature) while keeping the group's own internal correlation structure intact. Shuffling
    # each column separately would create unrealistic combinations that never occur in the real
    # data (e.g. a high elevation paired with a wind speed that never actually happens together
    # at high elevation).
    #
    # Uses the model ALREADY FIT on the real data -- no refitting needed, unlike section 6's
    # ablation refits. Answers a different question: "how much does the model's actual score
    # drop if this group's real values are replaced with a random, unrelated shuffling of
    # themselves" -- not "how much predictive capacity is lost if this group were genuinely
    # unavailable to refit with" (what an ablation refit measures).
    from models.common.metrics import compute_metrics

    rng = np.random.default_rng(seed)

    baseline_predictions = predict_fn(eval_df, model, feature_columns)
    baseline_r2 = compute_metrics(eval_df[target_col].values, baseline_predictions)["r2"]

    r2_drops = []
    for _ in range(n_repeats):
        shuffled_df = eval_df.copy()
        shuffle_order = rng.permutation(len(shuffled_df))
        shuffled_df[group_columns] = shuffled_df[group_columns].to_numpy()[shuffle_order]

        shuffled_predictions = predict_fn(shuffled_df, model, feature_columns)
        shuffled_r2 = compute_metrics(shuffled_df[target_col].values, shuffled_predictions)["r2"]
        r2_drops.append(baseline_r2 - shuffled_r2)

    return {
        "baseline_r2": baseline_r2,
        "mean_r2_drop": float(np.mean(r2_drops)),
        "std_r2_drop": float(np.std(r2_drops)),
        "n_repeats": n_repeats,
    }


def residual_morans_i(train_df, eval_df, feature_columns, fit_fn, predict_fn, target_col="mean_cr_residual", max_distance=5000):
    # Fits fresh on feature_columns, then measures whether the resulting residuals are still
    # spatially clustered. semivariogram_range() finds the real distance scale to test at
    # first -- picking an arbitrary distance (or a fixed neighbour-count) risks testing "is this
    # plot like the plot right next to it" instead of real regional clustering (see
    # models/spatial_attribution/spatial_autocorrelation.py's own header comment for why).
    from models.spatial_attribution.spatial_autocorrelation import global_morans_i, semivariogram_range

    model = fit_fn(train_df, feature_columns, target_col=target_col)
    predictions = predict_fn(eval_df, model, feature_columns)
    residuals = eval_df[target_col].values - predictions

    range_estimate, status = semivariogram_range(
        eval_df["x"].values, eval_df["y"].values, residuals, max_distance=max_distance,
    )
    if status not in ("resolved", "exceeds_window"):
        return {"morans_i": np.nan, "morans_i_p": np.nan, "variogram_status": status, "variogram_range_m": range_estimate}

    # Aberfoyle is genuinely several separate forest blocks (see
    # spatial_autocorrelation.py's own header comment) -- libpysal prints one "island (no
    # neighbors)" line per isolated plot directly to stdout (not a catchable Python warning),
    # which can run to thousands of lines. Expected, not an error -- suppressed here so it
    # doesn't flood notebook output.
    with contextlib.redirect_stdout(io.StringIO()):
        morans_i, morans_p = global_morans_i(
            eval_df["x"].values, eval_df["y"].values, residuals, distance=range_estimate,
        )
    return {"morans_i": morans_i, "morans_i_p": morans_p, "variogram_status": status, "variogram_range_m": range_estimate}


def category_morans_i_before_after(train_df, eval_df, fit_fn, predict_fn, all_columns=None, category_groups=None, target_col="mean_cr_residual"):
    # "Before" = the full model's residuals (every category still in). "After" = residuals once
    # ONE category has been removed and the model refit without it. If removing a category makes
    # Moran's I go UP, that category was explaining real spatial structure -- the residuals left
    # behind once it's gone are more spatially clustered than before, i.e. real spatial pattern
    # remains unexplained. If Moran's I barely changes, that category wasn't the one responsible
    # for whatever spatial structure exists in the residuals.
    if all_columns is None:
        all_columns = ALL_FEATURE_COLUMNS
    if category_groups is None:
        category_groups = CATEGORY_GROUPS

    rows = []
    full_result = residual_morans_i(train_df, eval_df, all_columns, fit_fn, predict_fn, target_col)
    rows.append({"category_removed": "(none -- full model)", **full_result})

    for category_name, category_columns in category_groups.items():
        columns_without_category = [c for c in all_columns if c not in category_columns]
        result = residual_morans_i(train_df, eval_df, columns_without_category, fit_fn, predict_fn, target_col)
        rows.append({"category_removed": category_name, **result})

    return pd.DataFrame(rows)

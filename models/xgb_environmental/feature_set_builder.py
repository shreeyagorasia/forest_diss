"""Builds the "Set1-5" nested feature hierarchy, shared across three different questions:

  - RSQ1: Avenue 1's predictive tiers (feeds dnn_env_terrain/pinn_env_terrain*, target
    elev_percentile_95th -- see models/common/torch_data.py::ENV_TERRAIN_FEATURE_SETS).
  - RSQ2: Avenue 1's attribution screen (feeds Elastic Net/XGBoost/NLME, target
    mean_cr_residual -- see models/xgb_environmental/xgb_environmental.py::FEATURE_SETS).
  - RSQ3: Avenue 2's scope system (feeds Elastic Net/XGBoost/GNNWR, target
    local_y_max_difference -- see models/growth_curve_attribution/broad_environmental_check.py).

Same five-set recipe every time, only the target column and candidate pool change per RSQ:

  Set1 = baseline only (the 5 stand/management columns every model already gets).
  Set2 = Set1 + top 5 deduplicated candidates, ranked by |Spearman rho| against THIS RSQ's
         own target.
  Set3 = Set1 + terrain/wind candidates that individually clear a 0.10 |rho| signal gate.
  Set4 = Set1 + every remaining category's candidates that individually clear the same gate.
  Set5 = Set1 + every deduplicated candidate, gate not applied.

Every function here takes target_column as an explicit argument -- never a shared module-level
constant -- specifically so RSQ1/RSQ2/RSQ3 can never accidentally rank against each other's
target by reusing the wrong constant (a real bug found in the existing stage1-4 tiers: they were
built ranking against mean_cr_residual even though they feed a model that predicts
elev_percentile_95th, a completely different target).

The dedup step (drop_deterministic_duplicates, find_near_exact_duplicates,
choose_representative_loser) and the VIF step (iterative_vif_reduction) are NOT reimplemented
here -- they're imported as-is from models/xgb_environmental/multicollinearity_screen.py, the
same shared module Avenue 2's own screening notebook already imports from. This file only adds
the layer on top: ranking, per-column gating, and Set1-5 assembly.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from models.common.metrics import compute_metrics
from models.common.splits import SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, SPLIT_SEED, spatial_block_split
from models.xgb_environmental.multicollinearity_screen import (
    choose_representative_loser,
    compute_correlation_matrix,
    compute_vif_table,
    drop_deterministic_duplicates,
    find_near_exact_duplicates,
)
from models.xgb_environmental.grouped_analysis import CATEGORY_GROUPS
from models.xgb_environmental.xgb_environmental import fit_with_columns as xgb_fit_with_columns
from models.xgb_environmental.xgb_environmental import predict_with_columns as xgb_predict_with_columns

# Set1's baseline columns. Starts from CATEGORY_GROUPS["stand_structure"] (already the single
# existing definition of these 5 columns -- confirmed identical to broad_environmental_check.py's
# own MANAGEMENT_COLUMNS), with two further exclusions on top:
#
# 1. thinning_status deliberately NOT included, matching existing precedent: both
#    CATEGORY_GROUPS["stand_structure"] and MANAGEMENT_COLUMNS already exclude it, since it's a
#    derived re-bucketing of time_since_thinning/recent_thinning_5yr/Thin (already in this list)
#    -- adding it too would feed overlapping information twice.
# 2. Thin deliberately NOT included -- confirmed, directly against the real plot-level data (not
#    assumed from the VIF number alone), that Thin + time_since_thinning_missing == 1.0 EXACTLY
#    for all 71,766 plots in plot_environmental_features.parquet, no exceptions. This is why the
#    first version of run_rsq2_vif_pass below saw VIF=inf for these two: it's a genuine
#    deterministic duplicate, the same kind of relationship DETERMINISTIC_DUPLICATES in
#    multicollinearity_screen.py already documents for other pairs (e.g. inverse_slope_proxy =
#    -slope_degrees), just never previously written down for this pair. time_since_thinning_missing
#    is kept, not Thin: its actual job is as a missingness flag paired with time_since_thinning
#    (torch_data.py's fill_missing_time_since_thinning -- "the missingness flag is what actually
#    tells the model this is a placeholder, not a real elapsed time"), so time_since_thinning +
#    time_since_thinning_missing together already carry everything Thin does, losslessly. This
#    only changes what this module unions onto Set1-5 -- it does NOT touch the existing production
#    DNN/PINN pipeline in torch_data.py, which still feeds both columns through its own separate
#    NUMERIC_SCALED_COLUMNS/BINARY_PASSTHROUGH_COLUMNS pathway (a pre-existing, separate design
#    decision, out of scope for this fix -- flagged, not silently changed).
SET1_BASELINE_COLUMNS = [
    column for column in CATEGORY_GROUPS["stand_structure"] if column != "Thin"
]

# The gate used for Set3/Set4. Deliberately low ("carries any real individual signal at all", not
# "is a strong predictor on its own") -- same threshold and same reasoning as the existing
# STAGE3_SIGNAL_THRESHOLD in notebooks/environmental_data/multicollinearity_screen_av1.ipynb.
SIGNAL_GATE_THRESHOLD = 0.10


def dedup_candidates(df, candidate_columns, feature_provenance, target_column, near_exact_threshold=0.95):
    """Stages 1-2 only: drop deterministic duplicates, then drop near-exact empirical duplicates
    (|Spearman rho| >= near_exact_threshold). No VIF here -- VIF is a separate, later step, and
    (per the project's own design) only RSQ2 ever needs it (see run_rsq2_vif_pass below).

    This is the exact same two-stage logic as the existing multicollinearity_screen_av1.ipynb's
    own dedup_only() helper, promoted here into a reusable function so RSQ1/RSQ2/RSQ3 can each
    call the identical code instead of three separate near-identical copies.

    Returns (kept_columns, drop_log_rows).
    """
    drop_log_rows = []

    after_stage1, stage1_drops = drop_deterministic_duplicates(candidate_columns)
    drop_log_rows.extend(stage1_drops)

    correlation_matrix = compute_correlation_matrix(df, after_stage1)
    near_exact_pairs = find_near_exact_duplicates(correlation_matrix, threshold=near_exact_threshold)

    kept = list(after_stage1)
    for column_a, column_b, rho in near_exact_pairs:
        # A column may already have been dropped by an earlier pair in this same loop (e.g. a
        # three-way near-exact cluster) -- skip pairs where that's already happened.
        if column_a not in kept or column_b not in kept:
            continue
        losers = choose_representative_loser([column_a, column_b], feature_provenance, df, target_column)
        for loser in losers:
            kept.remove(loser)
            drop_log_rows.append({
                "column": loser, "stage": "2_near_exact_duplicate",
                "reason": f"rho={rho:.3f} with its kept partner",
            })

    return kept, drop_log_rows


def rank_by_target_correlation(df, candidate_columns, target_column):
    """Univariate |Spearman rho| of every candidate column against target_column, sorted
    strongest first.

    Uses nan_policy="omit" throughout, matching multicollinearity_screen.py's own convention
    (choose_representative_loser already uses this). Deliberately NOT
    explain_signal.spearman_with_target -- that function relies on the caller having already
    dropped every NaN row first (an implicit precondition, fine for its one call site) -- unsafe
    to reuse generically here across three tables with different missingness patterns.
    """
    rows = []
    for column in candidate_columns:
        rho, _ = spearmanr(df[column], df[target_column], nan_policy="omit")
        rows.append({"variable": column, "spearman_rho": rho, "abs_rho": abs(rho)})
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False).reset_index(drop=True)


def gate_columns_by_signal(df, candidate_columns, target_column, threshold=SIGNAL_GATE_THRESHOLD):
    """Per-column signal gate: a column only passes if ITS OWN |Spearman rho| with target_column
    is >= threshold.

    Deliberately per-column, not per-category. The existing AV1 notebook's own
    category_signal_strength() lets a whole category in based on its single strongest column --
    fine for a coarse "does this category carry any signal at all" question, but Set3/Set4 here
    are meant to keep the pool small ("fewer variables, to rule out multicollinearity"), so a weak
    variable inside an otherwise-strong category should not ride along just because a neighbour in
    the same category is strong.

    Returns (passed_columns, rho_table) -- rho_table has every candidate's own rho, not just the
    ones that passed, so a human reading the notebook can see exactly what just missed the cut.
    """
    rho_table = rank_by_target_correlation(df, candidate_columns, target_column)
    passed = rho_table.loc[rho_table["abs_rho"] >= threshold, "variable"].tolist()
    return passed, rho_table


def _spatial_holdout_split(df, seed=SPLIT_SEED):
    """One spatial_block_split (a single train/test partition, NOT the project's real pooled
    5-fold spatial CV) -- deliberately cheaper, screening-only. The Set eventually CHOSEN from
    this notebook still gets evaluated properly under the project's real 5-fold spatial CV later
    (that's a separate, later step -- this is just deciding which candidates are worth including
    in the first place). Still spatially blocked, not a plain random split, matching this
    project's own standard discipline throughout (a random split lets a model see near-identical
    neighbouring plots on both sides of the split, overstating how well it generalises).
    """
    labelled_df = df.copy()
    labelled_df["split"] = spatial_block_split(
        labelled_df, block_col=SPATIAL_BLOCK_COL, buffer_distance=SPATIAL_BUFFER_METRES, seed=seed,
    )
    train_df = labelled_df[labelled_df["split"] == "train"]
    test_df = labelled_df[labelled_df["split"] == "test"]
    return train_df, test_df


def permutation_importance_ranking(df, candidate_columns, baseline_columns, target_column, control_columns=None, seed=SPLIT_SEED):
    """Fits ONE reference XGBoost model (baseline + control + every candidate together) on a
    spatial train/test split, then for each candidate column, shuffles that column's values in
    the test set and re-predicts with the SAME already-fitted model (no retraining) -- the drop
    in test R2 is that column's permutation importance. XGBoost is used as a common reference
    model across all three RSQs: fast, handles both continuous and categorical (one-hot) columns
    natively, and already the shared model family used elsewhere in this project's own
    attribution tooling (explain_signal.py).

    control_columns (default: none) are included in the reference model's design matrix, exactly
    like baseline_columns, but are NEVER permuted/ablated themselves and never appear in the
    returned ranking -- they exist purely so importance gets measured "given this column is
    already in the model," not in a model missing it. RSQ1 needs this for Age: the first real run
    of this notebook, without Age in the reference model at all, got a reference R2 of -0.168
    (worse than predicting the mean) on elev_percentile_95th (raw height) -- Age is this
    project's own single dominant predictor of height (~63% of variance elsewhere in this
    project's notes), so a model missing it entirely is badly misspecified, and every
    permutation/ablation number computed against it is unreliable. Age is NOT circular for RSQ1
    the way it is for RSQ2/RSQ3 (whose targets are residuals already built FROM Age via the CR
    curve) -- it only needs to be excluded from what gets UNIONED into the exported Set2-5 lists
    (RSQ1's real dnn_env_terrain model already gets Age through its own separate pathway), not
    from the reference model used to rank everything else correctly.

    Known limitation, disclosed rather than hidden: permutation importance is biased under
    correlated predictors (Strobl et al. 2008) -- shuffling one column while its correlated
    partners stay fixed creates combinations that never occur in the real data, which can over-
    or under-state a column's importance. Dedup (stages 1-2) already removes near-exact
    duplicates before this runs, which reduces but does not eliminate the risk for the more
    loosely correlated columns that survive dedup. Reported alongside drop-column ablation and
    Spearman ranking specifically so disagreement between methods is visible, not hidden inside
    one number treated as ground truth.

    Returns (ranked_table, baseline_r2) -- ranked_table sorted by r2_drop_from_permutation, most
    important first.
    """
    control_columns = list(control_columns) if control_columns else []
    all_columns = list(baseline_columns) + control_columns + list(candidate_columns)
    clean_df = df.dropna(subset=all_columns + [target_column])
    train_df, test_df = _spatial_holdout_split(clean_df, seed=seed)

    model = xgb_fit_with_columns(train_df, all_columns, target_col=target_column, seed=seed)
    baseline_predictions = xgb_predict_with_columns(test_df, model, all_columns)
    baseline_r2 = compute_metrics(test_df[target_column].values, baseline_predictions)["r2"]

    rng = np.random.default_rng(seed)
    rows = []
    for column in candidate_columns:
        shuffled_test_df = test_df.copy()
        shuffled_test_df[column] = rng.permutation(shuffled_test_df[column].to_numpy())
        shuffled_predictions = xgb_predict_with_columns(shuffled_test_df, model, all_columns)
        shuffled_r2 = compute_metrics(test_df[target_column].values, shuffled_predictions)["r2"]
        rows.append({"variable": column, "r2_drop_from_permutation": baseline_r2 - shuffled_r2})

    ranked_table = pd.DataFrame(rows).sort_values("r2_drop_from_permutation", ascending=False).reset_index(drop=True)
    return ranked_table, baseline_r2


def drop_column_ablation_ranking(df, candidate_columns, baseline_columns, target_column, control_columns=None, seed=SPLIT_SEED):
    """Fits ONE reference XGBoost model on baseline + control + every candidate together (same
    spatial split, same seed, and the same reference model family as
    permutation_importance_ranking, so the two are directly comparable), then for EACH candidate
    column, refits WITHOUT that one column (control_columns and baseline_columns always stay in)
    and measures the drop in test R2 against the full model. Slower than permutation importance
    (one full refit per candidate, not just a reshuffle) but answers a more direct question: does
    the model's actual held-out performance get worse without this column, given everything else
    already in the model -- the "remove one and see what happens" check.

    control_columns: see permutation_importance_ranking's own docstring -- same purpose (RSQ1
    needs Age here for the same reason), same guarantee (never itself ablated, never in the
    returned ranking).

    Returns (ranked_table, full_r2) -- ranked_table sorted by r2_drop_from_removal, most
    important first (removing it hurt the most).
    """
    control_columns = list(control_columns) if control_columns else []
    all_columns = list(baseline_columns) + control_columns + list(candidate_columns)
    clean_df = df.dropna(subset=all_columns + [target_column])
    train_df, test_df = _spatial_holdout_split(clean_df, seed=seed)

    full_model = xgb_fit_with_columns(train_df, all_columns, target_col=target_column, seed=seed)
    full_predictions = xgb_predict_with_columns(test_df, full_model, all_columns)
    full_r2 = compute_metrics(test_df[target_column].values, full_predictions)["r2"]

    rows = []
    for column in candidate_columns:
        reduced_columns = [c for c in all_columns if c != column]
        reduced_model = xgb_fit_with_columns(train_df, reduced_columns, target_col=target_column, seed=seed)
        reduced_predictions = xgb_predict_with_columns(test_df, reduced_model, reduced_columns)
        reduced_r2 = compute_metrics(test_df[target_column].values, reduced_predictions)["r2"]
        rows.append({"variable": column, "r2_drop_from_removal": full_r2 - reduced_r2})

    ranked_table = pd.DataFrame(rows).sort_values("r2_drop_from_removal", ascending=False).reset_index(drop=True)
    return ranked_table, full_r2


def combine_importance_ranks(spearman_table, permutation_table, ablation_table):
    """Combines all three importance signals into one rank-aggregated table -- an item has to
    look important across MULTIPLE different lenses to rank highly overall, not just fool one
    particular method's own blind spot (Spearman ignores context/interactions; permutation is
    biased under correlation; drop-column ablation depends on one model/split). Rank aggregation
    (not averaging the raw numbers) is used because the three signals are on completely different
    scales (a correlation coefficient vs. an R2 drop) -- ranks are the one thing directly
    comparable across all three.

    Returns one row per variable: each method's own raw score, each method's own rank (1 = most
    important), and average_rank across all three -- sorted by average_rank ascending (most
    important first). build_set2/gate_columns_by_combined_rank below both just read the
    "variable" column off the top of this table, in order, same interface as the old
    Spearman-only rho_table had.
    """
    merged = spearman_table[["variable", "abs_rho"]].merge(
        permutation_table[["variable", "r2_drop_from_permutation"]], on="variable", how="outer"
    ).merge(
        ablation_table[["variable", "r2_drop_from_removal"]], on="variable", how="outer"
    )
    # rank(ascending=False): the BIGGEST score gets rank 1 (most important), for all three
    # columns -- a bigger |rho|, a bigger permutation R2 drop, and a bigger ablation R2 drop all
    # mean "more important" in the same direction.
    merged["spearman_rank"] = merged["abs_rho"].rank(ascending=False)
    merged["permutation_rank"] = merged["r2_drop_from_permutation"].rank(ascending=False)
    merged["ablation_rank"] = merged["r2_drop_from_removal"].rank(ascending=False)
    merged["average_rank"] = merged[["spearman_rank", "permutation_rank", "ablation_rank"]].mean(axis=1)
    return merged.sort_values("average_rank", ascending=True).reset_index(drop=True)


def gate_columns_by_combined_rank(combined_table, candidate_columns, top_fraction=0.5):
    """Per-column gate for Set3/Set4, replacing the old fixed |rho|>=0.10 cutoff: a column passes
    if its average_rank places it in the top `top_fraction` of THIS category's own candidates --
    relative to its own category, not an arbitrary absolute threshold that has to be separately
    defended. Ties (e.g. a category of 3 candidates asking for the "top half") round UP, so a
    small category doesn't lose every member to integer rounding.

    Returns (passed_columns, category_table) -- category_table is combined_table filtered and
    re-sorted to just this category's own candidates, so a human can see the whole category's
    ranking, not just who passed.
    """
    category_table = combined_table[combined_table["variable"].isin(candidate_columns)].copy()
    category_table = category_table.sort_values("average_rank", ascending=True).reset_index(drop=True)
    cutoff_count = int(np.ceil(len(category_table) * top_fraction))
    passed = category_table.head(cutoff_count)["variable"].tolist()
    return passed, category_table


def build_set2(ranked_table, baseline_columns, top_n=5):
    """Set2 = baseline + top N deduplicated candidates by importance rank.

    Takes any already-sorted table with a "variable" column, most-important-first -- in practice
    combine_importance_ranks()'s combined table (Spearman + permutation + drop-column ablation),
    not a single method's own ranking, so the notebook only ever ranks each RSQ's candidate pool
    once and every Set2/Set3/Set4 downstream reads off the same combined ranking.
    """
    top_columns = ranked_table.head(top_n)["variable"].tolist()
    return list(baseline_columns) + top_columns


def build_set3(terrain_wind_gated_columns, baseline_columns):
    """Set3 = baseline + gated terrain/wind candidates only -- the narrowest environmental
    addition, aimed at keeping the variable count low to avoid multicollinearity."""
    return list(baseline_columns) + list(terrain_wind_gated_columns)


def build_set4(terrain_wind_gated_columns, other_gated_columns, baseline_columns):
    """Set4 = baseline + gated terrain/wind + gated everything else (climate, soil/site,
    edge-position, ...)."""
    return list(baseline_columns) + list(terrain_wind_gated_columns) + list(other_gated_columns)


def build_set5(dedup_survivors, baseline_columns):
    """Set5 = baseline + every deduplicated candidate, gate not applied -- the widest,
    "for the sake of proof" tier."""
    return list(baseline_columns) + list(dedup_survivors)


def run_rsq2_vif_pass(df, set_columns, protected_columns=None, threshold=5.0, max_iterations=50):
    """RSQ2-only extra step: iterative VIF reduction on top of the dedup+gate result.

    Elastic Net/NLME coefficients and SHAP are collinearity-sensitive in a way RSQ1's neural nets
    (which can absorb correlated raw inputs without their collinearity corrupting what gets
    reported) and RSQ3's whole-category scopes (a different question -- "does adding a category
    help", not a per-coefficient attribution) are not -- see the notebook's own markdown for the
    full reasoning.

    NOT a plain call to multicollinearity_screen.py's own iterative_vif_reduction() -- that
    function treats every column symmetrically and will happily drop a Set1 baseline column if
    ITS OWN VIF is worst. First real run of this notebook hit exactly that: Thin and
    time_since_thinning_missing (both baseline) came back VIF=inf against each other -- a plot
    never thinned in any survey has Thin=0 and time_since_thinning_missing=1 in EVERY one of its
    survey rows, so the two are close to perfect complements. Baseline is supposed to be present
    in every set unconditionally (the agreed Set1-5 design) -- silently dropping it here would
    break that promise. protected_columns (default: SET1_BASELINE_COLUMNS) are still included in
    the design matrix, so they still affect every OTHER column's VIF number correctly -- they are
    only ever exempt from being the one column removed each iteration.

    Returns (kept_columns, drop_log_rows) -- same shape as multicollinearity_screen.py's
    iterative_vif_reduction.
    """
    if protected_columns is None:
        protected_columns = SET1_BASELINE_COLUMNS
    protected_columns = set(protected_columns)

    remaining_columns = list(set_columns)
    drop_log_rows = []
    for _ in range(max_iterations):
        droppable_columns = [column for column in remaining_columns if column not in protected_columns]
        if len(droppable_columns) == 0 or len(remaining_columns) <= 2:
            # Nothing left that's allowed to be dropped, or too few columns left for VIF to mean
            # anything -- stop either way.
            break

        # compute_vif_table already sorts worst-first across ALL remaining_columns (protected and
        # not) -- filtering to droppable rows afterward, rather than computing VIF on only the
        # droppable subset, keeps every protected column in the design matrix so its collinearity
        # with the droppable columns is still fully accounted for.
        vif_table = compute_vif_table(df, remaining_columns)
        droppable_vif_table = vif_table[vif_table["column"].isin(droppable_columns)]
        worst_droppable_row = droppable_vif_table.iloc[0]
        if worst_droppable_row["vif"] <= threshold:
            break

        remaining_columns.remove(worst_droppable_row["column"])
        drop_log_rows.append({
            "column": worst_droppable_row["column"],
            "stage": "3_vif",
            "reason": (
                f"VIF={worst_droppable_row['vif']:.2f} against the other {len(remaining_columns)} "
                f"remaining candidates (threshold {threshold})"
            ),
        })

    return remaining_columns, drop_log_rows


def strip_baseline_for_export(full_columns, baseline_columns):
    """RSQ1 only: models/common/torch_data.py::assert_env_terrain_features_disjoint_from_noenv()
    rejects any terrain-tier feature list that repeats a baseline column, because RSQ1's baseline
    is already fed to the main network through a separate pathway (NUMERIC_SCALED_COLUMNS/
    BINARY_PASSTHROUGH_COLUMNS) -- listing it again here would feed the same information twice.
    Set1-5 are built WITH baseline included (so all three RSQs share one comparable recipe), then
    this strips it back out at the point of exporting an RSQ1 tier into ENV_TERRAIN_FEATURE_SETS.
    RSQ2/RSQ3 exports do NOT call this -- their models take one flat column list with no second
    baseline pathway, so baseline must stay in for management to have any effect there at all.
    """
    baseline_set = set(baseline_columns)
    return [column for column in full_columns if column not in baseline_set]

"""A reusable, mechanical multicollinearity screen for the environmental candidate pool.

This exists because the environmental variable pool (terrain, wind, climate, soil, forest-edge --
NOT the stand-metric/management columns already fed into every model's main network, see
models/common/torch_data.py's NUMERIC_SCALED_COLUMNS/BINARY_PASSTHROUGH_COLUMNS) grew ad hoc over
many sessions, with correlation only checked AFTER fitting, as interpretive context. Backwards.
A 2026-08-06 llm-council review of the first draft of this screen (documented in
documentation/experiment_log.md) found real problems with the original 4-stage design and asked
for two fixes, both implemented here:

  1. Stage 3's original tie-break rule ("prefer whichever variable already won in an earlier,
     unaudited result") was circular. It would permanently reward variables for having won an
     ad hoc round for reasons unconnected to their real explanatory value. Replaced with: prefer
     an externally-measured variable over a derived one, then prefer whichever has the higher
     |Spearman rho| with a real target computed FRESH, right now, inside this module.
  2. Running VIF only on the survivors of a separate pairwise-clustering step can hide genuine
     3-or-more-variable collinearity, because the evidence gets deleted before VIF ever sees it.
     Fixed by merging the old "stage 3" (pairwise clustering) and "stage 4" (VIF) into ONE
     iterative VIF pass over the full post-deduplication pool.

So the real pipeline here is three stages, not four:
  Stage 1. Drop deterministic duplicates (a documented closed-form function of another
             candidate already in the pool. These carry zero new information, by definition).
  Stage 2. Drop near-exact empirical duplicates (|Spearman rho| >= 0.95).
  Stage 3. Iterative VIF reduction on everything left (catches multivariate collinearity a
             pairwise check alone would miss).

A spatial-confound guard runs on top of stage 3's survivors: a variable that is high-VIF on the
raw data but comfortably low-VIF once each compartment's own mean is subtracted out is tracking
the same REGIONAL trend as its neighbours (e.g. elevation and temperature, via the lapse rate),
not literally duplicating a measurement. That's a real, defensible variable. It gets FLAGGED for
a human to look at, never auto-dropped, per the council review's second finding.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# Stage 1: every column here is a documented, closed-form mathematical function of another
# candidate already in the pool. Confirmed against xgb_environmental.FEATURE_PROVENANCE's own
# text (searched for "DERIVED from"), not guessed. These are not "highly correlated with"
# something else, they ARE something else, recomputed. Dropping them loses zero information.
DETERMINISTIC_DUPLICATES = {
    "gwa_wind_p95_10m": "DERIVED from gwa_weibull_a_10m and gwa_weibull_k_10m: A * [-ln(1-0.95)]^(1/k)",
    "gwa_prob_above_critical_10m": "DERIVED from gwa_weibull_a_10m and gwa_weibull_k_10m: exp[-(20/A)^k]",
    "gwa_wind_p95_50m": "DERIVED from gwa_weibull_a_50m and gwa_weibull_k_50m, same formula as the 10m version",
    "gwa_prob_above_critical_50m": "DERIVED from gwa_weibull_a_50m and gwa_weibull_k_50m, same formula as the 10m version",
    "inverse_slope_proxy": "DERIVED from slope_degrees (= -slope_degrees exactly, not new information)",
}


def drop_deterministic_duplicates(candidate_columns):
    """Stage 1. Returns (kept_columns, drop_log_rows)."""
    kept = [column for column in candidate_columns if column not in DETERMINISTIC_DUPLICATES]
    drop_log_rows = [
        {"column": column, "stage": "1_deterministic_duplicate", "reason": DETERMINISTIC_DUPLICATES[column]}
        for column in candidate_columns
        if column in DETERMINISTIC_DUPLICATES
    ]
    return kept, drop_log_rows


def compute_correlation_matrix(df, columns):
    """Spearman, not Pearson. Several candidates are ordinal or skewed (whcl, wind speed,
    curvature), and Spearman only assumes a monotonic relationship. Matches every other
    correlation check already done in this project (see variable_registry_audit.ipynb)."""
    return df[columns].corr(method="spearman")


def find_near_exact_duplicates(correlation_matrix, threshold=0.95):
    """Every pair of columns correlated at |rho| >= threshold. Each pair appears once."""
    pairs = []
    columns = list(correlation_matrix.columns)
    for i, column_a in enumerate(columns):
        for column_b in columns[i + 1:]:
            rho = correlation_matrix.loc[column_a, column_b]
            if pd.notna(rho) and abs(rho) >= threshold:
                pairs.append((column_a, column_b, rho))
    return pairs


def _provenance_is_external(column, feature_provenance):
    """True if this column's provenance text starts with "external" (an independently
    measured/downloaded value), False if it starts with "own calculation" (derived from other
    data already in this repo). Columns missing from feature_provenance default to False --
    treated as "derived" is the more cautious default when we don't actually know."""
    text = feature_provenance.get(column, "")
    return text.strip().startswith("external")


def choose_representative_loser(candidates, feature_provenance, df, target_column):
    """Given 2+ mutually near-duplicate/correlated columns, decide which ones to DROP, keeping
    exactly one. Tie-break order (rewritten after the 2026-08-06 llm-council review):

      1. Prefer to KEEP whichever is "external" (independently measured) over "own calculation"
         (derived). Drop derived columns first.
      2. If still tied, prefer to KEEP whichever has the higher |Spearman rho| with
         `target_column`, computed FRESH right here. This replaces the original tie-break rule
         ("prefer whichever already won in an earlier result"), which the council review found
         circular: it would have permanently rewarded whichever variable won some earlier,
         unaudited round, for reasons having nothing to do with real explanatory value.

    Returns a list of the columns to drop (everything except the single best-ranked survivor).
    """
    scored_candidates = []
    for column in candidates:
        is_external = _provenance_is_external(column, feature_provenance)
        rho_with_target, _ = spearmanr(df[column], df[target_column], nan_policy="omit")
        scored_candidates.append((column, is_external, abs(rho_with_target)))

    # Sort ascending by (is_external, |rho|). The single BEST candidate ends up last, since
    # True > False and a bigger |rho| sorts later. Easiest way to read off "keep the last one."
    scored_candidates.sort(key=lambda row: (row[1], row[2]))
    keeper = scored_candidates[-1][0]
    losers = [column for column, _, _ in scored_candidates if column != keeper]
    return losers


def compute_vif_table(df, columns):
    """Variance Inflation Factor for each column: VIF = 1 / (1 - R^2), where R^2 comes from
    regressing that one column on every OTHER column in the list. A VIF of 5 means 80% of that
    column's variance is already explained by the rest of the pool. The standard rule-of-thumb
    cutoff (Dormann et al. 2013, Ecography). Uses statsmodels, already a dependency elsewhere in
    this project (models/spatial_attribution/lisa.py, nlme.py)."""
    # statsmodels' variance_inflation_factor does NOT add an intercept for you. Without one,
    # each column's "R^2 from the others" is computed by a regression forced through the
    # origin, which inflates VIF hugely and unpredictably for columns that aren't already
    # centred near zero (elevation in the hundreds vs. a -1..1 index, say). add_constant() adds
    # that intercept column so VIF measures genuine collinearity, not an artefact of scale.
    #
    # .astype(float) is required, not cosmetic: some tables store a binary flag as bool (e.g.
    # time_since_thinning_missing/recent_thinning_5yr in the row-level model_table.parquet --
    # the same columns are float 0/1 fractions in the plot-level environmental export, which is
    # why this was never hit there). A DataFrame mixing bool and float64 columns can produce an
    # object-dtype array from .to_numpy(), which statsmodels' add_constant() cannot handle
    # ("ufunc 'isfinite' not supported for the input types"). Confirmed by hitting exactly this
    # running VIF on a row-level table for the first time.
    clean_df = df[columns].dropna()
    design_matrix = add_constant(clean_df.to_numpy().astype(float), has_constant="add")
    rows = []
    for i, column in enumerate(columns):
        # Index i+1, not i. Add_constant() puts the new constant column first (index 0).
        vif_value = variance_inflation_factor(design_matrix, i + 1)
        rows.append({"column": column, "vif": vif_value})
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def iterative_vif_reduction(df, columns, threshold=5.0, max_iterations=50):
    """Repeatedly compute VIF for the CURRENT column list. If the highest VIF exceeds
    `threshold`, drop that ONE column and recompute from scratch. Removing one column changes
    every other column's VIF, so dropping several at once from a single VIF table would be
    wrong. Stops once every remaining column is <= threshold (or after max_iterations, a safety
    net against an infinite loop from a numerically unstable design matrix)."""
    remaining_columns = list(columns)
    drop_log_rows = []
    for _ in range(max_iterations):
        if len(remaining_columns) <= 2:
            # VIF needs at least 2 other columns to regress against. Stop shrinking further.
            break
        vif_table = compute_vif_table(df, remaining_columns)
        worst_row = vif_table.iloc[0]
        if worst_row["vif"] <= threshold:
            break
        remaining_columns.remove(worst_row["column"])
        drop_log_rows.append({
            "column": worst_row["column"],
            "stage": "3_vif",
            "reason": f"VIF={worst_row['vif']:.2f} against the other {len(remaining_columns)} remaining candidates (threshold {threshold})",
        })
    return remaining_columns, drop_log_rows


def residualize_against_compartment(df, columns, compartment_column="cpmt"):
    """Subtract each compartment's own mean from every column (a simple fixed-effects
    transform). If two columns are correlated mainly because they both vary smoothly across the
    SAME region of the forest (e.g. elevation and temperature, via the lapse rate), that shared
    regional trend lives mostly BETWEEN compartments. Removing each compartment's own mean
    strips that shared trend out, leaving only within-compartment variation. A much better
    test of whether two variables are genuinely the same measurement twice."""
    compartment_means = df.groupby(df[compartment_column])[columns].transform("mean")
    return df[columns] - compartment_means


def spatial_confound_flags(df, columns, compartment_column="cpmt", threshold=5.0):
    """Compare each surviving column's VIF on the raw data vs. on the compartment-residualized
    data. A column that is high-VIF raw but comfortably low-VIF once the shared regional trend
    is removed is a spatial-confound case, not a genuine duplicate. This function only FLAGS
    it, it never removes anything. A human should read these rows before trusting stage 3's
    output at face value."""
    raw_vif = compute_vif_table(df, columns).set_index("column")["vif"]
    residualized_df = residualize_against_compartment(df, columns, compartment_column)
    residualized_vif = compute_vif_table(residualized_df, columns).set_index("column")["vif"]

    rows = []
    for column in columns:
        raw_value = raw_vif.get(column, np.nan)
        residualized_value = residualized_vif.get(column, np.nan)
        rows.append({
            "column": column,
            "raw_vif": raw_value,
            "compartment_residualized_vif": residualized_value,
            "likely_spatial_confound_not_true_duplicate": bool(raw_value > threshold and residualized_value <= threshold),
        })
    return pd.DataFrame(rows).sort_values("raw_vif", ascending=False).reset_index(drop=True)


def run_full_screen(
    df, candidate_columns, feature_provenance, target_column,
    compartment_column="cpmt", near_exact_threshold=0.95, vif_threshold=5.0,
):
    """The full corrected screen: stage 1 (deterministic duplicates), stage 2 (near-exact
    empirical duplicates), stage 3 (iterative VIF on the full remaining pool). Then a spatial-
    confound flag pass on the survivors, informational only.

    Returns (final_kept_columns, drop_log_df, spatial_confound_flags_df).
    """
    drop_log_rows = []

    after_stage1, stage1_drops = drop_deterministic_duplicates(candidate_columns)
    drop_log_rows.extend(stage1_drops)

    correlation_matrix = compute_correlation_matrix(df, after_stage1)
    near_exact_pairs = find_near_exact_duplicates(correlation_matrix, threshold=near_exact_threshold)
    after_stage2 = list(after_stage1)
    for column_a, column_b, rho in near_exact_pairs:
        # A column may already have been dropped by an earlier pair in this same loop (e.g. a
        # three-way near-exact cluster). Skip pairs where that's already happened.
        if column_a not in after_stage2 or column_b not in after_stage2:
            continue
        losers = choose_representative_loser([column_a, column_b], feature_provenance, df, target_column)
        for loser in losers:
            after_stage2.remove(loser)
            drop_log_rows.append({
                "column": loser, "stage": "2_near_exact_duplicate",
                "reason": f"rho={rho:.3f} with its kept partner",
            })

    after_stage3, stage3_drops = iterative_vif_reduction(df, after_stage2, threshold=vif_threshold)
    drop_log_rows.extend(stage3_drops)

    flags_df = spatial_confound_flags(df, after_stage3, compartment_column=compartment_column, threshold=vif_threshold)

    return after_stage3, pd.DataFrame(drop_log_rows), flags_df

# Purpose: the "NLME" step planned in progress_notes.md's spatial-question analysis plan --
# an interpretable "how much does terrain/wind explain y_max" number, sequenced deliberately
# BEFORE/ALONGSIDE Env-PINN's architecture decisions (not after), so the feature list handed to
# pinn_env_terrain's y_max(terrain, wind) sub-network is informed by an independent statistical
# method, not just XGBoost/SHAP.
#
# Real nonlinear mixed-effects fitting (refitting the Chapman-Richards curve itself, with
# y_max varying per plot via a random effect) needs iterative nonlinear optimisation tooling
# Python doesn't have a mature, well-tested equivalent of R's nlme/lme4 for -- so this uses the
# TWO-STAGE approximation already scoped in progress_notes.md ("adapting the two-stage
# template, not copying it exactly"):
#   Stage 1 (already done elsewhere): fit the global Chapman-Richards curve once, pooled across
#     all plots -- y_max/k/p are the single fixed constants already saved in
#     outputs/chapman_richards/<cohort>/params.json.
#   Stage 2 (this file): model the LEFTOVER residual (mean_cr_residual = observed height minus
#     that global curve's prediction) as a LINEAR mixed model: fixed effects = the candidate
#     terrain/wind covariates, random effect = a per-compartment intercept. This is a real,
#     standard, well-established statistical technique for exactly this situation (a residual
#     that plausibly clusters by group), not an ad-hoc substitute.
#
# Distributional note (2026-07-31): mean_cr_residual_4survey has moderate, not severe,
# non-normality (skew=-0.48, excess kurtosis=0.74, checked directly rather than assumed) --
# GAMLSS (which relaxes the normality assumption this model makes) was considered and
# deliberately not built, since Python's GAMLSS tooling is thin and the actual violation here is
# moderate. Point estimates (the fixed-effect coefficients) are fairly robust to this; standard
# errors/p-values are the part that can be distorted -- read them as approximate, and always
# cross-check a fixed effect's significance against the XGBoost refit-ablation result for the
# same variable (which makes no distributional assumption at all), not from this model alone.

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.mixed_linear_model import MixedLM


def fit_nlme(df, fixed_effect_columns, group_col="cpmt", target_col="mean_cr_residual", reml=True):
    # Standardises every fixed-effect column first (mean 0, std 1) -- same reason
    # elasticnet_environmental.py does this: makes each coefficient readable the same way
    # regardless of the variable's own natural units ("this many metres of residual, per one
    # standard-deviation change in this variable"), and metres/degrees/class-IDs become directly
    # comparable to each other.
    #
    # reml=True is the right default for THIS function's normal use (reporting a single,
    # already-decided model's coefficients/p-values via fixed_effects_table() -- REML is the
    # standard choice for final parameter estimation). variance_explained_by_fixed_effects()
    # below overrides this to reml=False when it needs to, for a different reason -- see there.
    model_df = df[[target_col, group_col] + fixed_effect_columns].dropna().copy()

    standardised_columns = {}
    for column in fixed_effect_columns:
        mean, std = model_df[column].mean(), model_df[column].std()
        standardised_columns[column] = (model_df[column] - mean) / std
    for column in fixed_effect_columns:
        model_df[column] = standardised_columns[column]

    formula = f"{target_col} ~ " + " + ".join(fixed_effect_columns)
    model = MixedLM.from_formula(formula, groups=group_col, data=model_df)
    result = model.fit(reml=reml)

    return result, model_df


def variance_explained_by_fixed_effects(df, fixed_effect_columns, group_col="cpmt", target_col="mean_cr_residual"):
    # Fits WITH the candidate fixed effects, and a NULL model with only a per-compartment
    # random intercept (no fixed effects at all except the overall mean) -- comparing the two
    # models' compartment-level random-effect variance answers "how much of the between-
    # compartment variation does terrain/wind explain", the actual "how much does terrain/wind
    # explain" number this whole step exists to produce. If the random-effect variance barely
    # shrinks once the fixed effects are added, they aren't explaining the compartment-level
    # pattern, whatever their individual p-values say.
    #
    # reml=False for BOTH models here -- caught in review (2026-08-01), not the original
    # implementation: REML variance-component estimates aren't cleanly comparable across models
    # that differ in their FIXED EFFECTS structure (the null model here has none, the full model
    # has 6) -- REML profiles out the fixed effects via a transform that depends on the fixed-
    # effects design matrix X itself, so changing X between the two models being compared
    # confounds "the fixed effects explained variance" with "the REML transform changed." This
    # is textbook guidance from Pinheiro & Bates (the reference behind R's own nlme package) --
    # use ML, not REML, whenever comparing variance components/likelihoods across models with
    # different fixed effects; reserve REML for a single, already-decided model's final
    # estimation (which is what fit_nlme()'s own default reml=True is for, in
    # fixed_effects_table()'s use case -- not this comparison).
    model_df = df[[target_col, group_col] + fixed_effect_columns].dropna().copy()

    null_model = MixedLM.from_formula(f"{target_col} ~ 1", groups=group_col, data=model_df)
    null_result = null_model.fit(reml=False)

    full_result, _ = fit_nlme(df, fixed_effect_columns, group_col=group_col, target_col=target_col, reml=False)

    null_group_variance = null_result.cov_re.iloc[0, 0]
    full_group_variance = full_result.cov_re.iloc[0, 0]
    proportion_explained = (null_group_variance - full_group_variance) / null_group_variance

    return {
        "null_group_variance": null_group_variance,
        "full_group_variance": full_group_variance,
        "proportion_of_compartment_variance_explained": proportion_explained,
        "null_residual_variance": null_result.scale,
        "full_residual_variance": full_result.scale,
    }


def check_residual_normality(result):
    # Promised diagnostic, not skipped: checks the FITTED model's own residuals (after
    # accounting for the fixed effects and the random compartment intercept), not just the raw
    # unconditional mean_cr_residual distribution this module's docstring already checked.
    # Terrain/wind explaining away some of the skew in the raw residual is a real possibility,
    # not assumed -- this is what actually tests it.
    residuals = result.resid
    return {
        "skewness": float(stats.skew(residuals)),
        "excess_kurtosis": float(stats.kurtosis(residuals)),
        "n": len(residuals),
    }


def fixed_effects_table(result):
    # One row per fixed effect (Intercept excluded -- it's not a candidate variable, just the
    # model's overall mean), with the standard "read with caution" reminder attached directly
    # to the data structure, not just left in a comment somewhere it can get separated from.
    summary = pd.DataFrame({
        "coefficient": result.fe_params,
        "std_error": result.bse_fe,
        "p_value": result.pvalues[result.fe_params.index],
    })
    summary = summary.drop(index="Intercept", errors="ignore")
    return summary.reindex(summary["coefficient"].abs().sort_values(ascending=False).index)

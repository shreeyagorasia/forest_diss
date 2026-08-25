# Which RQ this closes a gap in, and why: RQ2b item 3 (why NLME and Elastic Net disagree on
# `topex`'s coefficient. Strong and stable in NLME, weak and sign-flipping in EN. While
# `slope_degrees` is stable in both). The results draft flagged this as an untested candidate
# mechanism: NLME has an explicit compartment-level random intercept that absorbs compartment-
# level correlation; Elastic Net has no such term. If a predictor's real variation sits mostly
# BETWEEN compartments rather than within them, a plain regression with no random-effect term
# has nothing to separate "this predictor's real effect" from "whatever unmeasured
# compartment-level factors happen to correlate with it in this particular fold's training
# data". A textbook spatial-confounding setup. This only applies if the precondition holds:
# does `topex` actually vary mostly between compartments? That precondition is checked here,
# directly, on already-loaded data. No model fitting at all.
#
# Method: a one-way random-effects ANOVA intraclass correlation (ICC). The standard
# between/within variance decomposition. ICC close to 1 means a variable is almost constant
# within a compartment and varies mostly across compartments; ICC close to 0 means the opposite
# (mostly plot-to-plot variation within the same compartment).
#
# Run as: python -m models.spatial_attribution.rq2b_topex_variance_decomposition

from models.xgb_environmental.data import load_plots_for_cohort


def compute_icc(plots_df, column, group_col="cpmt"):
    grand_mean = plots_df[column].mean()
    group_stats = plots_df.groupby(group_col)[column].agg(["mean", "count"])
    n_groups = len(group_stats)
    n_total = len(plots_df)

    ss_between = ((group_stats["mean"] - grand_mean) ** 2 * group_stats["count"]).sum()
    ss_within = sum(((group - group.mean()) ** 2).sum() for _, group in plots_df.groupby(group_col)[column])

    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n_total - n_groups)

    # Standard one-way random-effects ANOVA formula for the "average group size" term, correct
    # even when compartments have unequal numbers of plots (they do here, from single plots to
    # over a thousand, per the methodology chapter's own note on why NLME uses random intercepts
    # only, not random slopes).
    group_sizes = group_stats["count"].to_numpy()
    n_bar = (n_total - (group_sizes ** 2).sum() / n_total) / (n_groups - 1)

    var_between = max((ms_between - ms_within) / n_bar, 0.0)
    var_within = ms_within
    return var_between / (var_between + var_within)


def main():
    df = load_plots_for_cohort("4survey")
    plots = df[["identification", "cpmt", "topex", "slope_degrees"]].drop_duplicates("identification")
    print(f"n plots: {len(plots):,}, n compartments: {plots['cpmt'].nunique():,}")

    for column in ["topex", "slope_degrees"]:
        icc = compute_icc(plots, column)
        print(f"{column}: ICC (between-compartment share of total variance) = {icc:.4f}")


if __name__ == "__main__":
    main()

# Purpose: two cheap checks for whether the temporal-stability gap (early-years-fit curve
# increasingly mispredicting a plot's own later height, see temporal_stability_check.py) has a
# real, findable structure -- a genuine mid-trajectory disturbance signature, or a recorded
# thinning event -- rather than staying an unexplained "maybe environment, maybe something else"
# question. Both use data already on disk, no external source needed.

import numpy as np
import pandas as pd

from models.common.splits import TEMPORAL_YEARS
from models.growth_curve_attribution.data import load_filtered_growth_curve_table
from models.growth_curve_attribution.temporal_stability_check import HEIGHT_COLUMN, compute_shape_term, fit_y_max_per_plot


def fit_all_years_and_compute_residuals(cohort):
    # Same fit_y_max_per_plot() as the temporal stability check, but here it sees EVERY year a
    # plot has (not just the early ones) -- this is Candidate A's own full fit, and the residual
    # left over at each individual survey year is what Check 1 looks at.
    df = load_filtered_growth_curve_table(cohort)
    df = df.dropna(subset=["y_max_yldc"]).copy()  # only plots with a valid p1-p5/yldc combination

    y_max_fit_table = fit_y_max_per_plot(df)
    df = df.merge(y_max_fit_table, on="identification", how="inner")

    df["predicted_height"] = df["y_max_fit"] * compute_shape_term(df)
    df["residual"] = df[HEIGHT_COLUMN] - df["predicted_height"]
    return df


def residual_by_survey_year(df_with_residuals):
    # Population-level view: is a PARTICULAR calendar year systematically off across every
    # plot, not just one -- a signature of something broad happening in that window (storm,
    # drought), rather than plot-by-plot idiosyncrasy.
    return df_with_residuals.groupby("LiDAR_year")["residual"].agg(["count", "mean", "std"])


def per_plot_residual_range(df_with_residuals):
    # Per-plot view: how much does each plot's own residual swing between its best-fit and
    # worst-fit survey year. A plot with an unusually large swing is a candidate for a real
    # mid-trajectory disturbance-and-recovery, not just noise scattered around one stable curve.
    grouped = df_with_residuals.groupby("identification")["residual"]
    residual_range = (grouped.max() - grouped.min()).rename("residual_range")
    return residual_range.reset_index()


def derive_structural_change_intervals(df):
    """Derive consecutive-survey height, canopy and standing-volume changes.

    Fractions use the earlier survey as the denominator. Very small earlier
    values make fractions unstable, so raw changes and denominator-validity
    flags are retained. Positive ``*_drop_frac`` means a loss.
    """
    required = ["identification", "LiDAR_year", HEIGHT_COLUMN, "CanopyCover", "Vol95"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Structural-change check needs columns: {missing}")

    rows = df[required].sort_values(["identification", "LiDAR_year"]).copy()
    grouped = rows.groupby("identification", sort=False)
    for column in ["LiDAR_year", HEIGHT_COLUMN, "CanopyCover", "Vol95"]:
        rows[f"previous_{column}"] = grouped[column].shift(1)

    rows["survey_interval_years"] = rows["LiDAR_year"] - rows["previous_LiDAR_year"]
    for source, short_name in [
        (HEIGHT_COLUMN, "height"), ("CanopyCover", "canopy"), ("Vol95", "volume")
    ]:
        previous = rows[f"previous_{source}"]
        rows[f"{short_name}_change"] = rows[source] - previous
        rows[f"{short_name}_drop_frac"] = np.where(
            previous > 1e-6, (previous - rows[source]) / previous, np.nan
        )
        rows[f"{short_name}_denominator_valid"] = previous > 1e-6
    return rows.loc[rows["previous_LiDAR_year"].notna()].reset_index(drop=True)


def classify_structural_change_intervals(
    intervals,
    height_drop_threshold=0.25,
    joint_collapse_threshold=0.70,
    stable_structure_threshold=0.10,
):
    """Add transparent diagnostic labels; do not infer disturbance cause.

    ``clearfell_like`` means height, canopy and volume all collapse together.
    ``measurement_inconsistent`` means height collapses while canopy changes
    little and volume does not fall. The remaining height collapses are
    ambiguous candidates, including possible partial windthrow.
    """
    out = intervals.copy()
    height_collapse = out["height_drop_frac"] >= height_drop_threshold
    joint = (
        height_collapse
        & (out["canopy_drop_frac"] >= joint_collapse_threshold)
        & (out["volume_drop_frac"] >= joint_collapse_threshold)
    )
    inconsistent = (
        height_collapse
        & (out["canopy_drop_frac"] <= stable_structure_threshold)
        & (out["volume_drop_frac"] <= 0)
    )
    out["structural_change_class"] = np.select(
        [joint, inconsistent, height_collapse],
        ["clearfell_like", "measurement_inconsistent", "ambiguous_disturbance"],
        default="no_height_collapse",
    )
    out["height_collapse_flag"] = height_collapse
    out["joint_structure_collapse_flag"] = joint
    out["measurement_inconsistent_flag"] = inconsistent
    out["ambiguous_disturbance_flag"] = height_collapse & ~joint & ~inconsistent
    return out


def summarize_plot_disturbance_status(cohort):
    """One row per plot: whether ANY of its consecutive-survey intervals looks like a clearfell
    or a measurement-inconsistent collapse (both make that plot's own y_max fit untrustworthy --
    mixing pre- and post-event rows into one curve), plus the ambiguous-disturbance flag and raw
    drop fractions kept as metadata for plots that stay in the modelling population.

    A plot can have more than one flagged interval (up to 3-5 consecutive gaps per plot) -- ANY
    single clearfell-like or measurement-inconsistent interval is enough to exclude that plot,
    since the curve-fit uses every one of a plot's rows together, not interval-by-interval.
    """
    df = load_filtered_growth_curve_table(cohort)
    intervals = derive_structural_change_intervals(df)
    classified = classify_structural_change_intervals(intervals)

    per_plot = classified.groupby("identification").agg(
        any_clearfell_like=("joint_structure_collapse_flag", "any"),
        any_measurement_inconsistent=("measurement_inconsistent_flag", "any"),
        any_ambiguous_disturbance=("ambiguous_disturbance_flag", "any"),
        max_height_drop_frac=("height_drop_frac", "max"),
        max_canopy_drop_frac=("canopy_drop_frac", "max"),
        max_volume_drop_frac=("volume_drop_frac", "max"),
    ).reset_index()

    # Excluded from the curve-FITTING population: a felled/replanted stand (clearfell_like) or a
    # survey that doesn't agree with itself (measurement_inconsistent) has no real trajectory to
    # fit a fixed-shape curve to. ambiguous_disturbance plots are NOT excluded -- kept in the
    # modelling population with a flag/feature, per the explicit decision that possible genuine
    # disturbance (e.g. partial windthrow) is part of what this analysis aims to explain, not
    # noise to remove.
    per_plot["exclude_from_curve_fit"] = per_plot["any_clearfell_like"] | per_plot["any_measurement_inconsistent"]
    return per_plot


def thinning_during_gap_flag(df, gap_start_year, gap_end_year):
    # True for a plot whose recorded last_thinn falls inside (gap_start_year, gap_end_year] --
    # i.e. that plot's most recent known thinning happened during the unmeasured gap.
    # last_thinn is CONSTANT across every one of a plot's own rows (checked directly -- 0 of
    # 71,766 4survey plots show any variation), so this is a single per-plot fact, not something
    # that needs comparing between two specific survey rows.
    one_row_per_plot = df.drop_duplicates("identification")[["identification", "last_thinn"]].copy()

    thinned_during_gap = (
        one_row_per_plot["last_thinn"].gt(gap_start_year) & one_row_per_plot["last_thinn"].le(gap_end_year)
    )
    one_row_per_plot["thinned_during_gap"] = thinned_during_gap
    return one_row_per_plot[["identification", "thinned_during_gap"]]


def thinning_during_gap_bias_check(cohort):
    # Tests whether the bias found in the temporal stability check (early-years-fit curve
    # increasingly over/under-predicting a plot's own later height) lines up with a RECORDED
    # thinning event landing inside the unmeasured gap, rather than staying an unexplained
    # "maybe environment, maybe disturbance" result.
    df = load_filtered_growth_curve_table(cohort)
    df = df.dropna(subset=["y_max_yldc"]).copy()

    fitting_years = TEMPORAL_YEARS[cohort]["train_years"]
    evaluation_years = TEMPORAL_YEARS[cohort]["val_years"] + TEMPORAL_YEARS[cohort]["test_years"]
    gap_start_year = max(fitting_years)
    gap_end_year = max(evaluation_years)

    fitting_rows = df[df["LiDAR_year"].isin(fitting_years)]
    y_max_fit_table = fit_y_max_per_plot(fitting_rows)

    thinning_flag = thinning_during_gap_flag(df, gap_start_year=gap_start_year, gap_end_year=gap_end_year)

    results = {}
    for evaluation_year in evaluation_years:
        evaluation_rows = df[df["LiDAR_year"] == evaluation_year].merge(y_max_fit_table, on="identification", how="inner")
        evaluation_rows = evaluation_rows.merge(thinning_flag, on="identification", how="left")

        evaluation_rows = evaluation_rows.copy()
        evaluation_rows["predicted_height"] = evaluation_rows["y_max_fit"] * compute_shape_term(evaluation_rows)
        evaluation_rows["residual"] = evaluation_rows[HEIGHT_COLUMN] - evaluation_rows["predicted_height"]

        results[evaluation_year] = evaluation_rows.groupby("thinned_during_gap")["residual"].agg(["count", "mean", "std"])

    return results

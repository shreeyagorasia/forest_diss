# Purpose: refined disturbance-flag diagnostics proposed 2026-08-20, evaluation-only.
#
# IMPORTANT -- deliberately NOT wired into the curve-fitting pipeline. The existing
# disturbance_checks.py::summarize_plot_disturbance_status()'s exclude_from_curve_fit column
# feeds directly into y_max_fit, which is the target both Q1 (mean_cr_residual) and Q2
# (local_y_max_difference) are built from. Changing that exclusion list would shift those
# targets and require every downstream model (Q1's EN/XGBoost/LMM, Q2's EN/XGBoost/GNNWR, both
# cohorts, every feature set) to be refit to keep the dissertation's own reported numbers
# consistent. GNNWR alone is cluster-bound, hours-scale (see the reference-density ablation run
# earlier this project). Decided 2026-08-20, no retraining available before submission: this
# file adds the refined diagnostics as NEW, separate functions only. Nothing in
# disturbance_checks.py is modified, and exclude_from_curve_fit's actual behaviour is untouched.
# Only 22 plots total (18/58,112 4survey + 4/13,769 6survey) would even be affected by the one
# rule change that touches exclusion (see below) -- small, but "small" still means "a rerun is
# needed to report it correctly," not "safe to claim without one."
#
# LIMITATION, worth stating explicitly in the dissertation: the chapter's reported
# clearfell_like/measurement_inconsistent exclusion uses the ORIGINAL rules below, not the
# refined ones. The refined logic is implemented, tested, and its effect size on plot
# COUNT is known (this file); its effect on downstream R2/coefficient numbers is not, since that
# would require the rerun this decision explicitly avoids given time constraints.

import numpy as np
import pandas as pd

from models.growth_curve_attribution.data import load_filtered_growth_curve_table
from models.growth_curve_attribution.disturbance_checks import (
    derive_structural_change_intervals,
    classify_structural_change_intervals,
)

# 90th percentile of deficit magnitude among growth-deficit plots (4survey, computed 2026-08-20)
# -- chosen to be comparably selective to the existing ambiguous_disturbance flag (~1.8% of the
# population) without being as narrow as clearfell_like (~0.4%). Data-driven, not arbitrary.
# Deliberately calibrated on 4survey only and applied unchanged to 6survey too, matching this
# project's own convention of treating 4survey as the primary cohort throughout (6survey is a
# nested sensitivity check, not an equal-weight second cohort) -- NOT re-calibrated per cohort.
# Known consequence, not a bug: at this same absolute threshold, 6survey flags a higher share of
# its population (8.70%, vs. 4survey's 3.57%), because 6survey's longer survey history (adds
# 2002/2006) accumulates more growth-span noise for the same cutoff. Reported as-is.
CHRONIC_DEFICIT_THRESHOLD_M = 6.17

# Bottom-quartile topex, population-wide (4survey, computed 2026-08-20) -- "high exposure" for
# the purposes of the exposure-aware reclassification below. Same threshold already used and
# reported in TEMP_rq3_flagged_plot_mechanism_split_2026-08-20.tex.
HIGH_EXPOSURE_TOPEX_THRESHOLD = -10.07


def exposure_aware_measurement_inconsistent(cohort, topex_by_plot):
    """Refined Rule 2: does NOT change which plots are excluded from curve fitting.

    Re-derives the same clearfell/measurement-inconsistent/ambiguous classification
    disturbance_checks.py already computes (identical thresholds -- height_drop_threshold=0.25,
    joint_collapse_threshold=0.70, stable_structure_threshold=0.10, unchanged, per the decision
    to leave Rule 1's numbers alone). The only new step: for every interval currently classified
    "measurement_inconsistent", check whether that plot sits in high-exposure terrain. If so,
    relabel it "possible_wind_damage" instead -- a real ecological event (top breakage) produces
    the exact same signature (height drops, canopy/volume stable) as a genuine measurement
    artefact, and exposure makes the ecological reading at least as plausible as the artefact
    one. Returns a per-INTERVAL table (a plot can have more than one flagged interval); does not
    touch or call anything in disturbance_checks.py's own exclude_from_curve_fit logic.
    """
    df = load_filtered_growth_curve_table(cohort)
    intervals = derive_structural_change_intervals(df)
    classified = classify_structural_change_intervals(intervals)

    classified = classified.merge(topex_by_plot, on="identification", how="left")
    is_measurement_inconsistent = classified["structural_change_class"] == "measurement_inconsistent"
    is_high_exposure = classified["topex"] <= HIGH_EXPOSURE_TOPEX_THRESHOLD

    classified["refined_class"] = classified["structural_change_class"]
    reclassify = is_measurement_inconsistent & is_high_exposure
    classified.loc[reclassify, "refined_class"] = "possible_wind_damage"

    n_reclassified = reclassify.sum()
    n_original = is_measurement_inconsistent.sum()
    print(f"{cohort}: {n_reclassified} / {n_original} measurement_inconsistent intervals "
          f"reclassified as possible_wind_damage (high exposure)")
    return classified


def chronic_underperformance_candidate(cohort):
    """New Rule 4: a plot whose cumulative growth (first survey to last) falls short of its
    yield-class benchmark's own predicted growth over the same span by at least
    CHRONIC_DEFICIT_THRESHOLD_M. No abrupt within-interval drop required -- this is exactly the
    failure mode none of disturbance_checks.py's three existing categories can see, since all
    three require a single-interval height collapse. Not excluded from anything; a candidate
    label only, same philosophy as ambiguous_disturbance (kept, not treated as noise).
    """
    table = load_filtered_growth_curve_table(cohort)
    records = []
    for plot_id, group in table.sort_values("Age").groupby("identification"):
        if len(group) < 2:
            continue
        first, last = group.iloc[0], group.iloc[-1]
        observed_growth = last["elev_percentile_95th"] - first["elev_percentile_95th"]
        benchmark_growth = last["top_height95_yldc_predicted"] - first["top_height95_yldc_predicted"]
        deficit = benchmark_growth - observed_growth
        records.append({
            "identification": plot_id,
            "deficit_m": deficit,
            "chronic_underperformance_candidate": deficit >= CHRONIC_DEFICIT_THRESHOLD_M,
        })
    result = pd.DataFrame(records)
    n_flagged = result["chronic_underperformance_candidate"].sum()
    print(f"{cohort}: {n_flagged} / {len(result):,} plots flagged as chronic_underperformance_candidate "
          f"({100*n_flagged/len(result):.2f}%)")
    return result

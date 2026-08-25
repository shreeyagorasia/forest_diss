# Purpose: RQ3's outlier diagnosis (rq3_outlier_diagnosis.py) found that the worst-residual plots
# can't be explained by any environmental variable, even with a large SHAP push. The working
# theory was "these are disturbance/data-quality artifacts, not unexplained environmental signal".
# This script actually checks that theory, instead of leaving it as an untested guess: it pulls
# every "worst residual" identification found across all 6 (set, cohort) outlier-diagnosis runs,
# and joins them against the SAME disturbance flags the notebook
# (av2_local_growth_curve_grouped_importance.ipynb) uses for its own compartment-pattern map --
# but calls the standalone functions directly (disturbance_checks.py), so nothing about the
# notebook itself needs to be opened, edited, or re-run.
#
# Run as: python -m models.growth_curve_attribution.cross_reference_outliers_with_disturbance

from pathlib import Path

import pandas as pd

from models.common.saving import model_output_dir
from models.growth_curve_attribution.disturbance_checks import (
    fit_all_years_and_compute_residuals,
    per_plot_residual_range,
    summarize_plot_disturbance_status,
)

MODEL_NAME = "rq3_en_xgb"
SET_NAMES = ["nested_set2_top10", "nested_set3_gated_terrain_wind_vif", "nested_set4_gated_all_vif"]
COHORTS = ["4survey", "6survey"]


def collect_outlier_identifications():
    # Reads every already-saved xgboost_outlier_diagnosis.csv (6 of them, one per set x cohort)
    # and records, per identification, which (set, cohort) combos flagged it as a top-N worst
    # residual. A plot appearing in MULTIPLE combos is a stronger signal than one appearing once.
    rows = []
    for set_name in SET_NAMES:
        for cohort in COHORTS:
            output_dir = model_output_dir(f"{MODEL_NAME}_{set_name}", cohort, split_type="spatial_block_kfold")
            csv_path = output_dir / "xgboost_outlier_diagnosis.csv"
            if not csv_path.exists():
                print(f"  Skipping {set_name}/{cohort}: {csv_path} not found")
                continue
            outliers = pd.read_csv(csv_path)
            for _, row in outliers.iterrows():
                rows.append({
                    "identification": row["identification"],
                    "cpmt": row["cpmt"],
                    "cohort": cohort,
                    "set_name": set_name,
                    "residual": row["residual"],
                })
    return pd.DataFrame(rows)


def main():
    outlier_table = collect_outlier_identifications()

    # Count how many of the 6 (set, cohort) combos flagged each plot. 6 is the max possible
    # for a 4survey OR 6survey plot alone (a plot only exists in one cohort, so really the max
    # per plot is 3, one per set, within whichever cohort it belongs to).
    appearance_counts = (
        outlier_table.groupby("identification")
        .agg(n_appearances=("set_name", "nunique"), cohort=("cohort", "first"), cpmt=("cpmt", "first"))
        .reset_index()
        .sort_values("n_appearances", ascending=False)
    )
    print("===== Outlier plots, how many of the 3 sets (within their own cohort) flag them =====\n")
    print(appearance_counts.to_string(index=False))

    # Pull disturbance flags separately per cohort (the underlying growth-curve table itself is
    # cohort-specific. 4survey and 6survey are different row selections of the same source data).
    disturbance_tables = []
    residual_range_tables = []
    for cohort in COHORTS:
        disturbance_tables.append(summarize_plot_disturbance_status(cohort))
        curve_rows = fit_all_years_and_compute_residuals(cohort)
        residual_range_tables.append(per_plot_residual_range(curve_rows))
    disturbance_status = pd.concat(disturbance_tables, ignore_index=True).drop_duplicates("identification")
    residual_range = pd.concat(residual_range_tables, ignore_index=True).drop_duplicates("identification")

    merged = (
        appearance_counts
        .merge(disturbance_status, on="identification", how="left")
        .merge(residual_range, on="identification", how="left")
    )

    print("\n===== Cross-referenced against disturbance_checks.py flags =====\n")
    display_columns = [
        "identification", "cpmt", "cohort", "n_appearances", "residual_range",
        "any_clearfell_like", "any_measurement_inconsistent", "any_ambiguous_disturbance",
        "exclude_from_curve_fit",
    ]
    print(merged[display_columns].to_string(index=False))

    n_flagged_any = merged[
        merged["any_clearfell_like"] | merged["any_measurement_inconsistent"] | merged["any_ambiguous_disturbance"]
    ].shape[0]
    print(f"\n{n_flagged_any} of {len(merged)} unique outlier plots carry AT LEAST ONE disturbance/measurement flag.")

    output_path = Path("TEMP_results") / "rq3_outlier_disturbance_crossref.csv"
    merged.to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    main()

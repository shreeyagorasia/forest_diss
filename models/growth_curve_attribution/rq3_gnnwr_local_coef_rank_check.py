# Purpose: RQ3 item 1 found that CanopyCover dominates EN/XGBoost attribution on 4survey but is
# outranked on 6survey (by cpmt_compactness_ratio or windward_topex, SHAP-only comparison. EN/
# XGBoost coefficient tables were never computed for 6survey at all, too noisy per that item's own
# source file). GNNWR was flagged as "not covered by this item at all" even though its own per-plot
# local coefficients (coef_* columns) already exist in every saved test_predictions.csv. This
# script closes that gap: does GNNWR's own CanopyCover coefficient also lose its #1 rank on 6survey?
#
# No new fitting. Reads the already-saved test_predictions.csv from every GNNWR fold (Set2/3/4,
# both cohorts, seed 42, the same fits already cited elsewhere in RQ3), pools all 5 folds, and
# ranks each set's coef_* columns by mean absolute value. The standard way to collapse a
# per-plot local coefficient into a single importance-like number, directly analogous to EN's
# |coefficient| or XGBoost's mean |SHAP| already used for this same comparison elsewhere in RQ3.
#
# Run as: python -m models.growth_curve_attribution.rq3_gnnwr_local_coef_rank_check

from pathlib import Path

import pandas as pd

GNNWR_DIR = Path("outputs/growth_curve_attribution/gnnwr")
SETS = ["nested_set2_top10", "nested_set3_gated_terrain_wind_vif", "nested_set4_gated_all_vif"]
COHORTS = ["4survey", "6survey"]
N_FOLDS = 5


def load_pooled_predictions(set_name, cohort):
    frames = []
    for fold in range(N_FOLDS):
        path = GNNWR_DIR / f"gnnwr_{set_name}_{cohort}_reffull_fold{fold}of5_test_predictions.csv"
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def rank_coefficients(pooled_df):
    coef_columns = [c for c in pooled_df.columns if c.startswith("coef_")]
    mean_abs_coef = {col.replace("coef_", ""): pooled_df[col].abs().mean() for col in coef_columns}
    ranked = sorted(mean_abs_coef.items(), key=lambda item: item[1], reverse=True)
    return ranked


def main():
    for set_name in SETS:
        for cohort in COHORTS:
            pooled_df = load_pooled_predictions(set_name, cohort)
            ranked = rank_coefficients(pooled_df)
            top5 = ", ".join(f"{name} ({value:.3f})" for name, value in ranked[:5])
            canopy_rank = next(i for i, (name, _) in enumerate(ranked, start=1) if name == "CanopyCover")
            print(f"{set_name} | {cohort} | n={len(pooled_df):,} | CanopyCover rank={canopy_rank}/{len(ranked)}")
            print(f"  Top 5 by mean |local coefficient|: {top5}")
        print()


if __name__ == "__main__":
    main()

# Run as: python -m models.growth_curve_attribution.run_importance_stability_check

from models.growth_curve_attribution.explain_signal import FINAL_FEATURE_COLUMNS, build_full_table
from models.growth_curve_attribution.importance_stability_check import per_fold_gain_importance, summarize_rank_stability

COHORT = "4survey"


def main():
    table, feature_columns = build_full_table(COHORT)
    per_fold_gain = per_fold_gain_importance(table, feature_columns)
    summary = summarize_rank_stability(per_fold_gain)
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()

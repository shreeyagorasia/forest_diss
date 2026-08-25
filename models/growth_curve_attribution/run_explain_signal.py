# Run as: python -m models.growth_curve_attribution.run_explain_signal
#
# Cheap version of "explain the confirmed signal": one XGBoost fit on the full cleaned 4survey
# population (no CV sweep), Spearman correlation with the target as a near-free first pass, SHAP
# mean |value| as the model-based follow-up. See explain_signal.py's own header for why this
# feature list needs no further justification runs. Every choice is already backed by a check
# on disk.

from models.growth_curve_attribution.explain_signal import FINAL_FEATURE_COLUMNS, build_full_table, fit_and_explain, spearman_with_target

# 6survey's plot-level signal was confirmed null under 5-fold spatial CV. No point explaining a
# signal that doesn't exist there, so only 4survey is run here.
COHORT = "4survey"


def main():
    table, feature_columns = build_full_table(COHORT)
    print(f"{COHORT}: {len(table):,} plots, {len(feature_columns)} features")
    print(f"Feature list: {feature_columns}")

    print("\n===== Spearman correlation with local_y_max_difference (near-free first pass) =====")
    print(spearman_with_target(table, feature_columns).to_string(index=False))

    print("\n===== SHAP mean |value| (one XGBoost fit on the full cleaned population) =====")
    _, importance = fit_and_explain(table, feature_columns)
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()

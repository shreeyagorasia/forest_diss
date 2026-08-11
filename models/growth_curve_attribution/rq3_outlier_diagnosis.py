# Run as: python -m models.growth_curve_attribution.rq3_outlier_diagnosis --cohort 4survey --set-name nested_set4_gated_all_vif --top-n 10
#
# The actual point of building per-plot SHAP values (compute_rq3_shap.py) -- pulls the N largest-
# residual plots (worst XGBoost predictions for local_y_max_difference) and shows, for each one,
# WHY the model predicted what it did: every feature's SHAP contribution for that specific row,
# sorted by |SHAP|, not just "CanopyCover matters on average" (which the gain-importance/
# aggregate-SHAP tables already answer). This is genuinely different information -- a global
# importance ranking can't tell you why one specific plot's prediction is 20m off.
#
# Reads already-saved files only (predictions.csv + xgboost_shap_values.csv) -- no fitting, no
# refitting, purely local and fast.

import argparse

import pandas as pd

from models.common.saving import model_output_dir

MODEL_NAME = "rq3_en_xgb"


def diagnose_outliers(cohort, set_name, top_n=10):
    output_dir = model_output_dir(f"{MODEL_NAME}_{set_name}", cohort, split_type="spatial_block_kfold")
    predictions = pd.read_csv(output_dir / "predictions.csv")
    shap_values = pd.read_csv(output_dir / "xgboost_shap_values.csv")

    predictions["xgboost_residual"] = predictions["local_y_max_difference"] - predictions["xgboost_predicted"]
    predictions["abs_residual"] = predictions["xgboost_residual"].abs()

    feature_columns = [c for c in shap_values.columns if c not in ("identification", "fold")]
    shap_values = shap_values.set_index(["identification", "fold"])

    worst = predictions.sort_values("abs_residual", ascending=False).head(top_n)

    print(f"===== RQ3 outlier diagnosis: {cohort} / {set_name}, top {top_n} worst XGBoost residuals =====\n")
    rows_for_csv = []
    for _, plot in worst.iterrows():
        key = (plot["identification"], plot["fold"])
        if key not in shap_values.index:
            print(f"  identification={plot['identification']} (fold {plot['fold']}): no SHAP row found, skipping")
            continue
        plot_shap = shap_values.loc[key]
        direction = "OVER-predicted" if plot["xgboost_residual"] < 0 else "UNDER-predicted"
        print(
            f"  identification={plot['identification']}  cpmt={plot['cpmt']}  x={plot['x']:.0f}  y={plot['y']:.0f}  "
            f"observed={plot['local_y_max_difference']:.2f}  predicted={plot['xgboost_predicted']:.2f}  "
            f"residual={plot['xgboost_residual']:+.2f}  ({direction})"
        )
        top_shap = plot_shap[feature_columns].sort_values(key=abs, ascending=False).head(5)
        for feature, value in top_shap.items():
            print(f"      {feature:35s} SHAP={value:+.3f}")
        print()

        rows_for_csv.append({
            "identification": plot["identification"], "cpmt": plot["cpmt"], "x": plot["x"], "y": plot["y"],
            "observed": plot["local_y_max_difference"], "predicted": plot["xgboost_predicted"],
            "residual": plot["xgboost_residual"], "direction": direction,
            **{f"shap_{feature}": plot_shap[feature] for feature in feature_columns},
        })

    output_path = output_dir / "xgboost_outlier_diagnosis.csv"
    pd.DataFrame(rows_for_csv).to_csv(output_path, index=False)
    print(f"Saved -> {output_path}")
    return pd.DataFrame(rows_for_csv)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="4survey", choices=["4survey", "6survey"])
    parser.add_argument(
        "--set-name", default="nested_set4_gated_all_vif",
        choices=["nested_set2_top10", "nested_set3_gated_terrain_wind_vif", "nested_set4_gated_all_vif", "nested_set5_all_ungated_vif"],
    )
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    diagnose_outliers(args.cohort, args.set_name, top_n=args.top_n)


if __name__ == "__main__":
    main()

# Run as: python -m models.baselines.rq1_xgb_interaction_check
#
# Tests the "XGBoost wins because it captures feature interactions" guess directly, for RQ1's
# 4survey XGBoost-vs-DNN comparison. A companion check (rq1_compartment_subsample_check.py)
# already showed Linear -> RF -> DNN -> XGBoost's R2 ladder means the DNN already captures 80% of
# the total nonlinear/interaction gain over a plain Linear model, and that Random Forest (also a
# nonlinear, interaction-capturing tree ensemble) does WORSE than the DNN -- both facts weaken the
# "interactions specifically" story before this check even runs. This check asks the more direct
# question: inside XGBoost's own fitted model, how much of its prediction actually comes from
# interaction effects, versus single-feature (main) effects?
#
# Method: SHAP interaction values (shap.TreeExplainer.shap_interaction_values), the standard
# decomposition of a tree ensemble's SHAP value per row into a main-effect term per feature plus
# a pairwise interaction term per feature pair (Lundberg et al. 2018). Computed on a random
# sample of the held-out test set (interaction values are O(n * features^2), too slow to run on
# the full test set here).

import numpy as np
import shap
import xgboost as xgb

from models.common.splits import SPLIT_SEED
from models.common.torch_data import ENV_TERRAIN_FEATURE_SETS, TARGET_COLUMN, load_split_table_with_terrain
from models.xgb_baseline.xgb_baseline import FEATURE_COLUMNS, prepare_features

COHORT = "4survey"
FEATURE_SET_NAME = "nested_set3_gated_terrain_wind_vif"
# Winning 4survey config, same as rq1_compartment_subsample_check.py and
# TEMP_rq1_xgb_hyperparameter_search_2026-08-16.tex.
XGB_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.02, random_state=42, n_jobs=1)
INTERACTION_SAMPLE_SIZE = 2000
INTERACTION_SAMPLE_SEED = 42


def main():
    feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]
    full_df = load_split_table_with_terrain(COHORT, "spatial_block", feature_columns, split_seed=SPLIT_SEED)
    train_df = full_df[full_df["split"] == "train"]
    test_df = full_df[full_df["split"] == "test"]

    full_columns = FEATURE_COLUMNS + list(feature_columns)
    features_train = prepare_features(train_df, feature_columns=full_columns)
    features_test = prepare_features(test_df, feature_columns=full_columns, encoded_column_names=features_train.columns)

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(features_train, train_df[TARGET_COLUMN])

    r2 = model.score(features_test, test_df[TARGET_COLUMN])
    print(f"Sanity check -- single-split test R2: {r2:.4f} (Table 1's pooled 5-fold value is 0.674, so this should be in the same ballpark)")

    # Subsample the test set -- interaction values are expensive (O(n * n_features^2)).
    sample_df = features_test.sample(n=min(INTERACTION_SAMPLE_SIZE, len(features_test)), random_state=INTERACTION_SAMPLE_SEED)
    print(f"Computing SHAP interaction values on {len(sample_df)} test rows, {sample_df.shape[1]} features...")

    explainer = shap.TreeExplainer(model)
    interaction_values = explainer.shap_interaction_values(sample_df)  # shape: (n_rows, n_features, n_features)

    n_features = sample_df.shape[1]
    feature_names = sample_df.columns.tolist()

    # Main effect per feature = the diagonal of the interaction matrix (a feature "interacting
    # with itself" is its own main effect, by this decomposition's own definition).
    main_effect_abs = np.abs(np.diagonal(interaction_values, axis1=1, axis2=2))  # (n_rows, n_features)
    mean_main_effect_per_feature = main_effect_abs.mean(axis=0)

    # Interaction effect per row = sum of the absolute off-diagonal entries. The matrix is
    # symmetric (interaction(i,j) == interaction(j,i)), so summing the whole off-diagonal and
    # dividing by 2 avoids double-counting each pair.
    abs_interactions = np.abs(interaction_values)
    off_diagonal_mask = ~np.eye(n_features, dtype=bool)
    total_interaction_per_row = abs_interactions[:, off_diagonal_mask].reshape(len(sample_df), -1).sum(axis=1) / 2
    total_main_effect_per_row = main_effect_abs.sum(axis=1)

    mean_total_interaction = total_interaction_per_row.mean()
    mean_total_main_effect = total_main_effect_per_row.mean()
    interaction_share = mean_total_interaction / (mean_total_interaction + mean_total_main_effect)

    print(f"\nMean total |main effect| per row (summed over features): {mean_total_main_effect:.4f}")
    print(f"Mean total |interaction effect| per row (summed over pairs): {mean_total_interaction:.4f}")
    print(f"Interaction share of total |SHAP| magnitude: {interaction_share:.1%}")

    # Top individual feature pairs by mean |interaction|, across the sample.
    mean_interaction_matrix = abs_interactions.mean(axis=0)  # (n_features, n_features)
    pair_strengths = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            pair_strengths.append((feature_names[i], feature_names[j], mean_interaction_matrix[i, j]))
    pair_strengths.sort(key=lambda row: row[2], reverse=True)

    print("\nTop 10 feature pairs by mean |interaction|:")
    for name_i, name_j, strength in pair_strengths[:10]:
        print(f"  {name_i} x {name_j}: {strength:.4f}")

    print("\nTop 10 features by mean |main effect|:")
    ranked_main = sorted(zip(feature_names, mean_main_effect_per_feature), key=lambda row: row[1], reverse=True)
    for name, strength in ranked_main[:10]:
        print(f"  {name}: {strength:.4f}")


if __name__ == "__main__":
    main()

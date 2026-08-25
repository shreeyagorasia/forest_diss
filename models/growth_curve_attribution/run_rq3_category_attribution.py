# Run as: python -m models.growth_curve_attribution.run_rq3_category_attribution --cohort 4survey
#
# RQ3's category-level attribution. "which KIND of environmental variable (terrain/wind/soil/
# climate/edge/stand) explains the local curve deviation", distinct from run_rq3_en_xgb.py's
# per-VARIABLE coefficients/gain-importance. Reuses models/xgb_environmental/grouped_analysis.py
# wholesale. The same toolkit already used and validated in
# notebooks/growth_curve_attribution/av2_local_growth_curve_grouped_importance.ipynb. Rather
# than writing new category-attribution statistics. Only new code here is the wiring: loading
# RQ3's Set4 raw-column list from the manifest, building its table, and calling the existing
# grouped_permutation_importance()/category_morans_i_before_after() functions on it.
#
# Set4 (nested_set4_gated_all_vif) ONLY. The only RQ3 tier with at least one column from every
# CATEGORY_GROUPS category (checked directly against the manifest, 2026-08-10: Set2 has no
# soil_site/climate; Set3 "gated_terrain_wind" has no soil_site/climate/edge at all, and its own
# "wind" share is thin. Just topex/windward_topex, none of the GWA speed/Weibull columns RQ1's
# Set3 has). Running this on Set2/Set3 would report those missing categories at 0% importance --
# not because they don't matter, but because no candidate column from them was ever in the model.
#
# Single spatial_block split (not the 5-fold spatial CV run_rq3_en_xgb.py uses for the headline
# R2 numbers). Matches the precedent already set by the notebook this reuses, and
# category_morans_i_before_after() already does 7 refits (baseline + 6 categories removed) per
# call; running that 5x over folds would be 35 refits for an interpretability layer, not the
# headline accuracy comparison.

import argparse
import json

import pandas as pd

from models.common.saving import model_output_dir
from models.common.splits import SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, SPLIT_SEED, spatial_block_split
from models.elasticnet_environmental.elasticnet_environmental import drop_rows_with_missing_features
from models.growth_curve_attribution.broad_environmental_check import prepare_broad_table
from models.xgb_environmental.feature_set_builder import load_feature_set
from models.xgb_environmental.grouped_analysis import (
    CATEGORY_GROUPS,
    category_morans_i_before_after,
    expand_category_groups_for_columns,
    grouped_permutation_importance,
)
from models.xgb_environmental.xgb_environmental import fit_with_columns as xgb_fit
from models.xgb_environmental.xgb_environmental import predict_with_columns as xgb_predict

SET_NAME = "nested_set4_gated_all_vif"
TARGET = "local_y_max_difference"
MODEL_NAME = "rq3_category_attribution"


def run_one_cohort(cohort, seed=SPLIT_SEED):
    print(f"===== RQ3 category attribution: {cohort} / {SET_NAME} =====")

    raw_columns = load_feature_set("RSQ3", SET_NAME)
    # Same bare/dummy unpack-and-filter pattern as run_columns() (broad_environmental_check.py)
    #. Prepare_broad_table() only knows how to expand a BARE categorical name into every one of
    # its dummies, so specific dummy requests are unpacked back to their bare category first, then
    # the full dummy expansion is filtered back down to just what was actually requested.
    bare_columns = list(dict.fromkeys(column.split("=")[0] for column in raw_columns))
    table, all_model_columns = prepare_broad_table(cohort, bare_columns)

    continuous_requested = [column for column in raw_columns if "=" not in column]
    requested_dummies = [column for column in raw_columns if "=" in column]
    model_columns = continuous_requested + [column for column in requested_dummies if column in all_model_columns]
    missing_dummies = [column for column in requested_dummies if column not in all_model_columns]
    if missing_dummies:
        raise KeyError(f"Requested dummy columns not found after encoding: {missing_dummies}")

    category_groups = expand_category_groups_for_columns(CATEGORY_GROUPS, model_columns)
    empty_categories = [category for category, columns in category_groups.items() if not columns]
    if empty_categories:
        print(f"  Categories with zero matched columns in {SET_NAME}: {empty_categories}")

    table["split"] = spatial_block_split(
        table, block_col=SPATIAL_BLOCK_COL, buffer_distance=SPATIAL_BUFFER_METRES, seed=seed,
    )
    train = drop_rows_with_missing_features(table[table["split"] == "train"], model_columns)
    test = drop_rows_with_missing_features(table[table["split"] == "test"], model_columns).copy()
    print(f"  train rows: {len(train):,}  test rows: {len(test):,}  model columns: {len(model_columns)}")

    xgb_model = xgb_fit(train, model_columns, target_col=TARGET, n_estimators=500, max_depth=4, learning_rate=0.04)

    permutation_rows = []
    for category, category_columns in category_groups.items():
        if not category_columns:
            continue
        result = grouped_permutation_importance(
            xgb_model, test, model_columns, category_columns, xgb_predict, target_col=TARGET,
        )
        permutation_rows.append({"category": category, "n_columns": len(category_columns), **result})
    permutation_df = pd.DataFrame(permutation_rows)

    # "Before" (full model) + "after" (one category removed and refit) Moran's I on the
    # residuals. Does removing this category leave the residuals MORE spatially clustered than
    # before (that category was explaining real spatial structure), or barely change anything
    # (it wasn't responsible for the spatial pattern)?
    morans_df = category_morans_i_before_after(
        train, test, xgb_fit, xgb_predict,
        all_columns=model_columns, category_groups=category_groups, target_col=TARGET,
    )

    output_dir = model_output_dir(MODEL_NAME, cohort, split_type="spatial_block")
    output_dir.mkdir(parents=True, exist_ok=True)
    permutation_df.to_csv(output_dir / "category_permutation_importance.csv", index=False)
    morans_df.to_csv(output_dir / "category_morans_i_before_after.csv", index=False)
    with open(output_dir / "category_groups_used.json", "w") as f:
        json.dump(category_groups, f, indent=2)

    print("\n  Permutation importance by category (mean R2 drop when that category is shuffled):")
    print(permutation_df[["category", "n_columns", "mean_r2_drop", "std_r2_drop"]].to_string(index=False))
    print("\n  Moran's I before/after removing each category:")
    print(morans_df[["category_removed", "r2", "morans_i", "morans_i_p"]].to_string(index=False))
    print(f"\n  Saved -> {output_dir}")

    return permutation_df, morans_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="4survey", choices=["4survey", "6survey"])
    parser.add_argument("--seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()
    run_one_cohort(args.cohort, seed=args.seed)


if __name__ == "__main__":
    main()

# Run as (cluster only, see jobs/growth_curve_attribution/run_rq3_gnnwr_saturation_transform_check.sh):
#   python -m models.growth_curve_attribution.run_rq3_gnnwr_saturation_transform_check --held-out-fold 0 --k-folds 5 --use-gpu
#
# Tests whether GNNWR does better on Q2's target when given pre-saturated inputs instead of raw
# ones. GNNWR is structurally local-LINEAR -- it cannot represent a curve, even one that varies
# smoothly across space. The no-CanopyCover SHAP-dependence check (2026-08-22) found slope_degrees
# and windward_topex both have a clear SATURATING relationship to the target (steep-then-flat), not
# noise. This run swaps those two raw columns for pre-saturated versions
# (slope_degrees_capped15, windward_topex_clipped -- built by
# build_saturation_transformed_features.py, run once beforehand) so GNNWR's local straight line
# only needs to fit the already-linear middle section, not approximate the whole curve with one
# slope. Everything else about Set4 (the other 17 columns, the CanopyCover-inclusive population,
# the spatial-block CV) is unchanged -- this isolates the effect of the transform alone.
#
# Cheap, no-new-modelling test: same GNNWR architecture, same job script convention as every other
# GNNWR ablation in this project, only the input columns differ.

import argparse

import models.xgb_environmental.data as xgb_data_module
from models.common.splits import DEFAULT_K_FOLDS, SPLIT_SEED
from models.growth_curve_attribution.gnnwr_check import run_gnnwr
from models.xgb_environmental.feature_set_builder import load_feature_set

SET_NAME = "nested_set4_gated_all_vif"
ABLATION_SCOPE_LABEL = "nested_set4_saturation_transformed"

# Points every caller of load_environmental_features() (gnnwr_check.py's build_table_from_columns
# -> broad_environmental_check.py's prepare_broad_table) at the augmented parquet instead of the
# real one -- FEATURES_PATH is read inside load_environmental_features() at call time, so
# reassigning the module-level constant here is enough; no need to patch every import site
# separately. Original file is never touched (see build_saturation_transformed_features.py).
xgb_data_module.FEATURES_PATH = (
    xgb_data_module.FEATURES_PATH.parent / "plot_environmental_features_saturation_transformed.parquet"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="4survey", choices=["4survey", "6survey"])
    parser.add_argument("--max-epoch", type=int, default=200)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--reference-set-size", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--held-out-fold", type=int, default=None)
    parser.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS)
    args = parser.parse_args()

    if not xgb_data_module.FEATURES_PATH.exists():
        raise FileNotFoundError(
            f"{xgb_data_module.FEATURES_PATH} not found -- run "
            "`python -m models.growth_curve_attribution.build_saturation_transformed_features` first."
        )

    raw_columns = load_feature_set("RSQ3", SET_NAME)
    transformed_columns = [
        {"slope_degrees": "slope_degrees_capped15", "windward_topex": "windward_topex_clipped"}.get(c, c)
        for c in raw_columns
    ]
    print(f"RQ3 GNNWR saturation-transform check: {args.cohort} / {ABLATION_SCOPE_LABEL}")
    print(f"  swapped columns: slope_degrees -> slope_degrees_capped15, windward_topex -> windward_topex_clipped")
    print(f"  full column list: {transformed_columns}")

    run_gnnwr(
        cohort=args.cohort,
        scope=ABLATION_SCOPE_LABEL,
        max_epoch=args.max_epoch,
        early_stop=args.early_stop,
        use_gpu=args.use_gpu,
        reference_set_size=args.reference_set_size if args.reference_set_size > 0 else None,
        split_seed=args.split_seed,
        held_out_fold=args.held_out_fold,
        k_folds=args.k_folds,
        raw_columns=transformed_columns,
    )


if __name__ == "__main__":
    main()

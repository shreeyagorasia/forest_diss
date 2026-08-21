# Run as (cluster only, see jobs/growth_curve_attribution/run_rq3_gnnwr_canopycover_ablation.sh):
#   python -m models.growth_curve_attribution.run_rq3_gnnwr_canopycover_ablation --cohort 4survey --held-out-fold 0 --k-folds 5 --use-gpu
#
# CanopyCover-dropped ablation of Q2's own GNNWR model (Set4, i.e. RQ3's nested_set4_gated_all_vif).
# Thin copy of run_rq3_gnnwr.py -- only change is CanopyCover removed from raw_columns before the
# call to run_gnnwr(), and a distinct scope label ("nested_set4_no_canopycover") so this run's
# output never collides with or overwrites the real production Set4 GNNWR results (run_name is
# built from scope+cohort+reference-size+fold+seed in gnnwr_check.py's run_gnnwr(), and scope here
# is a plain label -- it does not affect which columns are used once raw_columns is passed).
#
# Purpose: the equivalent EN/XGBoost-without-CanopyCover ablation (already run locally, see
# models/growth_curve_attribution/rq3_canopycover_ablation_check.py) found Q2's R2 collapses from
# 0.240/0.250 (EN/XGBoost, with CanopyCover) to 0.042/0.061 (without) -- far more severe than the
# same check on Q1's own target. This checks whether GNNWR's own R2 edge over EN/XGBoost, and its
# Moran's I reduction, survive the same removal, or whether they were mostly riding on CanopyCover
# too.

import argparse

from models.common.splits import DEFAULT_K_FOLDS, SPLIT_SEED
from models.growth_curve_attribution.gnnwr_check import run_gnnwr
from models.xgb_environmental.feature_set_builder import load_feature_set

SET_NAME = "nested_set4_gated_all_vif"
ABLATION_SCOPE_LABEL = "nested_set4_no_canopycover"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="4survey", choices=["4survey", "6survey"])
    parser.add_argument("--max-epoch", type=int, default=200)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    # Default 0 (full/uncapped) -- the user's own cluster timing check found the capped and
    # uncapped runs take the same wall-clock time at this project's scale, so there is no reason
    # to cap here; matches how every other headline GNNWR result in this project is run.
    parser.add_argument("--reference-set-size", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--held-out-fold", type=int, default=None)
    parser.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS)
    args = parser.parse_args()

    raw_columns = load_feature_set("RSQ3", SET_NAME)
    ablated_columns = [c for c in raw_columns if c != "CanopyCover"]
    print(
        f"RQ3 GNNWR CanopyCover ablation: {args.cohort} / {SET_NAME} "
        f"({len(raw_columns)} raw columns -> {len(ablated_columns)} without CanopyCover)"
    )
    if len(ablated_columns) == len(raw_columns):
        raise ValueError("CanopyCover was not found in the Set4 column list -- check the column name before proceeding.")

    run_gnnwr(
        cohort=args.cohort,
        scope=ABLATION_SCOPE_LABEL,  # label only -- ignored for column resolution, but keeps output separate from the real Set4 run
        max_epoch=args.max_epoch,
        early_stop=args.early_stop,
        use_gpu=args.use_gpu,
        reference_set_size=args.reference_set_size if args.reference_set_size > 0 else None,
        split_seed=args.split_seed,
        held_out_fold=args.held_out_fold,
        k_folds=args.k_folds,
        raw_columns=ablated_columns,
    )


if __name__ == "__main__":
    main()

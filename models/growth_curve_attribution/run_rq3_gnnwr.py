# Run as (cluster only, see jobs/growth_curve_attribution/run_rq3_gnnwr.sh):
#   python -m models.growth_curve_attribution.run_rq3_gnnwr --cohort 4survey --set-name nested_set2_top5 --use-gpu
#
# RQ3's GNNWR driver for the new rank-aggregate environmental-feature methodology -- calls
# gnnwr_check.py's run_gnnwr() with raw_columns loaded from
# documentation/env_feature_sets_manifest.csv, instead of a named SCOPES entry. run_gnnwr() itself
# was extended with an optional raw_columns= parameter (2026-08-10) that, when given, resolves the
# table via build_table_from_columns() instead of build_scope_table() -- everything else about
# training/output-saving is untouched, so this wrapper is deliberately thin: parse args, load the
# Set, call run_gnnwr(), done. `scope` is still passed through as the RQ3 set_name, purely for
# run_gnnwr()'s own output-naming/logging -- it has no effect on which columns are actually used
# once raw_columns is given.

import argparse

from models.common.splits import DEFAULT_K_FOLDS, SPLIT_SEED
from models.growth_curve_attribution.gnnwr_check import DEFAULT_REFERENCE_SET_SIZE, run_gnnwr
from models.xgb_environmental.feature_set_builder import load_feature_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="4survey", choices=["4survey", "6survey"])
    parser.add_argument(
        "--set-name", default="nested_set2_top5",
        choices=["nested_set2_top5", "nested_set3_gated_terrain_wind", "nested_set4_gated_all", "nested_set5_all_ungated"],
    )
    parser.add_argument("--max-epoch", type=int, default=200)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--reference-set-size", type=int, default=DEFAULT_REFERENCE_SET_SIZE)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--held-out-fold", type=int, default=None)
    parser.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS)
    args = parser.parse_args()

    raw_columns = load_feature_set("RSQ3", args.set_name)
    print(f"RQ3 GNNWR: {args.cohort} / {args.set_name} ({len(raw_columns)} raw columns)")

    run_gnnwr(
        cohort=args.cohort,
        scope=args.set_name,  # label only -- ignored for column resolution once raw_columns is set
        max_epoch=args.max_epoch,
        early_stop=args.early_stop,
        use_gpu=args.use_gpu,
        reference_set_size=args.reference_set_size if args.reference_set_size > 0 else None,
        split_seed=args.split_seed,
        held_out_fold=args.held_out_fold,
        k_folds=args.k_folds,
        raw_columns=raw_columns,
    )


if __name__ == "__main__":
    main()

# Run as (cluster only, see jobs/growth_curve_attribution/run_rq3_gnnwr_canopycover_only.sh):
#   python -m models.growth_curve_attribution.run_rq3_gnnwr_canopycover_only --cohort 4survey --held-out-fold 0 --k-folds 5 --use-gpu
#
# CanopyCover-ONLY GNNWR on Q2's target (local_y_max_difference), Set4/4survey -- the mirror image
# of run_rq3_gnnwr_canopycover_ablation.py (which drops CanopyCover, keeping the other 18
# columns). Together the two ablations complete the matrix: with-all-19 (already run, R2=0.294),
# without-CanopyCover (already run, R2=0.049), and this one (CanopyCover alone). Purpose: check
# how much of GNNWR's full-Set4 R2 a single dominant variable can reach on its own -- if this gets
# close to 0.294, the other 18 variables are adding almost nothing even when CanopyCover is
# present, not just when it's absent.

import argparse

from models.common.splits import DEFAULT_K_FOLDS, SPLIT_SEED
from models.growth_curve_attribution.gnnwr_check import run_gnnwr

ABLATION_SCOPE_LABEL = "nested_set4_canopycover_only"


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

    raw_columns = ["CanopyCover"]
    print(f"RQ3 GNNWR CanopyCover-only: {args.cohort} / {ABLATION_SCOPE_LABEL} (1 raw column)")

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
        raw_columns=raw_columns,
    )


if __name__ == "__main__":
    main()

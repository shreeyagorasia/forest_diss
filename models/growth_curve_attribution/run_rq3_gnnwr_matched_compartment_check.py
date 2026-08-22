# Run as (cluster only, see jobs/growth_curve_attribution/run_rq3_gnnwr_matched_compartment_check.sh):
#   python -m models.growth_curve_attribution.run_rq3_gnnwr_matched_compartment_check --held-out-fold 0 --k-folds 5 --use-gpu
#
# Tests the leading open hypothesis for the unexplained 6-survey R2 collapse: is it having FEW
# COMPARTMENTS (47, vs. 4-survey's 231), not just fewer total points? An earlier check downsampled
# 4-survey's POINT COUNT to match 6-survey's while keeping all ~231 compartments (stratified
# sampling) and did NOT trigger a collapse (R2 stayed 0.12-0.34) -- that ruled out "too few points"
# but never actually tested "too few compartments" on its own, since it kept every compartment
# represented. This script does the opposite: keeps FULL point density, but restricts 4-survey to
# a random 47-compartment subset (matching 6-survey's own compartment count exactly), using
# gnnwr_check.py's compartment_subset= parameter (added 2026-08-22 specifically for this test).
#
# Always Set4 (nested_set4_gated_all_vif, the same set as every other headline GNNWR number) --
# this is a structural/sample-size question, not a feature-set question.

import argparse

from models.common.splits import DEFAULT_K_FOLDS, SPLIT_SEED
from models.growth_curve_attribution.gnnwr_check import run_gnnwr
from models.growth_curve_attribution.scale_comparison_check import build_plot_level_table
from models.xgb_environmental.feature_set_builder import load_feature_set

SET_NAME = "nested_set4_gated_all_vif"
ABLATION_SCOPE_LABEL = "nested_set4_matched_47_compartments"
N_COMPARTMENTS_TO_MATCH = 47  # 6-survey's actual compartment count, see scale_comparison_check.py
COMPARTMENT_SAMPLE_SEED = 42  # fixed so the same 47 compartments are picked every run


def pick_matched_compartment_subset():
    plot_table = build_plot_level_table("4survey", apply_disturbance_cleaning=True)
    all_compartments = sorted(plot_table["cpmt"].unique())
    print(f"4-survey has {len(all_compartments)} compartments total; sampling {N_COMPARTMENTS_TO_MATCH} to match 6-survey's count")
    import random

    rng = random.Random(COMPARTMENT_SAMPLE_SEED)
    subset = sorted(rng.sample(all_compartments, N_COMPARTMENTS_TO_MATCH))
    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-epoch", type=int, default=200)
    parser.add_argument("--early-stop", type=int, default=20)
    parser.add_argument("--use-gpu", action="store_true", default=False)
    parser.add_argument("--reference-set-size", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument("--held-out-fold", type=int, default=None)
    parser.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS)
    args = parser.parse_args()

    raw_columns = load_feature_set("RSQ3", SET_NAME)
    compartment_subset = pick_matched_compartment_subset()
    print(f"RQ3 GNNWR matched-compartment check: 4survey / {ABLATION_SCOPE_LABEL} ({len(raw_columns)} raw columns, {len(compartment_subset)} compartments)")

    run_gnnwr(
        cohort="4survey",
        scope=ABLATION_SCOPE_LABEL,
        max_epoch=args.max_epoch,
        early_stop=args.early_stop,
        use_gpu=args.use_gpu,
        reference_set_size=args.reference_set_size if args.reference_set_size > 0 else None,
        split_seed=args.split_seed,
        held_out_fold=args.held_out_fold,
        k_folds=args.k_folds,
        raw_columns=raw_columns,
        compartment_subset=compartment_subset,
    )


if __name__ == "__main__":
    main()

# Purpose: the llm-council's "one thing to do first" diagnostic for pinn_env_terrain_k's -0.71
# y_max/k correlation (see documentation/experiment_log.md's 2026-08-03 entry) -- compare it
# against the covariance the CLASSICAL (non-neural) Chapman-Richards fit's own y_max/k already
# has, using the exact same train rows (same cohort, same split_type, same split_seed) the frozen
# CR anchor itself was fit on.
#
# Why this matters: y_max and k trading off against each other is a well-known identifiability
# property of the Chapman-Richards curve, independent of any neural network -- a slightly lower
# ceiling reached slightly faster can produce nearly the same curve, over a limited observed age
# range, as a higher ceiling reached slower. If the classical fit ALREADY shows a comparable
# negative y_max/k correlation, pinn_env_terrain_k's two sub-networks didn't discover a new
# confound -- they inherited one baked into the curve shape itself. If the classical fit's
# correlation is near zero instead, the -0.71 is more likely something specific to the neural
# architecture (e.g. two structurally identical sub-networks fed identical terrain inputs).
#
# Run as: python -m models.pinn_env_terrain_k.check_cr_identifiability --cohort 4survey --split-type spatial_block

import argparse
import json

from models.baselines.run_baselines import MATURITY_AGE_MIN_DEFAULT, build_split_for_cohort
from models.chapman_richards.chapman_richards import fit as fit_chapman_richards
from models.common.saving import model_output_dir
from models.common.splits import SPLIT_SEED


def compute_correlation_from_covariance(pcov):
    # pcov is the 3x3 parameter covariance matrix scipy's curve_fit returns, in the same
    # (y_max, k, p) order chapman_richards() takes its arguments -- so index 0 is y_max and
    # index 1 is k. Converting a covariance to a correlation is just normalising by each
    # parameter's own variance: corr(a, b) = cov(a, b) / sqrt(var(a) * var(b)) -- the same
    # formula np.corrcoef() uses internally, just applied to curve_fit's covariance directly
    # instead of to a sample of per-plot values (there's only one classical fit, not one per
    # plot, so there's nothing to call np.corrcoef() on here).
    y_max_k_covariance = pcov[0, 1]
    y_max_variance = pcov[0, 0]
    k_variance = pcov[1, 1]
    return y_max_k_covariance / (y_max_variance * k_variance) ** 0.5


def check_for_cohort(cohort, split_type, split_seed):
    # Reuses run_baselines.py's own train-split construction directly (not re-implemented here)
    # so this checks the EXACT SAME rows the frozen CR anchor (that pinn_env_terrain_k actually
    # reads via load_cr_params) was itself fit on -- not a re-derived approximation of it.
    filtered_df, _ = build_split_for_cohort(
        cohort, split_type, split_seed=split_seed, maturity_age_min=MATURITY_AGE_MIN_DEFAULT,
    )
    cr_train_df = filtered_df[filtered_df["split"] == "train"]

    params, pcov = fit_chapman_richards(cr_train_df, return_covariance=True)
    classical_correlation = compute_correlation_from_covariance(pcov)

    print(f"  Classical (non-neural) CR fit, {len(cr_train_df):,} train rows:")
    print(f"    y_max={params['y_max']:.4f}  k={params['k']:.6f}  p={params['p']:.6f}")
    print(f"  Classical y_max/k correlation (from curve_fit's own parameter covariance): {classical_correlation:+.4f}")

    # Compare directly against pinn_env_terrain_k's own already-saved metrics.json, if it exists
    # for this cohort/split_type -- read, not assumed, since the exact seed/config it was
    # evaluated under may not match what's on disk.
    pinn_metrics_path = model_output_dir("pinn_env_terrain_k", cohort, split_type=split_type) / "metrics.json"
    if pinn_metrics_path.exists():
        with open(pinn_metrics_path) as f:
            pinn_metrics = json.load(f)
        neural_correlation = pinn_metrics.get("y_max_k_correlation")
        if neural_correlation is None:
            print(f"  (No 'y_max_k_correlation' key in {pinn_metrics_path} -- re-run evaluate_pinn_env_terrain_k.py.)")
        else:
            print(f"  pinn_env_terrain_k's learned y_max/k correlation (from its own metrics.json): {neural_correlation:+.4f}")
            # IMPORTANT CAVEAT, not a simple apples-to-apples comparison: the classical number is
            # a PARAMETER-ESTIMATION-UNCERTAINTY correlation for ONE pooled global fit (how
            # jointly uncertain the single y_max/k point estimate is) -- the neural number is a
            # CROSS-SECTIONAL POPULATION correlation across many different plots' own learned
            # per-plot (y_max, k) pairs. They are not the same statistic, so don't expect them to
            # match numerically -- but a classical correlation this extreme (near -1) is still
            # strong, independent evidence that y_max/k are hard to pin down independently from
            # height-vs-age data in this dataset at all, which makes it MORE plausible (not
            # ruled out) that the same underlying curve-shape ambiguity also limits how well
            # per-plot terrain features can identify two separate per-plot parameters -- this
            # script deliberately does NOT collapse that into a single pass/fail verdict.
            print(
                "  READ: these are DIFFERENT statistics (single-fit estimation-uncertainty "
                "correlation vs. cross-plot population correlation of learned adjustments) -- "
                "not directly comparable number-for-number. But the classical correlation being "
                f"this extreme ({classical_correlation:+.4f}, near -1) is itself strong evidence "
                "that y_max and k are weakly identified from this dataset's height-vs-age shape "
                "in general, which supports (does not rule out) the neural network hitting the "
                "same underlying ambiguity per plot. This does NOT by itself distinguish that "
                "from an architecture-specific artefact (two identical sub-networks on identical "
                "inputs) -- see the freeze-one-vary-other ablation for that question instead."
            )
    else:
        print(
            f"  (No pinn_env_terrain_k metrics.json found at {pinn_metrics_path} -- run "
            "evaluate_pinn_env_terrain_k.py first to compare against a real neural result.)"
        )

    return classical_correlation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default=None, help="Omit to run both cohorts.")
    parser.add_argument(
        "--split-type",
        choices=["plot_level", "spatial_block", "spatial_block_kfold", "temporal", "temporal_narrow_gap"],
        default="spatial_block",
    )
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    cohorts = [args.cohort] if args.cohort else ["4survey", "6survey"]
    for cohort in cohorts:
        print(f"===== {cohort} ({args.split_type}, split_seed={args.split_seed}) =====")
        check_for_cohort(cohort, args.split_type, args.split_seed)
        print()


if __name__ == "__main__":
    main()

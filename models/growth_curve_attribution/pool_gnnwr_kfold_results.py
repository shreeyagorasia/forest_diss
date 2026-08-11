# Purpose: after running gnnwr_check.py once per fold (--held-out-fold 0..k_folds-1, via
# jobs/growth_curve_attribution/run_gnnwr.sh), pool the resulting per-fold test-prediction CSVs
# into ONE pooled R2 -- the same "every plot's out-of-fold prediction, all folds concatenated,
# one R2 over the whole population" convention this project's Elastic Net/XGBoost spatial CV
# already uses (see spatial_cv_check.py's summarize_spatial_cv()), so GNNWR's headline number is
# finally comparable to the EN/XGBoost baseline (0.125/0.117 for terrain_wind), not just a single
# train/val/test split's one noisy estimate.
#
# Run this AFTER all k_folds jobs have finished and saved their CSVs -- it does not run GNNWR
# itself, just reads the already-saved outputs.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models.common.metrics import compute_metrics
from models.common.splits import DEFAULT_K_FOLDS
from models.growth_curve_attribution.gnnwr_check import OUTPUT_DIR, SCOPES
from models.growth_curve_attribution.scale_comparison_check import TARGET


def load_fold_predictions(cohort: str, scope: str, reference_set_size: int, k_folds: int = DEFAULT_K_FOLDS) -> pd.DataFrame:
    reference_set_label = "full" if reference_set_size is None else str(reference_set_size)
    fold_tables = []
    missing_folds = []
    for fold in range(k_folds):
        run_name = f"gnnwr_{scope}_{cohort}_ref{reference_set_label}_fold{fold}of{k_folds}"
        csv_path = OUTPUT_DIR / f"{run_name}_test_predictions.csv"
        if not csv_path.exists():
            missing_folds.append(fold)
            continue
        fold_table = pd.read_csv(csv_path)
        fold_table["held_out_fold"] = fold
        fold_tables.append(fold_table)

    if missing_folds:
        print(f"  Warning: missing CSVs for fold(s) {missing_folds} -- only pooling the {len(fold_tables)} folds found so far.")
    if not fold_tables:
        raise FileNotFoundError(f"No fold CSVs found for {cohort}/{scope}/ref{reference_set_label} in {OUTPUT_DIR}")

    return pd.concat(fold_tables, ignore_index=True)


def summarize_gnnwr_kfold(pooled: pd.DataFrame) -> dict:
    # Same convention as spatial_cv_check.py's summarize_spatial_cv(): a pooled R2 over every
    # plot across every fold (the headline number), plus the per-fold R2 mean/std so the spread
    # this pooled estimate is smoothing over stays visible, not hidden.
    pooled_r2 = compute_metrics(pooled[TARGET], pooled["denormalized_pred_result"])["r2"]

    per_fold_r2 = pooled.groupby("held_out_fold").apply(
        lambda rows: compute_metrics(rows[TARGET], rows["denormalized_pred_result"])["r2"],
        include_groups=False,
    )

    return {
        "pooled_r2": pooled_r2,
        "n_plots": len(pooled),
        "n_compartments": pooled["cpmt"].nunique(),
        "per_fold_r2_mean": float(per_fold_r2.mean()),
        "per_fold_r2_std": float(per_fold_r2.std()),
        "per_fold_r2_values": per_fold_r2.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default="4survey")
    parser.add_argument(
        "--scope", default="terrain_wind",
        choices=list(SCOPES) + [
            # Added 2026-08-11: the new nested_set* tiers (documentation/methodlogy_env_setpick.md)
            # aren't SCOPES entries -- scope is only ever used as a bare filename-pattern string
            # in load_fold_predictions()/the output path, never validated against SCOPES itself,
            # so widening this list is the only change needed to pool the new tiers' results too.
            "nested_set2_top10", "nested_set3_gated_terrain_wind_vif", "nested_set4_gated_all_vif",
        ],
    )
    parser.add_argument("--reference-set-size", type=int, default=16000, help="Pass 0 for the full-population runs.")
    parser.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS)
    args = parser.parse_args()

    reference_set_size = args.reference_set_size if args.reference_set_size > 0 else None
    pooled = load_fold_predictions(args.cohort, args.scope, reference_set_size, k_folds=args.k_folds)
    summary = summarize_gnnwr_kfold(pooled)

    print(f"\n{args.cohort} / {args.scope} / reference_set_size={reference_set_size or 'full'}")
    print(f"  Pooled R2 (headline, comparable to EN/XGBoost's own pooled 5-fold number): {summary['pooled_r2']:.4f}")
    print(f"  Per-fold R2: mean={summary['per_fold_r2_mean']:.4f}  std={summary['per_fold_r2_std']:.4f}")
    print(f"  Per-fold R2 values: {summary['per_fold_r2_values']}")
    print(f"  n_plots={summary['n_plots']:,}  n_compartments={summary['n_compartments']}")

    output_path = OUTPUT_DIR / f"gnnwr_{args.scope}_{args.cohort}_ref{reference_set_size or 'full'}_kfold_pooled_summary.csv"
    pd.DataFrame([summary]).to_csv(output_path, index=False)
    print(f"  Saved {output_path}")


if __name__ == "__main__":
    main()

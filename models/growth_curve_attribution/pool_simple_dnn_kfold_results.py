# Purpose: after running simple_dnn_check.py once per fold (--held-out-fold 0..k_folds-1), pool
# the resulting per-fold test-prediction CSVs into ONE pooled R2 -- the same "every plot's
# out-of-fold prediction, all folds concatenated, one R2 over the whole population" convention
# Elastic Net/XGBoost/GNNWR already use (see pool_gnnwr_kfold_results.py), so this plain-MLP
# control is finally comparable to those other models' own pooled 5-fold numbers instead of a
# single ~20% test-slice estimate.
#
# Run this AFTER all k_folds runs have finished and saved their CSVs -- it does not train
# anything itself, just reads the already-saved outputs.

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from models.common.metrics import compute_metrics
from models.common.splits import DEFAULT_K_FOLDS
from models.growth_curve_attribution.gnnwr_check import SCOPES
from models.growth_curve_attribution.scale_comparison_check import TARGET
from models.growth_curve_attribution.simple_dnn_check import OUTPUT_DIR


def load_fold_predictions(cohort: str, scope: str, k_folds: int = DEFAULT_K_FOLDS) -> pd.DataFrame:
    fold_tables = []
    missing_folds = []
    for fold in range(k_folds):
        csv_path = OUTPUT_DIR / f"simple_dnn_{scope}_{cohort}_fold{fold}of{k_folds}_test_predictions.csv"
        if not csv_path.exists():
            missing_folds.append(fold)
            continue
        fold_table = pd.read_csv(csv_path)
        fold_table["held_out_fold"] = fold
        fold_tables.append(fold_table)

    if missing_folds:
        print(f"  Warning: missing CSVs for fold(s) {missing_folds} -- only pooling the {len(fold_tables)} folds found so far.")
    if not fold_tables:
        raise FileNotFoundError(f"No fold CSVs found for {cohort}/{scope} in {OUTPUT_DIR}")

    return pd.concat(fold_tables, ignore_index=True)


def summarize_kfold(pooled: pd.DataFrame) -> dict:
    pooled_r2 = compute_metrics(pooled[TARGET], pooled["predicted"])["r2"]

    per_fold_r2 = pooled.groupby("held_out_fold").apply(
        lambda rows: compute_metrics(rows[TARGET], rows["predicted"])["r2"],
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
    parser.add_argument("--scope", choices=list(SCOPES), default="terrain_wind")
    parser.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS)
    args = parser.parse_args()

    pooled = load_fold_predictions(args.cohort, args.scope, k_folds=args.k_folds)
    summary = summarize_kfold(pooled)

    print(f"\n{args.cohort} / {args.scope}")
    print(f"  Pooled R2 (headline, comparable to EN/XGBoost/GNNWR's own pooled 5-fold number): {summary['pooled_r2']:.4f}")
    print(f"  Per-fold R2: mean={summary['per_fold_r2_mean']:.4f}  std={summary['per_fold_r2_std']:.4f}")
    print(f"  Per-fold R2 values: {summary['per_fold_r2_values']}")
    print(f"  n_plots={summary['n_plots']:,}  n_compartments={summary['n_compartments']}")

    output_path = OUTPUT_DIR / f"simple_dnn_{args.scope}_{args.cohort}_kfold_pooled_summary.csv"
    pd.DataFrame([summary]).to_csv(output_path, index=False)
    print(f"  Saved {output_path}")


if __name__ == "__main__":
    main()

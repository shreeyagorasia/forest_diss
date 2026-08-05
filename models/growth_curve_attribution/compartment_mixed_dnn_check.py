# Purpose: a neural version of a mixed-effects model -- an ordinary MLP on the environmental
# features (the "fixed effects"), PLUS one shrunk intercept per compartment (the "random
# effects"), added together. This directly answers a criticism of GNNWR raised in this project's
# own notes: this session already found real compartment-level clustering in the raw deviation
# target (ICC 0.399 for 4survey, 0.188 for 6survey -- see documentation/experiment_log.md) --
# evidence the true spatial structure is BLOCKY (values jump between compartments) rather than
# smoothly continuous, which is exactly what GNNWR's distance-based kernel assumes instead. This
# model encodes "blocky by compartment" directly, rather than "smooth by distance".
#
# Like simple_dnn_check.py, this model's size depends only on feature count and compartment
# count, NOT on the number of training ROWS -- so, same as the plain DNN, this runs comfortably
# on a laptop CPU. No cluster needed.
#
# TRAINING METHOD (2026-08-04, revised after a real bug was found and diagnosed): the first
# version of this script trained the fixed-effects network and the compartment intercepts
# JOINTLY with one Adam optimizer, with the compartment intercepts regularised by weight_decay
# (L2 shrinkage toward 0, mathematically equivalent to a Normal(0, sigma^2) random-effects
# prior). That converged to a compartment-intercept variance of essentially 0 REGARDLESS of the
# weight_decay value used (checked 0.0, 0.001, 0.01, 0.05 -- all gave ~0.019), which was a red
# flag: real shrinkage should be sensitive to its own strength. Diagnosed directly: validation
# loss (which drives early stopping and which checkpoint gets kept) NEVER involves the
# compartment intercepts, because every val/test compartment is one the model never saw in
# training (spatial_block_split holds out WHOLE compartments -- confirmed empirically, zero
# compartment overlap between train and val/test). So early stopping could halt training before
# the intercepts had converged at all, with nothing in the stopping criterion able to notice. A
# quick by-hand check confirmed real structure was being missed: a PLAIN pandas groupby-mean of
# a plain fixed-effects-only model's own training residuals, by compartment, showed a variance
# ratio of 0.688 (compartment-mean variance vs total residual variance) -- nowhere near the
# joint-trained model's reported ~0.000.
#
# Fixed by switching to the standard TWO-STAGE approach real mixed-effects software actually
# uses (this is not an approximation invented for this project -- it is the textbook empirical-
# Bayes / BLUP estimator for a one-way random-intercept model):
#   1. Fit the fixed-effects network ALONE first (reusing simple_dnn_check.py's SimpleMLP and
#      fit_mlp_with_early_stopping -- identical architecture and training loop, so this stage's
#      test R2 should come out the same as simple_dnn_check.py's own result).
#   2. With that network now FROZEN, compute each training compartment's mean residual, then
#      shrink it using this project's own compartment_pooling_check.compute_icc_one_way()
#      variance decomposition: shrinkage_i = variance_between / (variance_between +
#      variance_within / n_i). Small compartments (as few as 1 plot in this project's training
#      set) get pulled hard toward 0; large compartments (over 1,000 plots) are barely shrunk.
# This has no joint-optimisation convergence risk, because stage 2 is a closed-form calculation,
# not gradient descent.
#
# IMPORTANT, unchanged from before: because val/test compartments are never seen in training,
# there is no shrunk intercept available for them -- predictions there fall back to fixed
# effects alone (a random effect of exactly 0, the correct behaviour for a brand-new group with
# no data). So this model's TEST R2 is expected to come out close to simple_dnn_check.py's own
# result -- the compartment intercepts can only affect the TRAINING fit and the (separately
# useful) variance-decomposition report, not held-out prediction. That is an architectural fact
# of whole-compartment holdout, not a flaw in this script.
#
# Reuses build_scope_table() from gnnwr_check.py so target, split, and feature scope match every
# other model in this comparison.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models.common.metrics import compute_metrics
from models.common.splits import SPLIT_SEED
from models.growth_curve_attribution.compartment_pooling_check import compute_icc_one_way
from models.growth_curve_attribution.gnnwr_check import SCOPES, build_scope_table
from models.growth_curve_attribution.scale_comparison_check import TARGET
from models.growth_curve_attribution.simple_dnn_check import SimpleMLP, fit_mlp_with_early_stopping, standardize_features, to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "growth_curve_attribution" / "compartment_mixed_dnn"


def compute_shrunk_compartment_intercepts(train_residual, train_cpmt):
    # train_residual: the fixed-effects model's own residual (actual target minus its
    # prediction) for every training row. train_cpmt: which compartment each row belongs to.
    #
    # Returns a dict {compartment_id: shrunk_intercept} plus the variance decomposition, using
    # the standard empirical-Bayes / BLUP formula for a one-way random-intercept model:
    #   shrinkage_i = variance_between / (variance_between + variance_within / n_i)
    #   shrunk_intercept_i = shrinkage_i * (compartment i's own mean residual)
    # A compartment with very few rows (little of its own evidence) gets shrinkage_i close to 0
    # (pulled toward "no adjustment"); a compartment with lots of rows gets shrinkage_i close to
    # 1 (trusted almost as-is). This is exactly the partial-pooling behaviour a real
    # mixed-effects model gives, derived in closed form rather than found by gradient descent.
    decomposition = compute_icc_one_way(train_residual, train_cpmt)
    variance_between = decomposition["variance_between_compartments"]
    variance_within = decomposition["variance_within_compartments"]

    compartment_stats = pd.DataFrame({"residual": train_residual, "cpmt": train_cpmt}).groupby("cpmt")["residual"].agg(["mean", "count"])
    compartment_stats["shrinkage_factor"] = variance_between / (variance_between + variance_within / compartment_stats["count"])
    compartment_stats["shrunk_intercept"] = compartment_stats["shrinkage_factor"] * compartment_stats["mean"]

    intercept_lookup = compartment_stats["shrunk_intercept"].to_dict()
    return intercept_lookup, compartment_stats, decomposition


def run_compartment_mixed_dnn(
    cohort: str,
    scope: str,
    max_epoch: int = 300,
    patience: int = 20,
    seed: int = 42,
    hidden_sizes=(64, 32),
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    split_seed: int = SPLIT_SEED,
):
    torch.manual_seed(seed)

    table, feature_columns = build_scope_table(cohort, scope, split_seed=split_seed)
    train = table[table["split"] == "train"].copy()
    val = table[table["split"] == "val"].copy()
    test = table[table["split"] == "test"].copy()
    print(f"{cohort} / {scope}: train={len(train):,}  val={len(val):,}  test={len(test):,}  features={len(feature_columns)}")

    n_train_compartments = train["cpmt"].nunique()
    overlap_with_train = (set(val["cpmt"]) | set(test["cpmt"])) & set(train["cpmt"])
    print(f"  Train compartments: {n_train_compartments}  (val/test compartments overlapping train: {len(overlap_with_train)})")

    train_x_df, val_x_df, test_x_df = standardize_features(train, val, test, feature_columns)
    train_x, val_x, test_x = to_tensor(train_x_df), to_tensor(val_x_df), to_tensor(test_x_df)
    train_y, val_y, test_y = to_tensor(train[TARGET]), to_tensor(val[TARGET]), to_tensor(test[TARGET])

    # ----- Stage 1: fit the fixed-effects network alone, exactly like simple_dnn_check.py -----
    model = SimpleMLP(n_features=len(feature_columns), hidden_sizes=hidden_sizes, dropout=dropout)
    model, best_val_loss = fit_mlp_with_early_stopping(
        model, train_x, train_y, val_x, val_y, max_epoch=max_epoch, patience=patience, learning_rate=learning_rate,
    )

    with torch.no_grad():
        train_fixed_effect_pred = model(train_x).numpy()
        test_fixed_effect_pred = model(test_x).numpy()

    fixed_effects_only_metrics = compute_metrics(test[TARGET].values, test_fixed_effect_pred)
    print(f"  Fixed-effects-only test R2 = {fixed_effects_only_metrics['r2']:.4f}  (best validation loss = {best_val_loss:.4f})")

    # ----- Stage 2: closed-form shrunk compartment intercepts from the frozen network's own
    # training residuals -----
    train_residual = train[TARGET].values - train_fixed_effect_pred
    intercept_lookup, compartment_stats, decomposition = compute_shrunk_compartment_intercepts(train_residual, train["cpmt"])

    print(
        f"  Variance decomposition of residuals (after environment): "
        f"between-compartment={decomposition['variance_between_compartments']:.3f}, "
        f"within-compartment={decomposition['variance_within_compartments']:.3f}, "
        f"ICC (share explained by compartment identity)={decomposition['icc']:.3f}"
    )
    print(
        f"  Shrinkage factor range across the {n_train_compartments} train compartments: "
        f"min={compartment_stats['shrinkage_factor'].min():.3f}, max={compartment_stats['shrinkage_factor'].max():.3f} "
        f"(0 = fully shrunk to no adjustment, 1 = trusted as-is; small compartments shrink harder)"
    )

    # Test predictions: every test compartment is one the model never saw in training (see
    # module docstring), so there is no shrunk intercept for it -- fixed effects alone, which is
    # exactly test_fixed_effect_pred already computed above. Kept as an explicit variable here
    # (rather than reusing test_fixed_effect_pred silently) so this is visibly a DELIBERATE
    # architectural choice, not an oversight.
    test_predictions = test_fixed_effect_pred
    metrics = compute_metrics(test[TARGET].values, test_predictions)
    print(f"  Final test R2 (fixed effects only, as expected for held-out compartments) = {metrics['r2']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_table = test[["identification", "cpmt", TARGET]].copy()
    result_table["predicted"] = test_predictions
    output_path = OUTPUT_DIR / f"compartment_mixed_dnn_{scope}_{cohort}_test_predictions.csv"
    result_table.to_csv(output_path, index=False)

    compartment_stats_path = OUTPUT_DIR / f"compartment_mixed_dnn_{scope}_{cohort}_compartment_intercepts.csv"
    compartment_stats.to_csv(compartment_stats_path)
    print(f"  Saved {output_path}")
    print(f"  Saved {compartment_stats_path}")

    return metrics, decomposition, result_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default="4survey")
    parser.add_argument("--scope", choices=list(SCOPES), default="terrain_wind")
    parser.add_argument("--max-epoch", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    run_compartment_mixed_dnn(
        cohort=args.cohort,
        scope=args.scope,
        max_epoch=args.max_epoch,
        patience=args.patience,
        seed=args.seed,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()

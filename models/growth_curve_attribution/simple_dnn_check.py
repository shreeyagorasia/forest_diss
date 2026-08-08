# Purpose: a plain, ordinary feedforward neural network (a simple "multi-layer perceptron", or
# MLP) on the SAME per-plot growth-curve target and the SAME features as GNNWR and the Elastic
# Net / XGBoost models, as a control.
#
# Why this control matters: GNNWR is a neural network too, so if GNNWR beats XGBoost, that could
# be because (a) letting the environment-to-growth relationship vary SPATIALLY genuinely helps,
# or (b) it could just be that ANY neural network fits this data a bit better than a tree model,
# for reasons that have nothing to do with space at all. This plain MLP has no spatial-weighting
# machinery whatsoever -- every plot is treated identically regardless of where it is -- so
# comparing its test R2 against GNNWR's isolates whether the SPATIAL part of GNNWR is actually
# doing anything, separate from just "using a neural network."
#
# Unlike GNNWR, this network's size depends only on how many FEATURES there are (14-22), not on
# how many training ROWS there are -- so there is no repeat of the memory blow-ups documented in
# gnnwr_check.py. This script runs comfortably on a laptop CPU in seconds; no cluster needed.
#
# Reuses build_scope_table() from gnnwr_check.py so this uses the exact same cleaned target,
# compartment-based spatial_block_split, and feature scopes (terrain_wind /
# terrain_wind_plus_management) as GNNWR -- the only thing that differs between this model and
# GNNWR is the model architecture itself, so any R2 difference is a fair, apples-to-apples
# comparison.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.common.metrics import compute_metrics
from models.common.splits import DEFAULT_K_FOLDS, SPLIT_SEED
from models.growth_curve_attribution.gnnwr_check import SCOPES, build_scope_table
from models.growth_curve_attribution.scale_comparison_check import TARGET

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "growth_curve_attribution" / "simple_dnn"


class SimpleMLP(nn.Module):
    # A plain feedforward network: a chain of (Linear -> ReLU -> Dropout) blocks, ending in one
    # Linear layer with no activation function (the target, local_y_max_difference, can be
    # positive OR negative, so the last layer must be free to output any real number).
    def __init__(self, n_features, hidden_sizes=(64, 32), dropout=0.2):
        super().__init__()
        layers = []
        in_size = n_features
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = hidden_size
        layers.append(nn.Linear(in_size, 1))  # final layer: no activation, one number out
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)  # squeeze turns shape (rows, 1) into shape (rows,)


def standardize_features(train_df, val_df, test_df, feature_columns):
    # Standardize (subtract mean, divide by standard deviation) using ONLY the training set's
    # own mean/std -- the same "fit on train, apply to val/test" rule used everywhere else in
    # this project, so val/test information never leaks into how features are scaled.
    means = train_df[feature_columns].mean()
    stds = train_df[feature_columns].std()
    stds = stds.replace(0, 1)  # guard against a constant column dividing by zero
    train_scaled = (train_df[feature_columns] - means) / stds
    val_scaled = (val_df[feature_columns] - means) / stds
    test_scaled = (test_df[feature_columns] - means) / stds
    return train_scaled, val_scaled, test_scaled


def to_tensor(dataframe_or_series):
    return torch.tensor(dataframe_or_series.values, dtype=torch.float32)


def fit_mlp_with_early_stopping(model, train_x, train_y, val_x, val_y, max_epoch, patience, learning_rate, weight_decay=1e-5):
    # Shared by every plain-MLP model in this folder (simple_dnn_check.py and
    # compartment_mixed_dnn_check.py's fixed-effects stage) so there is exactly one training loop
    # to get right, not several near-identical copies that could quietly drift apart.
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.MSELoss()

    best_val_loss = float("inf")
    best_model_state = None
    epochs_since_improvement = 0

    # Full-batch training: every epoch, the whole training set goes through the network in one
    # go (no mini-batches / DataLoader needed). This is fine here because the network itself is
    # tiny (a handful of small Linear layers) -- unlike GNNWR, nothing here scales with the
    # number of training rows, so there is no memory concern doing it this simple way.
    for epoch in range(max_epoch):
        model.train()
        optimizer.zero_grad()
        train_predictions = model(train_x)
        train_loss = loss_function(train_predictions, train_y)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(val_x)
            val_loss = loss_function(val_predictions, val_y).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {name: value.clone() for name, value in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(f"  Stopped early at epoch {epoch} (no validation improvement for {patience} epochs)")
            break

    model.load_state_dict(best_model_state)  # go back to the epoch with the best validation loss
    model.eval()
    return model, best_val_loss


def run_simple_dnn(
    cohort: str,
    scope: str,
    max_epoch: int = 300,
    patience: int = 20,
    seed: int = 42,
    hidden_sizes=(64, 32),
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    split_seed: int = SPLIT_SEED,
    held_out_fold: int | None = None,
    k_folds: int = DEFAULT_K_FOLDS,
):
    # held_out_fold=None (default) keeps the original single train/val/test split. Passing
    # 0..k_folds-1 instead runs ONE fold of the same 5-fold spatial CV Elastic Net/XGBoost/GNNWR
    # already use (build_scope_table already supports this -- added for GNNWR, reused here
    # unchanged) -- run this once per fold, then pool the 5 test-prediction CSVs for a headline
    # number that is actually comparable to those other models' own pooled 5-fold R2, instead of
    # this model's previous single ~20% test-slice estimate.
    torch.manual_seed(seed)

    table, feature_columns = build_scope_table(
        cohort, scope, split_seed=split_seed, held_out_fold=held_out_fold, k_folds=k_folds,
    )
    train = table[table["split"] == "train"].copy()
    val = table[table["split"] == "val"].copy()
    test = table[table["split"] == "test"].copy()
    print(f"{cohort} / {scope}: train={len(train):,}  val={len(val):,}  test={len(test):,}  features={len(feature_columns)}")
    if held_out_fold is not None:
        print(f"  K-fold spatial CV: fold {held_out_fold} of {k_folds} held out as test (seed={split_seed})")

    train_x_df, val_x_df, test_x_df = standardize_features(train, val, test, feature_columns)
    train_x, val_x, test_x = to_tensor(train_x_df), to_tensor(val_x_df), to_tensor(test_x_df)
    train_y, val_y, test_y = to_tensor(train[TARGET]), to_tensor(val[TARGET]), to_tensor(test[TARGET])

    model = SimpleMLP(n_features=len(feature_columns), hidden_sizes=hidden_sizes, dropout=dropout)
    model, best_val_loss = fit_mlp_with_early_stopping(
        model, train_x, train_y, val_x, val_y, max_epoch=max_epoch, patience=patience, learning_rate=learning_rate,
    )

    with torch.no_grad():
        test_predictions = model(test_x).numpy()

    metrics = compute_metrics(test[TARGET].values, test_predictions)
    print(f"  Test R2 = {metrics['r2']:.4f}  (best validation loss = {best_val_loss:.4f})")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_table = test[["identification", "cpmt", TARGET]].copy()
    result_table["predicted"] = test_predictions
    fold_label = "" if held_out_fold is None else f"_fold{held_out_fold}of{k_folds}"
    output_path = OUTPUT_DIR / f"simple_dnn_{scope}_{cohort}{fold_label}_test_predictions.csv"
    result_table.to_csv(output_path, index=False)
    print(f"  Saved {output_path}")

    return metrics, result_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default="4survey")
    parser.add_argument("--scope", choices=list(SCOPES), default="terrain_wind")
    parser.add_argument("--max-epoch", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    parser.add_argument(
        "--held-out-fold", type=int, default=None,
        help="Run ONE fold of a 5-fold spatial CV instead of the default single split -- pass 0..k_folds-1, run once per fold, then pool the resulting CSVs.",
    )
    parser.add_argument("--k-folds", type=int, default=DEFAULT_K_FOLDS)
    args = parser.parse_args()

    run_simple_dnn(
        cohort=args.cohort,
        scope=args.scope,
        max_epoch=args.max_epoch,
        patience=args.patience,
        seed=args.seed,
        split_seed=args.split_seed,
        held_out_fold=args.held_out_fold,
        k_folds=args.k_folds,
    )


if __name__ == "__main__":
    main()

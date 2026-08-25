# Purpose: a small hyperparameter search over the plain-MLP architecture (SimpleMLP), trying to
# see whether tuning. Rather than just using one default architecture. Can close the gap with
# XGBoost (0.302 test R2 for terrain_wind_plus_management, 0.117 for terrain_wind).
#
# IMPORTANT: because compartment_mixed_dnn_check.py's compartment intercepts never apply to a
# held-out compartment (every val/test compartment is one the model never saw in training. See
# that file's own module docstring), its TEST R2 is architecturally IDENTICAL to whatever this
# plain fixed-effects-only MLP gets. So tuning THIS architecture is exactly the same optimisation
# problem as tuning the compartment-mixed DNN's test performance. One search covers both models.
#
# Selection discipline: every candidate configuration is picked by VALIDATION loss only. The test
# set is touched exactly ONCE, at the very end, for the single best-on-validation configuration --
# never used to choose between configurations. Repeatedly checking test R2 while trying different
# architectures would be a slow-motion version of the same leak this project found and fixed for
# XGBoost's eval-set earlier this session (models/growth_curve_attribution/scale_comparison_check.py).

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd
import torch

from models.common.metrics import compute_metrics
from models.common.splits import SPLIT_SEED
from models.growth_curve_attribution.gnnwr_check import SCOPES, build_scope_table
from models.growth_curve_attribution.scale_comparison_check import TARGET
from models.growth_curve_attribution.simple_dnn_check import SimpleMLP, fit_mlp_with_early_stopping, standardize_features, to_tensor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "growth_curve_attribution" / "simple_dnn"

# A modest, sensible grid. Not exhaustive, but covering the main knobs that actually change how
# much an MLP can fit: depth/width (hidden_sizes), regularisation strength (dropout, weight_decay),
# and how fast/coarsely it learns (learning_rate).
CANDIDATE_HIDDEN_SIZES = [(32, 16), (64, 32), (128, 64), (128, 64, 32), (256, 128, 64), (256, 128, 64, 32)]
CANDIDATE_DROPOUT = [0.0, 0.2, 0.4]
CANDIDATE_LEARNING_RATE = [3e-4, 1e-3, 3e-3]
CANDIDATE_WEIGHT_DECAY = [1e-5, 1e-4]


def run_search(cohort: str, scope: str, seed: int = 42, max_epoch: int = 300, patience: int = 20, split_seed: int = SPLIT_SEED):
    table, feature_columns = build_scope_table(cohort, scope, split_seed=split_seed)
    train = table[table["split"] == "train"].copy()
    val = table[table["split"] == "val"].copy()
    test = table[table["split"] == "test"].copy()
    print(f"{cohort} / {scope}: train={len(train):,}  val={len(val):,}  test={len(test):,}  features={len(feature_columns)}")

    train_x_df, val_x_df, test_x_df = standardize_features(train, val, test, feature_columns)
    train_x, val_x, test_x = to_tensor(train_x_df), to_tensor(val_x_df), to_tensor(test_x_df)
    train_y, val_y, test_y = to_tensor(train[TARGET]), to_tensor(val[TARGET]), to_tensor(test[TARGET])

    combos = list(itertools.product(CANDIDATE_HIDDEN_SIZES, CANDIDATE_DROPOUT, CANDIDATE_LEARNING_RATE, CANDIDATE_WEIGHT_DECAY))
    print(f"Trying {len(combos)} configurations (selecting by validation loss only)...")

    search_results = []
    best_val_loss = float("inf")
    best_config = None
    best_model_state = None

    for hidden_sizes, dropout, learning_rate, weight_decay in combos:
        torch.manual_seed(seed)
        model = SimpleMLP(n_features=len(feature_columns), hidden_sizes=hidden_sizes, dropout=dropout)
        model, val_loss = fit_mlp_with_early_stopping(
            model, train_x, train_y, val_x, val_y,
            max_epoch=max_epoch, patience=patience, learning_rate=learning_rate, weight_decay=weight_decay,
        )
        search_results.append({
            "hidden_sizes": hidden_sizes, "dropout": dropout, "learning_rate": learning_rate,
            "weight_decay": weight_decay, "val_loss": val_loss,
        })
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = (hidden_sizes, dropout, learning_rate, weight_decay)
            best_model_state = {name: value.clone() for name, value in model.state_dict().items()}

    hidden_sizes, dropout, learning_rate, weight_decay = best_config
    print(f"\nBest configuration by validation loss: hidden_sizes={hidden_sizes}, dropout={dropout}, "
          f"learning_rate={learning_rate}, weight_decay={weight_decay}  (val_loss={best_val_loss:.4f})")

    # Rebuild the winning architecture and load its saved weights, then touch the test set
    # exactly once, for this one configuration only.
    best_model = SimpleMLP(n_features=len(feature_columns), hidden_sizes=hidden_sizes, dropout=dropout)
    best_model.load_state_dict(best_model_state)
    best_model.eval()
    with torch.no_grad():
        test_predictions = best_model(test_x).numpy()
    metrics = compute_metrics(test[TARGET].values, test_predictions)
    print(f"Tuned test R2 (touched test set once, for this configuration only) = {metrics['r2']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    search_results_df = pd.DataFrame(search_results).sort_values("val_loss")
    search_path = OUTPUT_DIR / f"tune_simple_mlp_{scope}_{cohort}_search_results.csv"
    search_results_df.to_csv(search_path, index=False)
    print(f"Saved full search results to {search_path}")

    return metrics, best_config, search_results_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=["4survey", "6survey"], default="4survey")
    parser.add_argument("--scope", choices=list(SCOPES), default="terrain_wind")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epoch", type=int, default=300)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--split-seed", type=int, default=SPLIT_SEED)
    args = parser.parse_args()

    run_search(
        cohort=args.cohort, scope=args.scope, seed=args.seed,
        max_epoch=args.max_epoch, patience=args.patience, split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()

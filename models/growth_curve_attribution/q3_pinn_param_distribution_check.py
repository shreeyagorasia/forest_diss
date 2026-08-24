# Run as: PYTHONPATH=. .venv/bin/python -m models.growth_curve_attribution.q3_pinn_param_distribution_check
#
# Plots the full distribution of PINN-k's per-plot y_max_i and k_i predictions, pooled across
# all 5 spatial-block folds, bucketed against EACH FOLD'S OWN population Chapman-Richards curve.
# Built to check whether the single example plot used in Figure~\ref{fig:pinn-example-plot}
# (Q3 results) is representative or a special case -- reuses already-saved prediction files,
# no retraining.
#
# UPDATED 2026-08-24: was fold-0-only (n=11,508). Pooled here the same way as the rest of this
# session's Q3 figures (see build_q3_redraft_figures.py's load_pooled_pinn_k_deviation_data) --
# each fold refits its own population y_max/k baseline, so every plot's deviation is computed
# against ITS OWN fold's baseline before pooling, not one fixed baseline applied to all 5 folds
# (that would reintroduce the same Simpson's-paradox-style confound already found and fixed in
# the y_max-vs-k scatter figure this session).
#
# Finding still confirms, now on the full pooled set: y_max_i is tightly pinned near the
# population value for the large majority of plots (no implausible values at all); k_i is skewed
# strongly toward FASTER than population for the large majority -- not a symmetric spread of
# "some faster, some slower" real heterogeneity.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.common.saving import load_cr_params
from models.common.splits import SPLIT_SEED

FIGURES_DIR = "figures/fig_results/q3"

# --- Fold 0: from the already-saved example_curve predictions, same source as every other
# fold-0 PINN-k figure this session. ---
fold0 = pd.read_csv("temp_results_pinn/outputs/example_curve/test_set_predictions.csv")
fold0 = fold0[fold0["split"] == "test"].drop_duplicates("identification")
cr0 = load_cr_params("4survey", "spatial_block_kfold", split_seed=SPLIT_SEED, held_out_fold=0)
fold0["y_max_diff"] = fold0["y_max_pred"] - cr0["y_max"]
fold0["k_diff_pct"] = (fold0["k_pred"] - cr0["k"]) / cr0["k"] * 100
fold0 = fold0[["identification", "y_max_diff", "k_diff_pct"]]

# --- Folds 1-4: from the CORRECTED mechanism-check predictions, each carrying its OWN fold's
# population_y_max/population_k columns -- same files load_pooled_pinn_k_deviation_data() uses. ---
other_folds = []
for i in [1, 2, 3, 4]:
    df = pd.read_csv(
        f"temp_results_pinn/outputs/CORRECTED_2026-08-23_mechanism_checks/pinn_k_population_check/fold_{i}/predictions.csv"
    )
    df = df.drop_duplicates("identification")
    df["y_max_diff"] = df["y_max_pred"] - df["population_y_max"]
    df["k_diff_pct"] = (df["k_pred"] - df["population_k"]) / df["population_k"] * 100
    other_folds.append(df[["identification", "y_max_diff", "k_diff_pct"]])

plot_level = pd.concat([fold0] + other_folds, ignore_index=True)
n_before = len(plot_level)
plot_level = plot_level.drop_duplicates("identification")
print(f"Pooled 5-fold predictions: {n_before} rows -> {len(plot_level)} unique plots "
      "(should match, one held-out prediction per plot)")

# --- Bucket percentages, computed live from the pooled data (not hardcoded) ---
ymax_above_1m = (plot_level["y_max_diff"] > 1).mean() * 100
ymax_within_1m = (plot_level["y_max_diff"].abs() <= 1).mean() * 100
ymax_below_1m = (plot_level["y_max_diff"] < -1).mean() * 100

k_slower = (plot_level["k_diff_pct"] < 0).mean() * 100
k_near = plot_level["k_diff_pct"].between(0, 10).mean() * 100
k_moderate = plot_level["k_diff_pct"].between(10, 50).mean() * 100
k_much_faster = (plot_level["k_diff_pct"] > 50).mean() * 100

fig, (ax_ymax, ax_k) = plt.subplots(1, 2, figsize=(13, 5))

# --- Panel A: y_max_i distribution, bucketed ---
ymax_bucket_edges = [-3, -1, 1]  # boundaries in metres, vs. population y_max
ax_ymax.hist(plot_level["y_max_diff"], bins=60, color="#7B9E89", edgecolor="white", linewidth=0.3)
for edge in ymax_bucket_edges:
    ax_ymax.axvline(edge, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
ax_ymax.axvline(0, color="black", linewidth=1.2)
ax_ymax.set_xlabel("$y_{max,i}$ (PINN-$k$) $-$ own-fold population $y_{max}$ (m)")
ax_ymax.set_ylabel("Number of plots")
ax_ymax.set_title(
    f"$y_{{max,i}}$: tightly pinned near population for the large majority of plots\n"
    f"({ymax_above_1m:.1f}% above by >1m; {ymax_within_1m:.1f}% within +/-1m; "
    f"{ymax_below_1m:.1f}% below by 1m+)",
    fontsize=10,
)

# --- Panel B: k_i distribution, bucketed ---
k_bucket_edges = [0, 10, 50]  # boundaries in % difference from population k
ax_k.hist(plot_level["k_diff_pct"], bins=60, color="#C97B63", edgecolor="white", linewidth=0.3)
for edge in k_bucket_edges:
    ax_k.axvline(edge, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
ax_k.axvline(0, color="black", linewidth=1.2)
ax_k.set_xlabel("$k_i$ (PINN-$k$) vs. own-fold population $k$ (% difference)")
ax_k.set_ylabel("Number of plots")
ax_k.set_title(
    f"$k_i$: skewed toward FASTER than population for most plots\n"
    f"({k_slower:.1f}% slower; {k_near:.1f}% near population; {k_moderate:.1f}% moderately faster; "
    f"{k_much_faster:.1f}% much faster)",
    fontsize=10,
)

fig.suptitle(
    f"PINN-$k$'s per-plot parameters, pooled across all 5 folds (n={len(plot_level):,} plots): "
    "$y_{max,i}$ stays close to the population value; $k_i$ shows a population-wide shift, "
    "not symmetric heterogeneity",
    fontsize=11,
)
plt.tight_layout()
fig.savefig(f"{FIGURES_DIR}/q3_pinn_param_distribution.png", dpi=200, bbox_inches="tight")
print(f"Saved {FIGURES_DIR}/q3_pinn_param_distribution.png")

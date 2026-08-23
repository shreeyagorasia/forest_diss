# Builds the 5 new figures queued for the Q3 redraft (results_q3_23_08_0540am.tex, Blocks
# B/C/D/E). Uses the project's shared figure style (notebooks/results_figures_style.py) for
# consistency with every other results-chapter figure. All data already exists -- no new
# training, this is plotting only.
#
# Run: PYTHONPATH=. python temp_results_pinn/pinn_env_terrain_fix/build_q3_redraft_figures.py

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common.geo import load_compartment_boundaries
from models.common.saving import load_cr_params
from models.common.splits import SPLIT_SEED
from notebooks.results_figures_style import COLOR_PINN, COLOR_PINN_K, DIVERGING_CMAP, apply_rcparams

apply_rcparams()
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "fig_results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Figure 1 (Block C, main text): environment sweep, R2 vs feature-set breadth
# -----------------------------------------------------------------------------
def build_env_sweep_chart():
    conditions = ["No\nenvironment", "Set2\n(small)", "Set3\n(medium)", "Set4\n(large)"]
    pinn_r2 = [0.573, 0.627, 0.631, 0.618]
    pinn_k_r2 = [0.573, 0.622, 0.618, 0.620]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(conditions, pinn_r2, marker="o", color=COLOR_PINN, linewidth=2, label="PINN")
    ax.plot(conditions, pinn_k_r2, marker="o", color=COLOR_PINN_K, linewidth=2, label="PINN-$k$")
    ax.set_ylabel("Test $R^2$")
    ax.set_ylim(0.55, 0.65)
    ax.legend(frameon=False)
    ax.set_title("Environment helps once -- a bigger list does not help further")
    plt.tight_layout()
    out = FIGURES_DIR / "q3_env_sweep_r2.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 2 (Block D, main text): compartment-level map of y_max deviation
# -----------------------------------------------------------------------------
def build_compartment_map():
    preds = pd.read_csv("temp_results_pinn/outputs/example_curve/plain_pinn_fixed_test_set_predictions.csv")
    preds = preds.drop_duplicates("identification")

    master = pd.read_parquet("data/processed/master/clean_master_4survey.parquet")
    master_lookup = master[["identification", "blk", "cpmt"]].drop_duplicates("identification")
    preds = preds.merge(master_lookup, on="identification", how="left")

    cr_params = load_cr_params("4survey", "spatial_block_kfold", split_seed=SPLIT_SEED, held_out_fold=0)
    preds["deviation"] = preds["y_max_pred"] - cr_params["y_max"]

    compartment_mean = preds.groupby("cpmt")["deviation"].mean().reset_index()

    boundaries = load_compartment_boundaries()
    mapped = boundaries.merge(compartment_mean, on="cpmt", how="left")

    shared_vmax = mapped["deviation"].abs().quantile(0.98)
    norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-shared_vmax, vmax=shared_vmax)

    fig, ax = plt.subplots(figsize=(7, 7))
    mapped.plot(column="deviation", cmap=DIVERGING_CMAP, norm=norm, ax=ax,
                edgecolor="#cccccc", linewidth=0.2, missing_kwds={"color": "#f5f5f5"})
    sm = plt.cm.ScalarMappable(cmap=DIVERGING_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean y_max deviation from population curve (m)")
    ax.set_axis_off()
    ax.set_title("Where PINN's predicted ceiling drifts from the population curve")
    plt.tight_layout()
    out = FIGURES_DIR / "q3_pinn_ymax_deviation_map.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 3 (Block B, appendix): per-feature terrain importance bar chart
# -----------------------------------------------------------------------------
def build_importance_chart():
    # Fold 0's permutation-importance run predates the JSON-saving fix added later this
    # session (only folds 1/2 have saved files) -- these numbers are the fold-0 stdout
    # capture, already live-verified into temp_results_pinn/RESULTS_TABLE.md section 6.
    features_pct = sorted([
        ("windward_topex", 15.2), ("eastness", 14.2), ("slope_degrees", 12.8),
        ("gwa_weibull_k_50m", 12.1), ("elevation", 11.4), ("gwa_weibull_a_50m", 6.5),
        ("ceh_twi", 6.4), ("gwa_weibull_k_10m", 5.9), ("solar_radiation_index", 5.5),
        ("gwa_wind_speed_10m", 5.1), ("gwa_weibull_a_10m", 4.7),
    ], key=lambda x: x[1])
    features = [f for f, _ in features_pct]
    pct = [p for _, p in features_pct]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(features, pct, color=COLOR_PINN)
    ax.set_xlabel("Share of total importance (%)")
    ax.set_title("No single feature dominates PINN's personalised y_max")
    plt.tight_layout()
    out = FIGURES_DIR / "q3_pinn_terrain_importance.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 4 (Block D, appendix): one flagged plot's curve vs. population curve
# -----------------------------------------------------------------------------
def build_flagged_plot_example():
    EXAMPLE_PLOT_ID = 77226  # compartment 1129, most-inflated plain-PINN prediction (y_max_pred=77.75)

    master = pd.read_parquet("data/processed/master/clean_master_4survey.parquet")
    plot_rows = master[master["identification"] == EXAMPLE_PLOT_ID].sort_values("Age")

    preds = pd.read_csv("temp_results_pinn/outputs/example_curve/plain_pinn_fixed_test_set_predictions.csv")
    plot_pred = preds[preds["identification"] == EXAMPLE_PLOT_ID].iloc[0]
    plot_y_max = plot_pred["y_max_pred"]

    cr_params = load_cr_params("4survey", "spatial_block_kfold", split_seed=SPLIT_SEED, held_out_fold=0)
    age_range = np.linspace(0, 100, 200)
    population_curve = cr_params["y_max"] * (1 - np.exp(-cr_params["k"] * age_range)) ** cr_params["p"]
    # Plain PINN has no personalized k -- k stays at the population value, only y_max is personalized.
    plot_curve = plot_y_max * (1 - np.exp(-cr_params["k"] * age_range)) ** cr_params["p"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(plot_rows["Age"], plot_rows["elev_percentile_95th"], color="black", zorder=5, label="Observed (this plot)")
    ax.plot(age_range, population_curve, "--", color="gray",
            label=f"Population curve (y\\_max={cr_params['y_max']:.1f})")
    ax.plot(age_range, plot_curve, "-", color=COLOR_PINN,
            label=f"Plain PINN curve (y\\_max={plot_y_max:.1f})")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Top height (m)")
    ax.set_title(f"Flagged plot {EXAMPLE_PLOT_ID} (compartment 1129): an implausible ceiling")
    ax.legend()
    plt.tight_layout()
    out = FIGURES_DIR / "q3_pinn_implausible_example_plot.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 5 (Block E, appendix): y_max vs. k scatter
# -----------------------------------------------------------------------------
def build_ymax_k_scatter():
    df = pd.read_csv("temp_results_pinn/outputs/example_curve/test_set_predictions.csv")
    df = df[df["split"] == "test"].drop_duplicates("identification")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["y_max_pred"], df["k_pred"], s=4, alpha=0.15, color=COLOR_PINN_K)
    corr = np.corrcoef(df["y_max_pred"], df["k_pred"])[0, 1]
    ax.set_xlabel("Predicted y\\_max (m)")
    ax.set_ylabel("Predicted k")
    ax.set_title(f"PINN-$k$'s two parameters move together (r={corr:.2f})")
    plt.tight_layout()
    out = FIGURES_DIR / "q3_pinn_ymax_k_scatter.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    build_env_sweep_chart()
    build_compartment_map()
    build_importance_chart()
    build_flagged_plot_example()
    build_ymax_k_scatter()
    print("\nAll 5 figures built.")

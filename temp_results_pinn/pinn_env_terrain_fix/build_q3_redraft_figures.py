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


def load_pooled_compartment_deviation_data():
    """Return one held-out prediction per plot and compartment-level mean deviations."""
    master = pd.read_parquet("data/processed/master/clean_master_4survey.parquet")
    master_lookup = master[["identification", "blk", "cpmt"]].drop_duplicates("identification")

    fold0 = pd.read_csv("temp_results_pinn/outputs/example_curve/plain_pinn_fixed_test_set_predictions.csv")
    fold0 = fold0.drop_duplicates("identification").merge(master_lookup, on="identification", how="left")
    cr0 = load_cr_params("4survey", "spatial_block_kfold", split_seed=SPLIT_SEED, held_out_fold=0)
    fold0["deviation"] = fold0["y_max_pred"] - cr0["y_max"]

    other_folds = []
    for i in [1, 2, 3, 4]:
        df = pd.read_csv(
            f"temp_results_pinn/outputs/CORRECTED_2026-08-23_mechanism_checks/ymax_population_check/fold_{i}/predictions.csv"
        )
        df = df.drop_duplicates("identification")
        df["deviation"] = df["y_max_pred"] - df["population_y_max"]
        other_folds.append(df[["identification", "blk", "cpmt", "deviation"]])

    preds = pd.concat([fold0[["identification", "blk", "cpmt", "deviation"]]] + other_folds, ignore_index=True)
    n_before = len(preds)
    preds = preds.drop_duplicates("identification")
    print(
        f"Pooled 5-fold predictions: {n_before} rows -> {len(preds)} unique plots "
        "(should match, one held-out prediction per plot)"
    )

    compartment_summary = (
        preds.groupby(["blk", "cpmt"], as_index=False)
        .agg(
            mean_deviation=("deviation", "mean"),
            n_plots=("identification", "nunique"),
        )
    )
    return preds, compartment_summary


def add_map_colorbar(fig, ax, cmap, norm, label):
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(label)


def style_map_axes(ax, minx, miny, maxx, maxy, x_margin, y_margin):
    ax.set_xlim(minx - x_margin, maxx + x_margin)
    ax.set_ylim(miny - y_margin, maxy + y_margin)
    ax.set_xlabel("Easting (British National Grid, m)")
    ax.set_ylabel("Northing (British National Grid, m)")
    ax.ticklabel_format(style="plain", axis="both")
    ax.tick_params(axis="x", labelrotation=30)


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
    # Pool all 5 folds' TEST-set predictions -- each compartment is the test set in exactly
    # one of the 5 folds under spatial_block_kfold, so pooling gives every plot in the forest
    # exactly one genuine held-out prediction (same pattern Q1's own maps already use).
    _, compartment_summary = load_pooled_compartment_deviation_data()
    compartment_mean = compartment_summary.groupby("cpmt", as_index=False)["mean_deviation"].mean()
    compartment_mean = compartment_mean.rename(columns={"mean_deviation": "deviation"})

    boundaries = load_compartment_boundaries()
    mapped = boundaries.merge(compartment_mean, on="cpmt", how="left")

    shared_vmax = mapped["deviation"].abs().quantile(0.98)
    norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-shared_vmax, vmax=shared_vmax)

    # Zoom to the area that actually has data (not the full boundaries extent, which includes
    # far more of the forest than the filtered cohort covers) -- 5% margin around it.
    has_data = mapped[mapped["deviation"].notna()]
    minx, miny, maxx, maxy = has_data.total_bounds
    x_margin = (maxx - minx) * 0.08
    y_margin = (maxy - miny) * 0.08

    fig, ax = plt.subplots(figsize=(8, 7))
    mapped.plot(column="deviation", cmap=DIVERGING_CMAP, norm=norm, ax=ax,
                edgecolor="#cccccc", linewidth=0.2, missing_kwds={"color": "#f5f5f5"})
    style_map_axes(ax, minx, miny, maxx, maxy, x_margin, y_margin)
    add_map_colorbar(fig, ax, DIVERGING_CMAP, norm, "Plain PINN: mean y_max deviation from population curve (m)")

    # Label compartment 1129, the flagged hotspot -- centroid + a short offset arrow so the
    # label doesn't sit directly on top of the (already dark) compartment shape.
    cpmt1129 = boundaries[boundaries["cpmt"] == 1129]
    if len(cpmt1129) > 0:
        cx, cy = cpmt1129.geometry.centroid.iloc[0].coords[0]
        # Fixed label position, bottom-right of the zoomed map (empty area, no data there) --
        # not relative to the compartment's own centroid, which sits far to the right.
        label_x = maxx - x_margin * 0.3
        label_y = miny + y_margin * 0.3
        ax.annotate("Compartment 1129",
                    xy=(cx, cy), xytext=(label_x, label_y),
                    fontsize=8, ha="right",
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_title("Plain PINN: where the predicted ceiling ($y_{max}$) drifts from the population curve")
    plt.tight_layout()
    out = FIGURES_DIR / "q3_pinn_ymax_deviation_map.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 3 (Block D, appendix): hotspot-labeled companion map
# -----------------------------------------------------------------------------
def build_hotspot_label_map():
    _, compartment_summary = load_pooled_compartment_deviation_data()
    compartment_mean = compartment_summary.groupby("cpmt", as_index=False)["mean_deviation"].mean()
    compartment_mean = compartment_mean.rename(columns={"mean_deviation": "deviation"})

    boundaries = load_compartment_boundaries()
    mapped = boundaries.merge(compartment_mean, on="cpmt", how="left")
    has_data = mapped[mapped["deviation"].notna()]
    minx, miny, maxx, maxy = has_data.total_bounds
    x_margin = (maxx - minx) * 0.08
    y_margin = (maxy - miny) * 0.08

    shared_vmax = mapped["deviation"].abs().quantile(0.98)
    norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-shared_vmax, vmax=shared_vmax)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    mapped.plot(column="deviation", cmap=DIVERGING_CMAP, norm=norm, ax=ax,
                edgecolor="#cccccc", linewidth=0.2, missing_kwds={"color": "#f5f5f5"})
    style_map_axes(ax, minx, miny, maxx, maxy, x_margin, y_margin)
    add_map_colorbar(fig, ax, DIVERGING_CMAP, norm, "Plain PINN: mean y_max deviation from population curve (m)")

    labels = {
        2041: (-4500, 2200),
        2064: (2200, 1700),
        2057: (-4200, -1200),
        1130: (-2600, 1600),
        1129: (3200, -2200),
    }
    for cpmt, (dx, dy) in labels.items():
        row = boundaries[boundaries["cpmt"] == cpmt]
        if len(row) == 0:
            continue
        cx, cy = row.geometry.centroid.iloc[0].coords[0]
        n_plots = compartment_summary.loc[compartment_summary["cpmt"] == cpmt, "n_plots"]
        mean_dev = compartment_summary.loc[compartment_summary["cpmt"] == cpmt, "mean_deviation"]
        if n_plots.empty or mean_dev.empty:
            continue
        text = f"{cpmt} ({int(n_plots.iloc[0])})\n{mean_dev.iloc[0]:+.1f} m"
        ax.annotate(
            text,
            xy=(cx, cy),
            xytext=(cx + dx, cy + dy),
            fontsize=8,
            ha="center",
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#666666", alpha=0.9),
        )

    ax.set_title("Plain PINN: darkest red hotspot compartments on the pooled $y_{max}$ deviation map")
    plt.tight_layout()
    out = FIGURES_DIR / "q3_pinn_ymax_hotspot_summary.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 4 (Block D, appendix): PINN-k fold-0 deviation maps (current available data)
# -----------------------------------------------------------------------------
def build_pinn_k_deviation_maps_fold0():
    df = pd.read_csv("temp_results_pinn/outputs/example_curve/test_set_predictions.csv")
    df = df[df["split"] == "test"].drop_duplicates("identification").copy()
    cr0 = load_cr_params("4survey", "spatial_block_kfold", split_seed=SPLIT_SEED, held_out_fold=0)
    df["ymax_deviation"] = df["y_max_pred"] - cr0["y_max"]
    df["k_deviation"] = df["k_pred"] - cr0["k"]

    summary = (
        df.groupby("cpmt", as_index=False)
        .agg(
            ymax_deviation=("ymax_deviation", "mean"),
            k_deviation=("k_deviation", "mean"),
        )
    )

    boundaries = load_compartment_boundaries()
    mapped = boundaries.merge(summary, on="cpmt", how="left")
    has_data = mapped[mapped["ymax_deviation"].notna()]
    minx, miny, maxx, maxy = has_data.total_bounds
    x_margin = (maxx - minx) * 0.08
    y_margin = (maxy - miny) * 0.08

    ymax_vmax = mapped["ymax_deviation"].abs().quantile(0.98)
    k_vmax = mapped["k_deviation"].abs().quantile(0.98)
    ymax_norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-ymax_vmax, vmax=ymax_vmax)
    k_norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-k_vmax, vmax=k_vmax)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, column, norm, title, cbar_label in [
        (axes[0], "ymax_deviation", ymax_norm, "PINN-k fold 0: mean y_max deviation", "Mean y_max deviation (m)"),
        (axes[1], "k_deviation", k_norm, "PINN-k fold 0: mean k deviation", "Mean k deviation"),
    ]:
        mapped.plot(column=column, cmap=DIVERGING_CMAP, norm=norm, ax=ax,
                    edgecolor="#cccccc", linewidth=0.2, missing_kwds={"color": "#f5f5f5"})
        style_map_axes(ax, minx, miny, maxx, maxy, x_margin, y_margin)
        add_map_colorbar(fig, ax, DIVERGING_CMAP, norm, cbar_label)
        ax.set_title(title)

    plt.tight_layout()
    out = FIGURES_DIR / "q3_pinn_k_fold0_ymax_k_deviation_maps.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 5 (Block B, appendix): per-feature terrain importance bar chart
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
# Figure 6 (Block D, appendix): one flagged plot's curve vs. population curve
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
            label=f"Population curve ($y_{{max}}$={cr_params['y_max']:.1f})")
    ax.plot(age_range, plot_curve, "-", color=COLOR_PINN,
            label=f"Plain PINN curve ($y_{{max}}$={plot_y_max:.1f})")
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
# Figure 7 (Block E, appendix): y_max vs. k scatter
# -----------------------------------------------------------------------------
def build_ymax_k_scatter():
    df = pd.read_csv("temp_results_pinn/outputs/example_curve/test_set_predictions.csv")
    df = df[df["split"] == "test"].drop_duplicates("identification")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df["y_max_pred"], df["k_pred"], s=4, alpha=0.15, color=COLOR_PINN_K)
    corr = np.corrcoef(df["y_max_pred"], df["k_pred"])[0, 1]
    ax.set_xlabel("Predicted $y_{max}$ (m)")
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
    build_hotspot_label_map()
    build_pinn_k_deviation_maps_fold0()
    build_importance_chart()
    build_flagged_plot_example()
    build_ymax_k_scatter()
    print("\nAll 7 figures built.")

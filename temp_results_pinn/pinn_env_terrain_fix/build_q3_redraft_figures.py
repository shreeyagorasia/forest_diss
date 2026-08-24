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
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "fig_results" / "q3"
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
    # Grouped bar chart with real error bars -- SDs are the actual 5-fold spread for every
    # condition, all recomputed live from raw summary.json files (Set3's matches the ledger's
    # point estimate exactly: PINN 0.6308, PINN-k 0.6178). No-environment (2026-08-24): a TRUE
    # ablation -- same YMaxSubNetwork architecture as Set2/3/4, zero input columns -- not the
    # old pinn_noenv.py number, which used a structurally different network (see
    # RESULTS_TABLE.md section 5 correction). Lands close to the old single-run estimate
    # (0.573/0.573), within this run's own SD, despite the architecture difference.
    conditions = ["No\nenvironment", "Set2\n(small)", "Set3\n(medium,\nheadline)", "Set4\n(large)"]
    pinn_r2 = [0.5718, 0.6274, 0.6308, 0.6184]
    pinn_r2_err = [0.0323, 0.023, 0.0245, 0.029]
    pinn_k_r2 = [0.5749, 0.6223, 0.6178, 0.6196]
    pinn_k_r2_err = [0.0326, 0.017, 0.0204, 0.020]

    x = np.arange(len(conditions))
    bar_width = 0.35

    # Lighter, more clearly-green pastel shades, just for this chart -- the shared PINN/PINN-k
    # colors are a blue-leaning teal, and simply blending them toward white kept that blue
    # undertone (looked too blue, and the two shades read as too similar once lightened evenly).
    # These are picked by hand instead: PINN stays a light mint/sage, PINN-k is a genuinely
    # different green (olive, more yellow-leaning) rather than just a darker/lighter version of
    # the same hue -- both still light enough that the dark error-bar whiskers stay clearly
    # visible. Doesn't touch the shared style file, so every other figure keeps the original
    # shades.
    # User-picked pair: a light lime green for PINN, a darker teal-emerald for PINN-k -- clearly
    # different hues (not just light/dark steps of the same green) while both staying light
    # enough for the dark error-bar whiskers to show clearly.
    color_pinn_light = "#B0E892"
    color_pinn_k_light = "#027C68"

    # Dot-and-whisker (point-range) plot, not bars -- 2026-08-24. A bar's LENGTH is what a
    # reader perceives as the value, so truncating the baseline distorts the ratio between bars
    # (a small real difference can look huge). A dot's POSITION is what's read instead -- there
    # is no length to distort, so zooming the axis to the data's actual range is honest here,
    # not misleading, as long as the axis ticks stay visible (they do). Range chosen to fully
    # show every whisker, including no-environment's wide one (min low=0.5395, max high=0.6553
    # across all 8 points) -- not cropped tighter than that.
    fig, ax = plt.subplots(figsize=(7, 4.5))

    offset = 0.11
    ax.errorbar(
        x - offset, pinn_r2, yerr=pinn_r2_err, fmt="o", markersize=9, capsize=4,
        color=color_pinn_light, markeredgecolor="#333333", markeredgewidth=1.2,
        ecolor="#333333", elinewidth=1.4, label="PINN",
    )
    ax.errorbar(
        x + offset, pinn_k_r2, yerr=pinn_k_r2_err, fmt="o", markersize=9, capsize=4,
        color=color_pinn_k_light, markeredgecolor="#333333", markeredgewidth=1.2,
        ecolor="#333333", elinewidth=1.4, label="PINN-$k$",
    )

    # Highlight the headline condition (Set3) with a shaded vertical band behind it, so it reads
    # as "this is the one used everywhere else in the chapter" without a separate legend entry.
    headline_index = 2
    ax.axvspan(headline_index - 0.5, headline_index + 0.5, color="#f0f0f0", zorder=0)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_xlim(-0.5, len(conditions) - 0.5)
    ax.set_ylabel("Test $R^2$ (mean $\\pm$ SD, 5-fold)")
    ax.set_ylim(0.52, 0.68)
    ax.legend(frameon=False, loc="upper left")
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
                edgecolor="#cccccc", linewidth=0.2, missing_kwds={"color": "#f0f0f0", "hatch": "///", "edgecolor": "#bbbbbb", "linewidth": 0.3})
    style_map_axes(ax, minx, miny, maxx, maxy, x_margin, y_margin)
    add_map_colorbar(fig, ax, DIVERGING_CMAP, norm, "Plain PINN: mean y_max deviation from population curve (m)")

    # Label the two flagship compartments (rule-picked, see RESULTS_TABLE.md and the 2c
    # rewrite): 1129 (over-productive, single most extreme plot) and 2021 (under-productive,
    # top by both mean deviation and single most extreme plot). Fixed label positions in empty
    # map corners, not relative to each compartment's own centroid -- avoids the label sitting
    # on top of other dark compartments, which happened when positions were centroid-relative.
    flagship_labels = {
        1129: {"text": "Compartment 1129\n(over-productive)", "corner": (maxx - x_margin * 0.3, miny + y_margin * 0.3), "ha": "right"},
        2021: {"text": "Compartment 2021\n(under-productive)", "corner": (minx + x_margin * 0.3, maxy - y_margin * 0.3), "ha": "left"},
    }
    for cpmt, label_info in flagship_labels.items():
        cpmt_row = boundaries[boundaries["cpmt"] == cpmt]
        if len(cpmt_row) == 0:
            continue
        cx, cy = cpmt_row.geometry.centroid.iloc[0].coords[0]
        label_x, label_y = label_info["corner"]
        # Semi-transparent white box behind the text -- 2021's label sits over busy terrain
        # outlines near the map edge and was hard to read against them without one.
        ax.annotate(label_info["text"],
                    xy=(cx, cy), xytext=(label_x, label_y),
                    fontsize=8, ha=label_info["ha"],
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.75),
                    arrowprops=dict(arrowstyle="->", color="black", lw=0.8))

    ax.set_title("Plain PINN: where the predicted ceiling ($y_{max}$) drifts from the population curve")
    plt.tight_layout()
    out = FIGURES_DIR / "q3_pinn_ymax_deviation_map.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 3 (Block D, appendix): hotspot-labeled companion maps -- plain PINN (y_max) and
# PINN-k (k), side by side
# -----------------------------------------------------------------------------
def _draw_hotspot_panel(fig, ax, mapped, value_col, over_top3, under_top3, unit_fmt, unit_suffix, cbar_label, title):
    """Draws one hotspot panel: choropleth + top-3-per-side numbered markers + legend box +
    a colorbar pulled in tighter (smaller, closer to the map) than the shared add_map_colorbar
    helper -- requested for this figure specifically, other maps keep the wider default."""
    has_data = mapped[mapped[value_col].notna()]
    minx, miny, maxx, maxy = has_data.total_bounds
    x_margin = (maxx - minx) * 0.08
    y_margin = (maxy - miny) * 0.08

    vmax = mapped[value_col].abs().quantile(0.98)
    norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    mapped.plot(column=value_col, cmap=DIVERGING_CMAP, norm=norm, ax=ax,
                edgecolor="#cccccc", linewidth=0.2, missing_kwds={"color": "#f0f0f0", "hatch": "///", "edgecolor": "#bbbbbb", "linewidth": 0.3})
    style_map_axes(ax, minx, miny, maxx, maxy, x_margin, y_margin)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.06)
    sm = plt.cm.ScalarMappable(cmap=DIVERGING_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    legend_lines = []
    for code_prefix, top3, edge_color in [("O", over_top3, "#b2182b"), ("U", under_top3, "#2166ac")]:
        for rank, cpmt in enumerate(top3, start=1):
            code = f"{code_prefix}{rank}"
            row = mapped[mapped["cpmt"] == cpmt]
            if len(row) == 0 or row[value_col].isna().all():
                continue
            cx, cy = row.geometry.centroid.iloc[0].coords[0]
            # Clip label position to stay inside the zoomed map bounds -- keeps markers off the
            # title, axis ticks, and colorbar even for compartments near the map's edge.
            cx_clipped = min(max(cx, minx + x_margin * 0.4), maxx - x_margin * 0.4)
            cy_clipped = min(max(cy, miny + y_margin * 0.4), maxy - y_margin * 0.6)
            ax.annotate(code, xy=(cx_clipped, cy_clipped), fontsize=8, fontweight="bold",
                        ha="center", va="center", color="black",
                        bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=edge_color, lw=1.2))
            val = row[value_col].iloc[0]
            n = int(row["n_plots"].iloc[0]) if "n_plots" in row.columns else 0
            legend_lines.append(f"{code}=cpmt {cpmt}  {val:{unit_fmt}}{unit_suffix}  (n={n})")

    legend_text = "\n".join(legend_lines)
    ax.text(
        0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=6.5,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#666666", alpha=0.92),
    )
    ax.set_title(title, fontsize=10)


def build_hotspot_label_map():
    # Two-panel version, 2026-08-24: plain PINN's y_max hotspots (left) and PINN-k's k
    # hotspots (right), both under the n>=50 reliability rule, both directions labelled.
    # Colorbar pulled in tighter per request. Notably, the two models' top-3 lists overlap
    # heavily (2057 tops both; the under-side top-3 is the exact same 3 compartments, just a
    # different order) -- a real cross-model convergence, not staged.
    _, plain_summary = load_pooled_compartment_deviation_data()
    plain_summary = plain_summary.groupby("cpmt", as_index=False).agg(
        deviation=("mean_deviation", "mean"), n_plots=("n_plots", "sum"))
    boundaries = load_compartment_boundaries()
    plain_mapped = boundaries.merge(plain_summary, on="cpmt", how="left")

    _, k_summary = load_pooled_pinn_k_deviation_data()
    k_summary = k_summary.groupby("cpmt", as_index=False).agg(
        k_deviation=("k_deviation", "mean"), n_plots=("n_plots", "sum"))
    k_mapped = boundaries.merge(k_summary, on="cpmt", how="left")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))

    _draw_hotspot_panel(
        fig, axes[0], plain_mapped, "deviation",
        over_top3=[2057, 1130, 2130], under_top3=[2021, 1070, 2094],
        unit_fmt="+.1f", unit_suffix="m", cbar_label="Plain PINN: mean $y_{max}$ deviation (m)",
        title="Plain PINN: top-3 over/under-productive compartments (n$\\geq$50)",
    )
    _draw_hotspot_panel(
        fig, axes[1], k_mapped, "k_deviation",
        over_top3=[2057, 1027, 2163], under_top3=[2094, 2021, 1070],
        unit_fmt="+.4f", unit_suffix="", cbar_label="PINN-$k$: mean $k$ deviation",
        title="PINN-$k$: top-3 over/under-productive compartments (n$\\geq$50)",
    )

    plt.tight_layout()
    out = FIGURES_DIR / "q3_pinn_ymax_hotspot_summary.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 4 (Block D, main text): PINN-k deviation maps, pooled across all 5 folds
# -----------------------------------------------------------------------------
def load_pooled_pinn_k_deviation_data():
    """Pool all 5 folds' PINN-k test-set predictions -- same one-plot-one-prediction logic as
    load_pooled_compartment_deviation_data(), but for both y_max and k. Rebuilt 2026-08-24 to
    replace the fold-0-only version: pooling changed the answer (1129 does not hold up as
    PINN-k's k-flagship once pooled -- 2057 does, see RESULTS_TABLE.md section 13)."""
    master = pd.read_parquet("data/processed/master/clean_master_4survey.parquet")
    master_lookup = master[["identification", "blk", "cpmt"]].drop_duplicates("identification")

    fold0 = pd.read_csv("temp_results_pinn/outputs/example_curve/test_set_predictions.csv")
    fold0 = fold0[fold0["split"] == "test"].drop_duplicates("identification").merge(
        master_lookup, on="identification", how="left", suffixes=("", "_m"))
    cr0 = load_cr_params("4survey", "spatial_block_kfold", split_seed=SPLIT_SEED, held_out_fold=0)
    fold0["ymax_deviation"] = fold0["y_max_pred"] - cr0["y_max"]
    fold0["k_deviation"] = fold0["k_pred"] - cr0["k"]
    fold0 = fold0[["identification", "blk", "cpmt", "ymax_deviation", "k_deviation"]]

    other_folds = []
    for i in [1, 2, 3, 4]:
        df = pd.read_csv(
            f"temp_results_pinn/outputs/CORRECTED_2026-08-23_mechanism_checks/pinn_k_population_check/fold_{i}/predictions.csv"
        )
        df = df.drop_duplicates("identification")
        df["ymax_deviation"] = df["y_max_pred"] - df["population_y_max"]
        df["k_deviation"] = df["k_pred"] - df["population_k"]
        other_folds.append(df[["identification", "blk", "cpmt", "ymax_deviation", "k_deviation"]])

    preds = pd.concat([fold0] + other_folds, ignore_index=True)
    n_before = len(preds)
    preds = preds.drop_duplicates("identification")
    print(f"Pooled PINN-k 5-fold predictions: {n_before} rows -> {len(preds)} unique plots")

    compartment_summary = (
        preds.groupby(["blk", "cpmt"], as_index=False)
        .agg(
            ymax_deviation=("ymax_deviation", "mean"),
            k_deviation=("k_deviation", "mean"),
            n_plots=("identification", "nunique"),
        )
    )
    return preds, compartment_summary


def build_pinn_k_deviation_maps_pooled():
    _, compartment_summary = load_pooled_pinn_k_deviation_data()
    summary = compartment_summary.groupby("cpmt", as_index=False).agg(
        ymax_deviation=("ymax_deviation", "mean"), k_deviation=("k_deviation", "mean"))

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
        (axes[0], "ymax_deviation", ymax_norm, "PINN-$k$, pooled 5-fold: mean $y_{max}$ deviation", "Mean $y_{max}$ deviation (m)"),
        (axes[1], "k_deviation", k_norm, "PINN-$k$, pooled 5-fold: mean $k$ deviation", "Mean $k$ deviation"),
    ]:
        mapped.plot(column=column, cmap=DIVERGING_CMAP, norm=norm, ax=ax,
                    edgecolor="#cccccc", linewidth=0.2, missing_kwds={"color": "#f0f0f0", "hatch": "///", "edgecolor": "#bbbbbb", "linewidth": 0.3})
        style_map_axes(ax, minx, miny, maxx, maxy, x_margin, y_margin)
        add_map_colorbar(fig, ax, DIVERGING_CMAP, norm, cbar_label)
        ax.set_title(title)

    # Same 4 compartments labelled on BOTH panels, not different ones per side -- lets a reader
    # directly compare how each compartment's color changes left vs. right. Picked for variety,
    # not just "biggest magnitude" (which would just repeat 2057 four times): each shows a
    # different routing pattern. See RESULTS_TABLE.md section 13 for the underlying numbers.
    highlight_compartments = {
        1045: "A: biggest $y_{max}$ push, modest $k$ -- shows via ceiling, not growth rate",
        2057: "B: biggest $k$ push, modest $y_{max}$ -- shows via growth rate, not ceiling",
        1070: "C: biggest NEGATIVE $y_{max}$ push, $k$~0 -- under-productive, via ceiling only",
        2021: "D: plain PINN's own flagship -- looks unremarkable in PINN-$k$ (modest both)",
    }
    label_letters = {1045: "A", 2057: "B", 1070: "C", 2021: "D"}
    # Fixed corner offsets, chosen to match each compartment's actual side of the map (A/C sit
    # east of centre, B/D sit west of centre) -- an earlier version sent labels to corners by
    # rotation, not by actual position, which made the arrows cross each other. This way each
    # arrow is short and stays on its own side.
    corner_offsets = {
        1045: (0.92, 0.92),  # A -- east, north
        1070: (0.92, 0.08),  # C -- east, south (close to A's real position, same side)
        2057: (0.05, 0.92),  # B -- west, north
        2021: (0.05, 0.08),  # D -- west, south (close to B's real position, same side)
    }
    for ax in axes:
        for cpmt, letter in label_letters.items():
            row = boundaries[boundaries["cpmt"] == cpmt]
            if len(row) == 0:
                continue
            cx, cy = row.geometry.centroid.iloc[0].coords[0]
            frac_x, frac_y = corner_offsets[cpmt]
            label_x = minx + (maxx - minx) * frac_x
            label_y = miny + (maxy - miny) * frac_y
            ax.annotate(letter, xy=(cx, cy), xytext=(label_x, label_y),
                        fontsize=12, fontweight="bold", ha="center", va="center",
                        color="black",
                        bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec="#333333", lw=1.4),
                        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.9))

    legend_text = "\n".join(f"{v}" for v in highlight_compartments.values())
    # Reserve real space in the figure canvas for the legend (subplots_adjust), rather than
    # relying on bbox_inches="tight" to auto-pad around fig.text -- that auto-padding was
    # producing a much bigger gap than intended between the map and the text.
    # Fontsize bumped 11 -> 15, 2026-08-24 -- this figure is now printed noticeably larger in
    # the dissertation (1.15x textwidth, was 0.9x), and the legend text was reading small
    # relative to the rest of the page at the old size. Reserved rect height increased slightly
    # (0.07 -> 0.09) to keep the bigger text from crowding the map above it.
    plt.tight_layout(rect=[0, 0.09, 1, 1])
    fig.text(0.5, 0.045, legend_text, fontsize=15, ha="center", va="center")

    out = FIGURES_DIR / "q3_pinn_k_pooled_ymax_k_deviation_maps.png"
    # bbox_inches="tight" -- otherwise the legend text placed below the axes (fig.text at
    # negative y) gets clipped off in the saved file.
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 5 (Block B, appendix): per-feature terrain importance bar chart
# -----------------------------------------------------------------------------
def build_importance_chart():
    # Fold 0's permutation-importance run predates the JSON-saving fix added later this
    # session (only folds 1/2 have saved files) -- these numbers are the fold-0 stdout
    # capture, already live-verified into temp_results_pinn/RESULTS_TABLE.md section 6.
    #
    # Human-readable names for the y-axis, 2026-08-24 -- definitions pulled from
    # documentation/variable_registry_av1_av2.csv so they match the dissertation's own
    # variable descriptions, not invented. Raw column name is kept underneath each label
    # (smaller, grey) so the figure is still traceable back to the actual data column.
    READABLE_NAME = {
        "windward_topex": "Windward exposure (SW)",
        "eastness": "Eastness (aspect)",
        "slope_degrees": "Slope",
        "gwa_weibull_k_50m": "Wind variability (50m)",
        "elevation": "Elevation",
        "gwa_weibull_a_50m": "Wind speed scale (50m)",
        "ceh_twi": "Topographic wetness (TWI)",
        "gwa_weibull_k_10m": "Wind variability (10m)",
        "solar_radiation_index": "Solar radiation",
        "gwa_wind_speed_10m": "Wind speed (10m)",
        "gwa_weibull_a_10m": "Wind speed scale (10m)",
    }
    features_pct = sorted([
        ("windward_topex", 15.2), ("eastness", 14.2), ("slope_degrees", 12.8),
        ("gwa_weibull_k_50m", 12.1), ("elevation", 11.4), ("gwa_weibull_a_50m", 6.5),
        ("ceh_twi", 6.4), ("gwa_weibull_k_10m", 5.9), ("solar_radiation_index", 5.5),
        ("gwa_wind_speed_10m", 5.1), ("gwa_weibull_a_10m", 4.7),
    ], key=lambda x: x[1])
    features = [f for f, _ in features_pct]
    pct = [p for _, p in features_pct]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(features, pct, color=COLOR_PINN)
    ax.set_xlabel("Share of total importance (%)")
    # Small grey subtitle under the x-axis label, 2026-08-24: clarifies the METHOD -- this is
    # permutation importance (shuffle each feature, measure how much y_max_pred moves), not
    # VIF (a multicollinearity diagnostic, not an importance measure) or SHAP.
    ax.text(
        0.5, -0.16, "Permutation importance (10 shuffle repeats/feature)",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="grey",
    )
    ax.set_title("No single feature dominates PINN's personalised y_max")

    # Two-tier y-tick labels (readable name + raw column name below it, smaller/grey) can't
    # be done with a single set_yticklabels call -- matplotlib tick labels are one font/color
    # per tick. So: blank out the real tick labels, then place both lines by hand at each
    # bar's y-position, right-aligned just outside the axes (works with any renderer/backend,
    # unlike relying on tick label padding).
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    for i, raw_name in enumerate(features):
        ax.annotate(
            READABLE_NAME[raw_name],
            xy=(0, i), xycoords=("axes fraction", "data"),
            xytext=(-6, 6), textcoords="offset points",
            ha="right", va="center", fontsize=10, color="black",
        )
        ax.annotate(
            raw_name,
            xy=(0, i), xycoords=("axes fraction", "data"),
            xytext=(-6, -6), textcoords="offset points",
            ha="right", va="center", fontsize=7, color="grey",
        )

    plt.tight_layout()
    out = FIGURES_DIR / "q3_pinn_terrain_importance.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 6 (Block D, appendix): one flagged plot's curve vs. population curve
# -----------------------------------------------------------------------------
def build_flagged_plot_example():
    # Adds PINN-k's curve alongside plain PINN's, 2026-08-24 -- shows the parameter-routing
    # finding at the single-plot level, not just the compartment level. Same plot (77226),
    # same population baseline, three curves: plain PINN pushes the CEILING to an implausible
    # 77.75m; PINN-k keeps the ceiling modest (52.97m, close to population) but personalises
    # GROWTH RATE instead (k=0.0222, the dataset's single highest k value) -- reaches a
    # plausible height faster, not higher.
    EXAMPLE_PLOT_ID = 77226  # compartment 1129, most-inflated plain-PINN prediction (y_max_pred=77.75)

    master = pd.read_parquet("data/processed/master/clean_master_4survey.parquet")
    plot_rows = master[master["identification"] == EXAMPLE_PLOT_ID].sort_values("Age")

    preds = pd.read_csv("temp_results_pinn/outputs/example_curve/plain_pinn_fixed_test_set_predictions.csv")
    plot_pred = preds[preds["identification"] == EXAMPLE_PLOT_ID].iloc[0]
    plot_y_max = plot_pred["y_max_pred"]

    k_preds = pd.read_csv("temp_results_pinn/outputs/example_curve/test_set_predictions.csv")
    k_preds = k_preds[k_preds["split"] == "test"].drop_duplicates("identification")
    k_plot_pred = k_preds[k_preds["identification"] == EXAMPLE_PLOT_ID].iloc[0]
    plot_k_ymax = k_plot_pred["y_max_pred"]
    plot_k_k = k_plot_pred["k_pred"]

    cr_params = load_cr_params("4survey", "spatial_block_kfold", split_seed=SPLIT_SEED, held_out_fold=0)
    age_range = np.linspace(0, 100, 200)
    population_curve = cr_params["y_max"] * (1 - np.exp(-cr_params["k"] * age_range)) ** cr_params["p"]
    # Plain PINN has no personalized k -- k stays at the population value, only y_max is personalized.
    plot_curve = plot_y_max * (1 - np.exp(-cr_params["k"] * age_range)) ** cr_params["p"]
    # PINN-k personalizes both -- uses its OWN k, not the population k.
    plot_k_curve = plot_k_ymax * (1 - np.exp(-plot_k_k * age_range)) ** cr_params["p"]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(plot_rows["Age"], plot_rows["elev_percentile_95th"], color="black", zorder=5, label="Observed (this plot)")
    ax.plot(age_range, population_curve, "--", color="gray",
            label=f"Population curve ($y_{{max}}$={cr_params['y_max']:.1f}, $k$={cr_params['k']:.4f})")
    ax.plot(age_range, plot_curve, "-", color=COLOR_PINN,
            label=f"Plain PINN curve ($y_{{max}}$={plot_y_max:.1f}, $k$={cr_params['k']:.4f} fixed)")
    ax.plot(age_range, plot_k_curve, "-", color=COLOR_PINN_K,
            label=f"PINN-$k$ curve ($y_{{max}}$={plot_k_ymax:.1f}, $k$={plot_k_k:.4f})")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Top height (m)")
    ax.set_title(f"Flagged plot {EXAMPLE_PLOT_ID}: one plot, two different personalised ceilings")
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
    # Rebuilt 2026-08-24 on pooled DEVIATION data, not raw pooled predicted values. Checked:
    # each fold refits its own population y_max/k baseline, so pooling RAW y_max_pred/k_pred
    # across folds mixes in between-fold baseline drift and dilutes the real relationship (a
    # Simpson's-paradox-style confound) -- naive raw pooling gave r=0.12, while every individual
    # fold's own within-fold correlation was 0.26-0.75 (fold 0 alone: 0.50). Pooling DEVIATIONS
    # from each fold's own baseline removes that confound and gives r=0.43, the statistically
    # correct pooled number here -- unlike this chapter's other figures, where pooling the raw
    # values was fine (no fold-varying baseline to confound).
    preds, _ = load_pooled_pinn_k_deviation_data()

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(preds["ymax_deviation"], preds["k_deviation"], s=4, alpha=0.1, color=COLOR_PINN_K)
    corr = np.corrcoef(preds["ymax_deviation"], preds["k_deviation"])[0, 1]
    ax.axhline(0, color="#333333", linewidth=0.6, linestyle="--")
    ax.axvline(0, color="#333333", linewidth=0.6, linestyle="--")
    ax.set_xlabel("$y_{max}$ deviation from population curve (m)")
    ax.set_ylabel("$k$ deviation from population curve")
    ax.set_title(f"PINN-$k$'s two parameters move together (r={corr:.2f}, pooled 5-fold)")
    plt.tight_layout()
    out = FIGURES_DIR / "q3_pinn_ymax_k_scatter.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 8 (Block D, main text): plain PINN vs. PINN-k deviation distribution, pooled 5-fold
# -----------------------------------------------------------------------------
def build_plausibility_distribution_chart():
    # Item 4: visualises Table tab:plausibility-comparison (SD 7.27m vs 1.36m, pooled 5-fold,
    # recomputed 2026-08-24 -- the original table was fold-0-only). Same x-axis, overlaid --
    # PINN-k's near-spike shape next to plain PINN's much wider spread IS the point being made,
    # not a scaling problem to fix.
    plain_preds, _ = load_pooled_compartment_deviation_data()
    k_preds, _ = load_pooled_pinn_k_deviation_data()

    plain_dev = plain_preds["deviation"]
    k_dev = k_preds["ymax_deviation"]

    # Narrower/taller than before (was 10x3.5), 2026-08-24 -- this sits side by side with
    # fig:implausible-example (build_flagged_plot_example, figsize 7x5) in an equal-width
    # subfigure pair (0.48/0.48 textwidth). Matching this figure's aspect ratio (7/5=1.4) to
    # that one means both render at the same height once scaled to the same column width --
    # if either figure's figsize or the pair's subfigure widths change later, this needs
    # recalculating to keep matching.
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(-35, 30, 90)
    ax.hist(plain_dev, bins=bins, density=True, color="#A8D5BA", alpha=0.75,
            label=f"Plain PINN (SD={plain_dev.std():.1f}m)")
    ax.hist(k_dev, bins=bins, density=True, color="#027C68", alpha=0.75,
            label=f"PINN-$k$ (SD={k_dev.std():.1f}m)")
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("$y_{max}$ deviation from population curve (m)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.set_title("PINN-$k$'s ceiling stays tight -- plain PINN's does not")
    plt.tight_layout()
    out = FIGURES_DIR / "q3_plausibility_distribution.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 9 (Block D, appendix): plain PINN y_max deviation vs. independent yield class
# -----------------------------------------------------------------------------
def build_yieldclass_scatter():
    # Rebuilt 2026-08-24 on pooled data (58,073 plots), not fold-0-only (11,508) -- no
    # fold-varying-baseline confound here (yield class doesn't move fold to fold the way a
    # population y_max/k anchor does), so raw pooling is the correct, more robust choice, unlike
    # the y_max-vs-k scatter above. Both flagships (1129, 2021) now visible since pooling covers
    # every compartment, not just fold 0's.
    preds, _ = load_pooled_compartment_deviation_data()

    master = pd.read_parquet("data/processed/master/clean_master_4survey.parquet")
    yldc_lookup = master[["identification", "yldc"]].drop_duplicates("identification")
    df = preds.merge(yldc_lookup, on="identification", how="left").dropna(subset=["yldc"])
    corr = np.corrcoef(df["deviation"], df["yldc"])[0, 1]

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(df["yldc"], df["deviation"], s=4, alpha=0.08, color=COLOR_PINN)

    for cpmt, label, color in [(1129, "Compartment 1129", "#b2182b"), (2021, "Compartment 2021", "#2166ac")]:
        sub = df[df["cpmt"] == cpmt]
        if len(sub) > 0:
            ax.scatter(sub["yldc"], sub["deviation"], s=8, alpha=0.6, color=color, label=label)

    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Yield class (Forestry Commission, independent)")
    ax.set_ylabel("Plain PINN $y_{max}$ deviation (m)")
    ax.legend(frameon=False, markerscale=2, loc="upper left")
    ax.set_title(f"Yield class vs. personalised ceiling (r={corr:.2f}, pooled 5-fold)")
    plt.tight_layout()
    out = FIGURES_DIR / "q3_yieldclass_scatter.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 10 (Block D, appendix): master reference map -- every named compartment, one place
# -----------------------------------------------------------------------------
def build_all_compartments_reference_map():
    # Built 2026-08-24, requested as a single lookup source: every compartment named anywhere
    # in the chapter, numbered on one map, with a legend giving the reason each was highlighted
    # and where else it appears. Two gaps this closes: 1122/1069/2022 were named in
    # Table~\ref{tab:productivity-top5} but never shown on any map; 2163 was shown on the
    # hotspot map (fig:hotspot-summary) but never named in any text or table. Retired picks
    # (2031, 2142, 2229, 2219) are deliberately left off -- they're superseded, not part of the
    # argument.
    entries = [
        (1, 1129, "Flagship: single most extreme plot (over)", "Table tab:productivity-flagships; Fig fig:ymax-deviation-map, fig:implausible-example"),
        (2, 2021, "Flagship: top by mean AND most extreme plot (under)", "Table tab:productivity-flagships; Fig fig:ymax-deviation-map, fig:ymax-k-deviation-maps"),
        (3, 2057, "Yield-class mismatch; also PINN-k's own k-flagship", "Table tab:productivity-top5; Fig fig:hotspot-summary, fig:ymax-k-deviation-maps"),
        (4, 1027, "Yield-class mismatch (old, wind-exposed)", "Table tab:productivity-top5; Fig fig:hotspot-summary (PINN-k panel)"),
        (5, 1130, "Supporting over-productive, agrees with yield class", "Table tab:productivity-top5; Fig fig:hotspot-summary (plain PINN panel)"),
        (6, 2130, "Supporting over-productive, agrees with yield class", "Table tab:productivity-top5; Fig fig:hotspot-summary (plain PINN panel)"),
        (7, 1122, "Supporting over-productive, agrees with yield class", "Table tab:productivity-top5 only"),
        (8, 2163, "Supporting over-productive (PINN-k ranking)", "Fig fig:hotspot-summary (PINN-k panel) only"),
        (9, 1070, "Supporting under-productive, agrees with yield class", "Table tab:productivity-top5; Fig fig:hotspot-summary, fig:ymax-k-deviation-maps"),
        (10, 2094, "Supporting under-productive, agrees with yield class", "Table tab:productivity-top5; Fig fig:hotspot-summary"),
        (11, 1069, "Supporting under-productive, agrees with yield class", "Table tab:productivity-top5 only"),
        (12, 2022, "Supporting under-productive, weak agreement", "Table tab:productivity-top5 only"),
        (13, 1045, "Routing example: biggest y_max push, ordinary k", "Fig fig:ymax-k-deviation-maps only"),
    ]
    over_numbers = {1, 3, 4, 5, 6, 7, 8, 13}

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

    fig, ax = plt.subplots(figsize=(9, 9.5))
    mapped.plot(column="deviation", cmap=DIVERGING_CMAP, norm=norm, ax=ax,
                edgecolor="#cccccc", linewidth=0.2,
                missing_kwds={"color": "#f0f0f0", "hatch": "///", "edgecolor": "#bbbbbb", "linewidth": 0.3})
    style_map_axes(ax, minx, miny, maxx, maxy, x_margin, y_margin)
    add_map_colorbar(fig, ax, DIVERGING_CMAP, norm, "Plain PINN: mean $y_{max}$ deviation (m)")

    legend_lines = []
    for number, cpmt, reason, where in entries:
        row = boundaries[boundaries["cpmt"] == cpmt]
        if len(row) == 0:
            continue
        cx, cy = row.geometry.centroid.iloc[0].coords[0]
        edge_color = "#b2182b" if number in over_numbers else "#2166ac"
        # Slightly transparent circle fill (alpha on facecolor) so the map colour underneath
        # still shows through a little; smaller font, small pad so the digit sits centred
        # instead of looking pushed to one side of the circle.
        ax.annotate(str(number), xy=(cx, cy), fontsize=6.5, fontweight="bold",
                    ha="center", va="center", color="black",
                    bbox=dict(boxstyle="circle,pad=0.12", fc=(1, 1, 1, 0.75), ec=edge_color, lw=1.1))
        legend_lines.append(f"{number}. cpmt {cpmt} -- {reason}")

    # Legend underneath the map, single column -- a two-column layout at this font size kept
    # overlapping (lines too long for the column width).
    legend_text = "\n".join(legend_lines)
    plt.tight_layout(rect=[0, 0.28, 1, 1])
    fig.text(0.08, 0.23, legend_text, fontsize=10, va="top", ha="left")

    ax.set_title("Every named compartment in this chapter, in one place")
    out = FIGURES_DIR / "q3_all_compartments_reference_map.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")


# -----------------------------------------------------------------------------
# Figure 11 (Block D, appendix): the "good" example plot -- both models' curves
# -----------------------------------------------------------------------------
def build_good_example_plot_curve():
    # Updates the existing q3_pinn_example_plot_curve.png (built by run_example_plot_curve.py,
    # PINN-k only) to also show plain PINN's curve -- same treatment as the 77226 figure
    # (build_flagged_plot_example). This plot (119937) was picked by that script's own rule:
    # moderate deviation from the population curve, PINN-k's personalisation genuinely reduces
    # error (not just moves it), not an extreme/implausible case -- the "good" counterpart to
    # 77226's "implausible" case.
    import json
    EXAMPLE_PLOT_ID = 119937
    HEIGHT_COLUMN = "elev_percentile_95th"

    summary = json.load(open("temp_results_pinn/outputs/example_curve/example_plot_summary.json"))
    cr_params = load_cr_params("4survey", "spatial_block_kfold", split_seed=SPLIT_SEED, held_out_fold=0)

    plain_preds = pd.read_csv("temp_results_pinn/outputs/example_curve/plain_pinn_fixed_test_set_predictions.csv")
    plain_preds = plain_preds.drop_duplicates("identification")
    plain_row = plain_preds[plain_preds["identification"] == EXAMPLE_PLOT_ID].iloc[0]
    plain_y_max = plain_row["y_max_pred"]

    k_y_max = summary["plot_y_max"]
    k_k = summary["plot_k"]

    age_range = np.linspace(0, 100, 200)  # fixed range, matches the 77226 figure for comparability
    population_curve = cr_params["y_max"] * (1 - np.exp(-cr_params["k"] * age_range)) ** cr_params["p"]
    plain_curve = plain_y_max * (1 - np.exp(-cr_params["k"] * age_range)) ** cr_params["p"]
    k_curve = k_y_max * (1 - np.exp(-k_k * age_range)) ** cr_params["p"]

    # Narrower canvas than before (was 7x5), 2026-08-24 -- this plot sits next to the wider
    # terrain-importance chart in a side-by-side pair (0.58/0.38 textwidth split, see
    # results_q3_23_08_0540am.tex), and doesn't need the extra width to show the same three
    # curves clearly. Font sizes below are left at their normal absolute point sizes (not
    # scaled down to fit) so the text stays just as readable on the narrower canvas.
    # Height 5.87 (not 5), 2026-08-24: matches (a)'s rendered height once both are scaled to
    # their subfigure widths -- (a) is 6.5x5 at 0.58 textwidth (rendered height 0.446 textwidth);
    # for (b) at 0.38 textwidth to come out the same rendered height, its aspect ratio needs to
    # be 0.38/0.446 = 0.852 (width/height), i.e. 5/5.87. If either subfigure's textwidth
    # fraction or the OTHER figure's figsize changes later, this needs recalculating to match.
    fig, ax = plt.subplots(figsize=(5, 5.87))
    ax.scatter(summary["observed_age"], summary["observed_height"], color="black", zorder=5, label="Observed (this plot)")
    ax.plot(age_range, population_curve, "--", color="gray",
            label=f"Population curve ($y_{{max}}$={cr_params['y_max']:.1f}, $k$={cr_params['k']:.4f})")
    ax.plot(age_range, plain_curve, "-", color=COLOR_PINN,
            label=f"Plain PINN curve ($y_{{max}}$={plain_y_max:.1f}, $k$={cr_params['k']:.4f} fixed)")
    ax.plot(age_range, k_curve, "-", color=COLOR_PINN_K,
            label=f"PINN-$k$ curve ($y_{{max}}$={k_y_max:.1f}, $k$={k_k:.4f})")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Top height (m)")
    ax.set_title(f"Example plot {EXAMPLE_PLOT_ID}: a well-behaved personalisation")
    # Legend moved to bottom-right, 2026-08-24 -- the curves have already risen well clear of
    # this corner by the time age gets this high (all three sit above y=25 by age=60), so this
    # is the one corner nothing passes through, unlike "best" (matplotlib's auto-placement),
    # which was landing on top of the curves.
    ax.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    out = FIGURES_DIR / "q3_pinn_example_plot_curve.png"
    plt.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    build_env_sweep_chart()
    build_compartment_map()
    build_hotspot_label_map()
    build_pinn_k_deviation_maps_pooled()
    build_plausibility_distribution_chart()
    build_all_compartments_reference_map()
    build_good_example_plot_curve()
    build_yieldclass_scatter()
    build_importance_chart()
    build_flagged_plot_example()
    build_ymax_k_scatter()
    print("\nAll 7 figures built.")

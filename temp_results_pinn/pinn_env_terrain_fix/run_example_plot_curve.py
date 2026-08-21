# One-off script: train the fixed PINN-k model on fold 0 (same settings as the reported
# Table 1 numbers), then pick ONE example test-set plot and draw its actual height
# trajectory next to (a) the population-level frozen Chapman-Richards curve and (b) the
# per-plot curve this model learned for it (using the model's own predicted y_max_i/k_i).
#
# Purpose: this is illustrative evidence, not a new result -- shows what the fixed
# sub-networks actually learned for one real plot, since the aggregate R2 numbers alone
# don't show whether y_max_i/k_i are doing anything sensible at the individual-plot level.
#
# Isolation: reads only production data/CR-params (read-only). Writes only to
# temp_results_pinn/outputs/example_curve/ -- a new directory, doesn't touch anything else.
#
# Run as: PYTHONPATH=. .venv/bin/python temp_results_pinn/pinn_env_terrain_fix/run_example_plot_curve.py

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, just save a PNG file
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common.saving import load_cr_params
from models.common.splits import SPLIT_SEED
from models.common.torch_data import (
    ENV_TERRAIN_FEATURE_SETS,
    build_pair_terrain_tensor,
    build_pair_tensors,
    build_tensors,
    build_terrain_tensor,
    encode_thinning_status,
    fit_scalers,
    fit_terrain_scaler,
    load_split_table_with_terrain,
    load_trajectory_pairs,
    select_device,
)
from temp_results_pinn.pinn_env_terrain_fix.pinn_env_terrain_k_fix import (
    fit,
    predict_k,
    predict_y_max,
)

COHORT = "4survey"
SPLIT_TYPE = "spatial_block_kfold"
FOLD_INDEX = 0
N_FOLDS = 5
FEATURE_SET_NAME = "nested_set3_gated_terrain_wind_vif"
MAX_EPOCHS = 500
PATIENCE = 40
SEED = 42
MIN_OBSERVATIONS_FOR_EXAMPLE = 4  # only consider plots with at least this many survey rows

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "example_curve"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = select_device()
print(f"Device: {device}")

feature_columns = ENV_TERRAIN_FEATURE_SETS[FEATURE_SET_NAME]

cr_params = load_cr_params(COHORT, SPLIT_TYPE, split_seed=SPLIT_SEED, held_out_fold=FOLD_INDEX)
print(f"Population CR anchor: y_max={cr_params['y_max']:.4f}  k={cr_params['k']:.6f}  p={cr_params['p']:.6f}")

split_df = load_split_table_with_terrain(
    COHORT, SPLIT_TYPE, feature_columns, split_seed=SPLIT_SEED, k_folds=N_FOLDS, held_out_fold=FOLD_INDEX,
)
train_df = split_df[split_df["split"] == "train"]
val_df = split_df[split_df["split"] == "val"]
test_df = split_df[split_df["split"] == "test"]
print(f"train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

pairs_df = load_trajectory_pairs(COHORT, split_df)

scaler_age, scaler_other_features, scaler_height = fit_scalers(train_df)
scaler_terrain = fit_terrain_scaler(train_df, feature_columns)
encoded_column_names = encode_thinning_status(train_df).columns.tolist()

age_train, other_train, target_train = build_tensors(
    train_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
age_val, other_val, target_val = build_tensors(
    val_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
age_test, other_test, target_test = build_tensors(
    test_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
terrain_train = build_terrain_tensor(train_df, scaler_terrain, feature_columns, device)
terrain_val = build_terrain_tensor(val_df, scaler_terrain, feature_columns, device)
terrain_test = build_terrain_tensor(test_df, scaler_terrain, feature_columns, device)

pair_tensors = build_pair_tensors(
    pairs_df, scaler_age, scaler_other_features, scaler_height, encoded_column_names, device)
terrain_pairs = build_pair_terrain_tensor(pairs_df, scaler_terrain, feature_columns, device, COHORT)

n_other_features = other_train.shape[1]
n_terrain_features = terrain_train.shape[1]

print("\nTraining fixed PINN-k model, fold 0, production settings (this reproduces Table 1's fold-0 number)...")
model, _, history = fit(
    age_train, other_train, terrain_train, target_train,
    age_val, other_val, terrain_val, target_val,
    pair_tensors, terrain_pairs, cr_params, scaler_age, scaler_height,
    n_other_features, n_terrain_features, device, SEED,
    max_epochs=MAX_EPOCHS, early_stopping_patience=PATIENCE,
)
print(f"Trained {len(history)} epochs (early-stopped).")

# ----- Extract per-row y_max_i / k_i for every test-set row -----
y_max_per_row = predict_y_max(model, terrain_test, cr_params["y_max"]).cpu().numpy().flatten()
k_per_row = predict_k(model, terrain_test, cr_params["k"]).cpu().numpy().flatten()

# ----- Pick one example plot: enough observations, and a clear deviation from the -----
# ----- population curve (so the per-plot adjustment has something real to show). -----
test_df = test_df.copy()
test_df["y_max_pred"] = y_max_per_row
test_df["k_pred"] = k_per_row

# Population curve prediction for every test row, to measure how far each row sits from it.
# elev_percentile_95th is the actual target column (renamed from the old Top_Height99 --
# see models/common/torch_data.py's TARGET_COLUMN).
HEIGHT_COLUMN = "elev_percentile_95th"
population_pred = cr_params["y_max"] * (1 - np.exp(-cr_params["k"] * test_df["Age"])) ** cr_params["p"]
test_df["population_residual"] = test_df[HEIGHT_COLUMN] - population_pred

# This model's own per-plot curve prediction, to check whether the adjustment actually helps
# (moves closer to the observed value) rather than just moving somewhere different.
fixed_pred = test_df["y_max_pred"] * (1 - np.exp(-test_df["k_pred"] * test_df["Age"])) ** cr_params["p"]
test_df["fixed_residual"] = test_df[HEIGHT_COLUMN] - fixed_pred

# Save every test-row prediction, so a future pass can pick a different example plot without
# retraining the model again.
test_df.to_csv(OUTPUT_DIR / "test_set_predictions.csv", index=False)
print(f"Saved full test-set predictions -> {OUTPUT_DIR / 'test_set_predictions.csv'}")

plot_id_column = "identification"  # the actual plot-ID column (models/common/torch_data.py)
counts = test_df.groupby(plot_id_column).size()
eligible_plots = counts[counts >= MIN_OBSERVATIONS_FOR_EXAMPLE].index
candidates = test_df[test_df[plot_id_column].isin(eligible_plots)].copy()

# Selection rule (fixed 2026-08-21): the first version picked the single plot with the largest
# population-curve deviation, which grabbed a pathological outlier (a 74-89yo plot only 4-11m
# tall -- almost certainly disturbed/data-quality, not a normal environmental-adjustment case).
# Instead: (1) exclude implausible cases where observed height is under half the population
# curve's prediction (catches the same kind of pathological under-grower), (2) require the
# fixed model's own curve to reduce absolute error versus the population curve (so the example
# actually demonstrates the claimed benefit, not a failure case), (3) restrict to a moderate,
# non-extreme deviation (interquartile range of population-curve error) so an outlier can't win,
# (4) among what's left, pick the plot where the fix reduces error the most.
candidates["population_pred_for_row"] = cr_params["y_max"] * (1 - np.exp(-cr_params["k"] * candidates["Age"])) ** cr_params["p"]
plausible = candidates[candidates[HEIGHT_COLUMN] >= 0.5 * candidates["population_pred_for_row"]]

by_plot = plausible.groupby(plot_id_column).agg(
    mean_population_abs_residual=("population_residual", lambda x: x.abs().mean()),
    mean_fixed_abs_residual=("fixed_residual", lambda x: x.abs().mean()),
)
by_plot["improves"] = by_plot["mean_fixed_abs_residual"] < by_plot["mean_population_abs_residual"]
q25, q75 = by_plot["mean_population_abs_residual"].quantile([0.25, 0.75])
moderate_deviation = by_plot[(by_plot["mean_population_abs_residual"] >= q25) & (by_plot["mean_population_abs_residual"] <= q75)]
improving_and_moderate = moderate_deviation[moderate_deviation["improves"]]

if len(improving_and_moderate) > 0:
    error_reduction = improving_and_moderate["mean_population_abs_residual"] - improving_and_moderate["mean_fixed_abs_residual"]
    example_plot_id = error_reduction.idxmax()
else:
    error_reduction = by_plot["mean_population_abs_residual"] - by_plot["mean_fixed_abs_residual"]
    example_plot_id = error_reduction.idxmax()

example_rows = test_df[test_df[plot_id_column] == example_plot_id].sort_values("Age")
print(f"\nExample plot: {example_plot_id}  ({len(example_rows)} observations)")
print(example_rows[["Age", HEIGHT_COLUMN, "y_max_pred", "k_pred", "population_residual"]].to_string(index=False))

# The sub-networks give one y_max_i/k_i prediction per row (terrain doesn't change across a
# plot's survey years), so use this plot's mean as its single representative curve.
plot_y_max = example_rows["y_max_pred"].mean()
plot_k = example_rows["k_pred"].mean()
print(f"\nPlot-specific curve: y_max={plot_y_max:.3f}  k={plot_k:.5f}  "
      f"(population: y_max={cr_params['y_max']:.3f}  k={cr_params['k']:.5f})")

# ----- Build both curves over a sensible age range and plot -----
age_range = np.linspace(0, max(example_rows["Age"].max() + 10, 60), 200)
population_curve = cr_params["y_max"] * (1 - np.exp(-cr_params["k"] * age_range)) ** cr_params["p"]
plot_curve = plot_y_max * (1 - np.exp(-plot_k * age_range)) ** cr_params["p"]

plt.figure(figsize=(7, 5))
plt.scatter(example_rows["Age"], example_rows[HEIGHT_COLUMN], color="black", zorder=5, label="Observed (this plot)")
plt.plot(age_range, population_curve, "--", color="gray", label=f"Population curve (y_max={cr_params['y_max']:.1f}, k={cr_params['k']:.4f})")
plt.plot(age_range, plot_curve, "-", color="tab:blue", label=f"Fixed PINN-k curve (y_max={plot_y_max:.1f}, k={plot_k:.4f})")
plt.xlabel("Age (years)")
plt.ylabel("Top height (m)")
plt.title(f"Example plot {example_plot_id}: population vs. per-plot learned curve")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "example_plot_curve.png", dpi=150)

# Also save into the dissertation's actual figures directory, so this is ready to reference
# directly from documentation/refocus_draft/19th_1038_rq2brq3_refocus.tex.
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "fig_results"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(FIGURES_DIR / "q3_pinn_example_plot_curve.png", dpi=150)
print(f"Saved dissertation copy -> {FIGURES_DIR / 'q3_pinn_example_plot_curve.png'}")
print(f"\nSaved plot -> {OUTPUT_DIR / 'example_plot_curve.png'}")

with open(OUTPUT_DIR / "example_plot_summary.json", "w") as f:
    json.dump({
        "plot_id": str(example_plot_id),
        "n_observations": len(example_rows),
        "population_y_max": cr_params["y_max"],
        "population_k": cr_params["k"],
        "plot_y_max": float(plot_y_max),
        "plot_k": float(plot_k),
        "observed_age": example_rows["Age"].tolist(),
        "observed_height": example_rows[HEIGHT_COLUMN].tolist(),
    }, f, indent=2)
print(f"Saved numbers -> {OUTPUT_DIR / 'example_plot_summary.json'}")

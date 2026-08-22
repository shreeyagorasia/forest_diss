# Run as: PYTHONPATH=. .venv/bin/python -m models.growth_curve_attribution.rq3_free_k_pilot_check
#
# Cheap pilot for the "would freeing k help Q2's target?" question (see q2_brain_dump_21_08.md,
# section 7). Instead of building the full shrinkage/mixed-effects fix (days of work, not
# attempted), this fits each plot's own y_max AND growth-rate parameter (p4, "k" in
# Chapman-Richards terms) freely via per-plot non-linear least squares -- NO shrinkage toward the
# yield-class value, the cheap/crude version of the idea. p5 (the other shape parameter) stays
# fixed at the plot's own yldc value; freeing 3 parameters from 4 survey points is not attempted.
#
# Purpose: a fast, cheap signal of whether the growth-rate/ceiling entanglement is worth the
# expensive proper fix. If XGBoost's R2 on this new (free-k, no-shrinkage) target improves a lot
# over the current fixed-k target, that's evidence the entanglement really is costing real signal,
# and justifies the harder shrinkage build. If it does NOT improve (or gets worse), that's useful
# negative evidence too -- consistent with the identifiability concern (42% of plots never
# observed past age 40, so freely fitting 2 parameters from a still-rising curve is
# poorly-determined for a large share of the population) actually biting in practice, not just in
# theory.
#
# NOT the same as the honest future-work fix (shrinkage) -- this is the crude, no-shrinkage
# version, expected to be noisy for the young/short-window plots. That's the point: it's a cheap
# way to see whether pursuing the expensive, correct version is worth it at all.

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from models.common.metrics import compute_metrics
from models.common.splits import DEFAULT_K_FOLDS, SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, SPLIT_SEED, spatial_kfold_split
from models.growth_curve_attribution.data import load_filtered_growth_curve_table
from models.growth_curve_attribution.disturbance_checks import summarize_plot_disturbance_status
from models.xgb_environmental.data import load_environmental_features
from models.xgb_environmental.feature_set_builder import load_feature_set
from models.xgb_environmental.xgb_environmental import fit_with_columns, predict_with_columns

COHORT = "4survey"
HEIGHT_COLUMN = "Top_Height95"
SET_NAME = "nested_set4_gated_all_vif"


def chapman_richards_free_k(age, y_max, p4, p5):
    return y_max * (1 - np.exp(-p4 * age)) ** p5


def fit_free_k_per_plot(growth_rows):
    """Per-plot non-linear least squares for (y_max, p4), p5 held fixed at the plot's own yldc
    value. Initial guess and starting point taken from the plot's own yldc curve, since that's
    the best prior available -- not because we trust it, just to give the optimiser a sane start.
    Returns one row per plot: y_max_free, p4_free, converged (bool), n_fitting_points.
    """
    results = []
    for identification, group in growth_rows.groupby("identification"):
        age = group["Age"].to_numpy(dtype=float)
        height = group[HEIGHT_COLUMN].to_numpy(dtype=float)
        p5_fixed = group["p5"].iloc[0]
        y_max_guess = group["y_max_yldc"].iloc[0]
        p4_guess = group["p4"].iloc[0]

        if len(age) < 2 or pd.isna(p5_fixed) or pd.isna(y_max_guess) or pd.isna(p4_guess):
            results.append({"identification": identification, "y_max_free": np.nan, "p4_free": np.nan,
                             "converged": False, "n_fitting_points": len(age)})
            continue

        try:
            popt, _ = curve_fit(
                lambda a, y_max, p4: chapman_richards_free_k(a, y_max, p4, p5_fixed),
                age, height,
                p0=[y_max_guess, p4_guess],
                bounds=([1.0, 1e-5], [200.0, 5.0]),  # y_max in [1,200]m, p4 positive -- generous, not tuned
                maxfev=2000,
            )
            y_max_free, p4_free = popt
            converged = True
        except RuntimeError:
            y_max_free, p4_free, converged = np.nan, np.nan, False

        results.append({"identification": identification, "y_max_free": y_max_free, "p4_free": p4_free,
                         "converged": converged, "n_fitting_points": len(age)})

    return pd.DataFrame(results)


def main():
    growth_rows = load_filtered_growth_curve_table(COHORT).dropna(subset=["y_max_yldc"]).copy()
    disturbance_status = summarize_plot_disturbance_status(COHORT)
    excluded_ids = set(disturbance_status.loc[disturbance_status["exclude_from_curve_fit"], "identification"])
    growth_rows = growth_rows[~growth_rows["identification"].isin(excluded_ids)]
    print(f"Population after disturbance cleaning: {growth_rows['identification'].nunique():,} plots")

    print("Fitting free (y_max, k) per plot via non-linear least squares (this is the slow step)...")
    free_fit = fit_free_k_per_plot(growth_rows)
    n_total = len(free_fit)
    n_converged = free_fit["converged"].sum()
    print(f"Converged: {n_converged:,} / {n_total:,} ({n_converged / n_total * 100:.1f}%)")

    # Diagnostic: how extreme are the free-fit y_max values, and does it correlate with never
    # having been observed near the ceiling (age < 40)?
    max_age_per_plot = growth_rows.groupby("identification")["Age"].max()
    free_fit = free_fit.merge(max_age_per_plot.rename("max_age"), on="identification", how="left")
    free_fit["never_near_ceiling"] = free_fit["max_age"] < 40

    converged = free_fit[free_fit["converged"]].copy()
    extreme = converged[(converged["y_max_free"] > 60) | (converged["y_max_free"] < 5)]
    print(f"\nExtreme y_max_free values (<5m or >60m, physically implausible): {len(extreme):,} / {len(converged):,} ({len(extreme)/len(converged)*100:.1f}%)")
    print(f"  ...of which never observed past age 40: {extreme['never_near_ceiling'].mean()*100:.1f}%")
    print(f"  (compare: {converged['never_near_ceiling'].mean()*100:.1f}% of ALL converged plots never observed past age 40)")

    # Build the new target: Delta y_max_free = y_max_free - y_max_yldc, only for converged,
    # non-extreme fits (a real pipeline would need to decide what to do with the rest -- this
    # pilot just drops them, disclosed here, not hidden).
    plot_static = (
        growth_rows.sort_values("LiDAR_year").groupby("identification", as_index=False).first()
        [["identification", "cpmt", "x", "y", "y_max_yldc"]]
    )
    pilot_table = plot_static.merge(converged[["identification", "y_max_free", "converged"]], on="identification", how="inner")
    pilot_table = pilot_table[~pilot_table["identification"].isin(extreme["identification"])]
    pilot_table["local_y_max_difference_free"] = pilot_table["y_max_free"] - pilot_table["y_max_yldc"]
    print(f"\nFinal pilot population (converged, non-extreme): {len(pilot_table):,} plots")

    # --- Fit XGBoost on the NEW target, same Set4 features, same spatial CV convention ---
    # Cheap-pilot simplification: use only the Set4 columns present bare in the raw environmental
    # parquet, dropping cohort-suffixed (tas_mean) and one-hot categorical (ceh_*) columns rather
    # than reimplementing prepare_broad_table()'s cohort/dummy resolution for this pilot. Loses 5
    # of 19 columns -- an acceptable simplification for a fast "is this worth pursuing" signal,
    # not a headline number; disclosed here, not hidden.
    raw_columns_full = load_feature_set("RSQ3", SET_NAME)
    env = load_environmental_features()
    raw_columns = [c for c in raw_columns_full if c in env.columns]
    dropped = [c for c in raw_columns_full if c not in env.columns]
    print(f"\nUsing {len(raw_columns)}/{len(raw_columns_full)} Set4 columns (dropped, not bare in parquet: {dropped})")
    table = pilot_table.merge(env[["identification"] + raw_columns], on="identification", how="left")
    table = table.dropna(subset=raw_columns + ["local_y_max_difference_free"])
    print(f"After merging Set4 features (complete-case): {len(table):,} plots")

    r2_per_fold = []
    for fold in range(DEFAULT_K_FOLDS):
        table["split"] = spatial_kfold_split(
            table, block_col=SPATIAL_BLOCK_COL, k=DEFAULT_K_FOLDS, held_out_fold=fold,
            buffer_distance=SPATIAL_BUFFER_METRES, seed=SPLIT_SEED,
        )
        train_df = table[table["split"] == "train"]
        val_df = table[table["split"] == "val"]
        test_df = table[table["split"] == "test"]

        xgb_model = fit_with_columns(
            train_df, raw_columns, val_df=val_df, target_col="local_y_max_difference_free",
            n_jobs=1, n_estimators=500, max_depth=4, learning_rate=0.04,
        )
        xgb_pred = predict_with_columns(test_df, xgb_model, raw_columns)
        fold_r2 = compute_metrics(test_df["local_y_max_difference_free"], xgb_pred)["r2"]
        r2_per_fold.append(fold_r2)
        print(f"  fold {fold}: XGB R2={fold_r2:.4f} (train={len(train_df):,} test={len(test_df):,})")

    r2_per_fold = np.array(r2_per_fold)
    print(f"\nXGBoost on FREE-k (no shrinkage) target: fold-mean R2 = {r2_per_fold.mean():.3f} +/- {r2_per_fold.std():.3f}")
    print("Compare: XGBoost on current fixed-k target, Set4, WITH CanopyCover = 0.250")
    print("Compare: XGBoost on current fixed-k target, Set4, WITHOUT CanopyCover = 0.061")


if __name__ == "__main__":
    main()

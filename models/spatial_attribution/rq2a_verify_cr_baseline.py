# Purpose: a single, consistent recomputation of the CR baseline mean|residual| and the
# residual-reduction (mean reduction, % rows improved) for every model row in RQ2a's results
# table. DNN/PINN/PINN_k (Set3, seed 42) and XGBoost's env/no-env arms. All read directly from
# the predictions.csv files currently saved on disk, joined against the CURRENT chapman_richards
# baseline files, using the project's own compute_residual_reduction() unmodified.
#
# Why this exists: an audit (2026-08-16) found the CR baseline stated for every 6survey row in
# TEMP_rq2a_residual_reduction_results_2026-08-11.tex (3.057) does not match what a fresh
# recomputation from currently-saved files gives (3.047). Confirmed consistently across DNN,
# PINN, and PINN_k, all pointing to the same 3.047 figure, not a one-off fluke. Likely cause:
# chapman_richards_fold{0-4}'s predictions.csv were regenerated (fold split assignment CSVs get
# rewritten every time build_split_for_cohort() runs, which has happened repeatedly across this
# project's later sessions) after the 2026-08-11 note's numbers were computed, so the 6survey CR
# fit itself shifted slightly. The 4survey CR baseline (5.0285) is unaffected. It matches the
# original note (5.028) closely. This script is the reproducible source for the corrected 6survey
# number and should be the reference for these figures going forward, not the 2026-08-11 note's
# original 3.057.
#
# Separately confirms XGBoost's own no-env arm has a genuinely different (larger) test population
# than every other Set3 row (232,448 vs. 232,292 on 4survey, since it isn't filtered for missing
# environmental data). So its own CR baseline (5.030) is legitimately different from the other
# rows' (5.028), not an inconsistency to paper over.
#
# Run as: python -m models.spatial_attribution.rq2a_verify_cr_baseline

import pandas as pd

from models.spatial_attribution.rq2_residual_reduction import compute_residual_reduction, load_predictions

MODEL_RUNS = {
    "DNN": "rq1_dnn_env_terrain_nested_set3_gated_terrain_wind_vif_seed42",
    "PINN": "rq1_pinn_env_terrain_nested_set3_gated_terrain_wind_vif_seed42",
    "PINN_k": "rq1_pinn_env_terrain_k_nested_set3_gated_terrain_wind_vif_seed42",
    "XGBoost env": "xgb_env_terrain_fixed_nested_set3_gated_terrain_wind_vif",
    "XGBoost no-env": "xgb_noenv_fixed",
}
N_FOLDS = 5


def pooled_reduction(model_run, cohort):
    cr_frames, model_frames = [], []
    for fold in range(N_FOLDS):
        cr_frames.append(load_predictions(f"chapman_richards_fold{fold}", cohort, "spatial_block_kfold"))
        model_frames.append(load_predictions(model_run, cohort, "spatial_block_kfold", held_out_fold=fold))
    cr_all = pd.concat(cr_frames, ignore_index=True)
    model_all = pd.concat(model_frames, ignore_index=True)
    overall, _, _ = compute_residual_reduction(cr_all, model_all)
    return overall


def main():
    for cohort in ["4survey", "6survey"]:
        print(f"=== {cohort} ===")
        for name, run in MODEL_RUNS.items():
            overall = pooled_reduction(run, cohort)
            pct_of_baseline = overall["mean_reduction"] / overall["cr_mean_abs_residual"] * 100
            print(
                f"  {name}: n={overall['n_rows']:,}  CR={overall['cr_mean_abs_residual']:.4f}  "
                f"model={overall['model_mean_abs_residual']:.4f}  reduction={overall['mean_reduction']:.4f} "
                f"({pct_of_baseline:.1f}% of baseline)  pct_improved={overall['pct_rows_improved']:.1f}"
            )
        print()


if __name__ == "__main__":
    main()

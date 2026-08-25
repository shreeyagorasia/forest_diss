# Which RQ this closes a gap in, and why: RQ2a item 1 (does environmental conditioning trade
# accuracy between subgroups, helping plots the shared Chapman-Richards curve fits worst while
# hurting the plots it already fits best). The results draft flagged this open: the five-seed
# reseed check already run (TEMP_rq2a_residual_reduction_results_2026-08-11.tex) confirmed Q4's
# correction is reproducible across seeds, but could not tell whether Q1's degradation reflects a
# genuine trade-off from using real environmental signal, or is just what any flexible model with
# output variance would show on Q1 regardless of what it's fed, since Q1 was selected specifically
# because the CR baseline's error was already smallest there.
#
# This is the direct test: permute Set3's environmental columns across rows before fitting, so the
# model trains on real heights and a real no-environment feature set, but a FAKE, scrambled
# environment assignment. If the permuted-feature model still degrades Q1 by a similar margin to
# the real environment-conditioned model (xgb_env_terrain_fixed_..., already fit by
# rq2a_xgb_check.py), that is evidence the degradation doesn't require real environmental signal at
# all. If it shows much less degradation, that supports the current write-up's story as-is.
#
# Fits ONLY the permuted arm. The real env-conditioned and no-env arms already exist from
# rq2a_xgb_check.py's own run and are reused unchanged for comparison. Reuses fit_one_arm() and
# run_reduction_check() directly from that script, and compute_residual_reduction() from
# rq2_residual_reduction.py. No reimplementation of the fitting or quartile logic.
#
# Run as: python -m models.spatial_attribution.rq2a_permutation_check

import numpy as np

from models.baselines.run_baselines import DEFAULT_K_FOLDS, MATURITY_AGE_MIN_DEFAULT, SEED, build_split_for_cohort
from models.baselines.run_baselines_env import load_full_rows_with_split, merge_environmental_features
from models.common.saving import model_output_dir
from models.common.torch_data import ENV_TERRAIN_FEATURE_SETS
from models.spatial_attribution.rq2a_xgb_check import SET_NAME, fit_one_arm, run_reduction_check
from models.xgb_baseline.xgb_baseline import FEATURE_COLUMNS

PERMUTE_SEED = 42
RUN_NAME = "xgb_env_terrain_permuted_" + SET_NAME


def permute_env_columns(df, env_columns, seed):
    # One shared row-permutation applied to the whole environmental block together, not one
    # independent shuffle per column. This keeps the environmental variables' own correlation
    # structure intact (e.g. elevation and temperature still covary the way they really do in
    # Aberfoyle). It only breaks which plot each environmental vector is actually attached to.
    # An independent per-column shuffle would additionally scramble that correlation structure,
    # which is not the property this test is checking.
    rng = np.random.RandomState(seed)
    permuted_row_order = rng.permutation(len(df))
    permuted_df = df.copy()
    permuted_df[env_columns] = df[env_columns].to_numpy()[permuted_row_order]
    return permuted_df


def fit_and_save_fold(cohort, held_out_fold, k_folds=DEFAULT_K_FOLDS):
    name_suffix = f"_fold{held_out_fold}"
    _, split_assignment = build_split_for_cohort(
        cohort, "spatial_block_kfold", split_seed=SEED, maturity_age_min=MATURITY_AGE_MIN_DEFAULT,
        name_suffix=name_suffix, k_folds=k_folds, held_out_fold=held_out_fold,
    )
    filtered_df = load_full_rows_with_split(cohort, split_assignment)
    env_columns = list(ENV_TERRAIN_FEATURE_SETS[SET_NAME])
    env_df = merge_environmental_features(filtered_df, env_columns, cohort)

    # Permute once per fold, seeded off the fold number so each fold gets its own (but
    # reproducible) fake environment assignment rather than reusing the identical shuffle five
    # times over.
    permuted_df = permute_env_columns(env_df, env_columns, seed=PERMUTE_SEED + held_out_fold)

    train_df = permuted_df[permuted_df["split"] == "train"]
    val_df = permuted_df[permuted_df["split"] == "val"]
    test_df = permuted_df[permuted_df["split"] == "test"]
    predictions = fit_one_arm(train_df, val_df, test_df, FEATURE_COLUMNS + env_columns)

    output_dir = model_output_dir(RUN_NAME, cohort, f"fold_{held_out_fold}", split_type="spatial_block_kfold")
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    print(f"  Saved {RUN_NAME} / {cohort} / fold {held_out_fold} -> {output_dir / 'predictions.csv'} ({len(predictions)} test rows)")


def main():
    for cohort in ["4survey", "6survey"]:
        for fold in range(DEFAULT_K_FOLDS):
            fit_and_save_fold(cohort, fold)

    print("\n===== Permuted-environment control: residual reduction vs Chapman-Richards =====")
    for cohort in ["4survey", "6survey"]:
        overall_df, pooled_quartiles = run_reduction_check(RUN_NAME, cohort)
        print(f"\n{RUN_NAME} / {cohort}")
        print(f"  Mean reduction across folds: {overall_df['mean_reduction'].mean():.4f} (SD {overall_df['mean_reduction'].std():.4f})")
        print(pooled_quartiles.to_string())


if __name__ == "__main__":
    main()

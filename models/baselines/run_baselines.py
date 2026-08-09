# Run as: python -m models.baselines.run_baselines
#     or: python -m models.baselines.run_baselines --split-type spatial_block
#     or: python -m models.baselines.run_baselines --split-type temporal
#
# Fits Chapman-Richards, average-by-age, linear regression, and random
# forest baselines for both cohorts. This pass only fits and saves
# parameters/models -- evaluation happens separately in evaluate_baselines.py.
#
# --split-type plot_level (default) is the easy, already-established split:
# individual plots are shuffled randomly into train/val/test, so a test plot
# usually sits right next to a training plot. --split-type spatial_block is
# a harder, more realistic test: whole forestry compartments (cpmt) are
# shuffled into train/val/test instead, so a test plot's nearest training
# neighbour can be kilometres away. --split-type temporal is a third,
# different question: does the model predict a real FUTURE survey it never
# trained on -- train on 2008+2012 (4survey) / 2002-2012 (6survey), validate
# on 2021, test on 2023 (see TEMPORAL_YEARS in models/common/splits.py).
# Unlike the other two splits, the same plot legitimately appears in both
# train and test rows here -- that is expected, not a leak, since this
# split tests time generalisation, not plot or place generalisation. Every
# output path is kept separate per split type (outputs/spatial_block/...,
# outputs/temporal/... vs outputs/...), so running one never overwrites
# another.
#
# Every model's fit attempt is logged to outputs/run_logs/, one JSON file
# per model per cohort, whether it succeeds or fails -- see
# models/common/run_logging.py. A failed fit is caught, logged, and printed
# as a warning; it does NOT stop the other three models from being fit.

import argparse
from pathlib import Path

from models.average_by_age.average_by_age import fit as fit_average_by_age
from models.average_by_age.average_by_age import save_lookup
from models.chapman_richards.chapman_richards import fit as fit_chapman_richards
from models.chapman_richards.chapman_richards import save_params as save_cr_params
from models.common.data import filter_data, load_cohort_data, load_model_table
from models.common.run_logging import RunTimer, format_error, write_run_log, write_started_marker
from models.common.saving import model_output_dir as output_dir
from models.common.splits import (
    DEFAULT_K_FOLDS,
    SPATIAL_BLOCK_COL,
    SPATIAL_BUFFER_METRES,
    TEMPORAL_YEARS,
    TEMPORAL_YEARS_NARROW_GAP,
    plot_level_split,
    spatial_block_split,
    spatial_kfold_split,
    temporal_split,
)
from models.linear_baseline.linear_baseline import fit as fit_linear_baseline
from models.linear_baseline.linear_baseline import save_params as save_linear_params
from models.rf_baseline.rf_baseline import fit as fit_rf_baseline
from models.rf_baseline.rf_baseline import save_model as save_rf_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COHORTS = ["4survey", "6survey"]
SEED = 42

# filter_data()'s own default -- kept as its own named constant here (not just re-typing "30")
# so build_split_for_cohort()/run_for_cohort() can compare against it to decide whether a
# non-default --maturity-age-min needs its own suffixed output path, same pattern as split_seed.
MATURITY_AGE_MIN_DEFAULT = 30

# None of these four baselines use a GPU -- sklearn/scipy only. Recorded in
# every log entry anyway for schema consistency with the DNN/PINN logs,
# where device actually varies (cpu/cuda/mps).
DEVICE = "cpu"

# The prior dissertation's Chapman-Richards fit, for a quick sanity check.
PRIOR_CR_PARAMS = {"y_max": 46.1126, "k": 0.01866979, "p": 1.0175}


def build_split_for_cohort(
    cohort, split_type, split_seed=SEED, maturity_age_min=MATURITY_AGE_MIN_DEFAULT, name_suffix="",
    k_folds=DEFAULT_K_FOLDS, held_out_fold=0,
):
    # The split is computed ONCE per cohort, from the smallest shared table
    # (identification, LiDAR_year, blk, cpmt, Age, yldc, elev_percentile_95th --
    # everything Chapman-Richards and average-by-age need), then saved.
    # linear_baseline and rf_baseline reuse this exact split by merging onto
    # it (in load_train_rows), rather than recomputing the split separately
    # -- that guarantees all four baselines share identical train/val/test
    # membership, even though every model now reads from the same
    # consolidated model_table.parquet (2026-07-28, see
    # data_processing/export_model_tables.py).
    #
    # filter_data() now gates whole plots on their Age at the 2023 survey
    # (see models/common/data.py), so a plot's early rows here can still show
    # Age well under 20 -- that is expected, not a bug.
    df = load_cohort_data(cohort)
    filtered_df = filter_data(df, maturity_age_min=maturity_age_min)

    if split_type == "plot_level":
        # Returns a three-way train/val/test split (60/20/20). Neither
        # Chapman-Richards, average-by-age, nor linear regression has
        # anything to tune (no hyperparameters, no early stopping), and the
        # RF baseline isn't tuned yet either, so val is saved here for
        # schema consistency with later models only -- it is not read by
        # any model fitted in this script. Only train is used for fitting.
        filtered_df["split"] = plot_level_split(filtered_df, seed=split_seed)
    elif split_type == "spatial_block":
        # Whole compartments go to train/val/test together, with a 60m
        # buffer removing any TRAINING plot too close to a val/test plot --
        # see spatial_block_split()/apply_spatial_buffer() in
        # models/common/splits.py for why only the train side is buffered.
        # coordinates_df=None makes it load_plot_coordinates() itself.
        filtered_df["split"] = spatial_block_split(
            filtered_df,
            block_col=SPATIAL_BLOCK_COL,
            buffer_distance=SPATIAL_BUFFER_METRES,
            seed=split_seed,
        )
    elif split_type == "temporal":
        # Every plot in this data has full year coverage for its cohort (see
        # documentation/model_instructions), so this always covers every row
        # -- no "unassigned" rows are expected here.
        filtered_df["split"] = temporal_split(
            filtered_df,
            year_col="LiDAR_year",
            **TEMPORAL_YEARS[cohort],
        )
    elif split_type == "temporal_narrow_gap":
        # Same temporal_split() function, a different year assignment (2-year
        # train-to-test gap instead of temporal's 11 years) -- see
        # TEMPORAL_YEARS_NARROW_GAP in models/common/splits.py.
        filtered_df["split"] = temporal_split(
            filtered_df,
            year_col="LiDAR_year",
            **TEMPORAL_YEARS_NARROW_GAP[cohort],
        )
    elif split_type == "spatial_block_kfold":
        # A CR anchor fit on a plain spatial_block split's train set would leak some of THIS
        # fold's held-out compartments into the "frozen" physics anchor pinn_env_terrain/
        # pinn_env_terrain_k read -- fitting CR (and, for consistency, the other three baselines)
        # separately per fold, on that fold's own train set, keeps the anchor genuinely
        # fold-matched. See models/common/splits.py::spatial_kfold_split() for the fold mechanics.
        filtered_df["split"] = spatial_kfold_split(
            filtered_df,
            block_col=SPATIAL_BLOCK_COL,
            k=k_folds,
            held_out_fold=held_out_fold,
            buffer_distance=SPATIAL_BUFFER_METRES,
            seed=split_seed,
        )
    else:
        raise ValueError(f"Unknown split_type: {split_type!r}")

    split_path = output_dir(f"splits{name_suffix}", cohort, "split_assignment.csv", split_type=split_type)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df[["identification", "LiDAR_year", "split"]].to_csv(split_path, index=False)
    print(f"  Saved split assignment -> {split_path}")

    for split_name in sorted(filtered_df["split"].unique()):
        row_count = (filtered_df["split"] == split_name).sum()
        print(f"  {split_name} rows: {row_count:,}")

    split_assignment = filtered_df[["identification", "LiDAR_year", "split"]]
    return filtered_df, split_assignment


def load_train_rows(cohort, table_name, split_assignment, maturity_age_min=MATURITY_AGE_MIN_DEFAULT):
    # Every model reads the same consolidated model_table.parquet now (2026-07-28) --
    # table_name is kept as a parameter purely to label the assertion message below, not to pick
    # a different source file.
    #
    # maturity_age_min MUST match whatever build_split_for_cohort() used to build
    # split_assignment, or the row-count assertion below fails -- run_for_cohort() passes the
    # same value to both, so this can only drift if a caller bypasses run_for_cohort().
    table = load_model_table(cohort)
    filtered_table = filter_data(table, maturity_age_min=maturity_age_min)

    merged = filtered_table.merge(split_assignment, on=["identification", "LiDAR_year"], how="inner")
    assert len(merged) == len(filtered_table), (
        f"Row count changed after merging {table_name} with the shared split assignment -- "
        "the source tables may no longer agree on which rows survive filtering."
    )

    return merged[merged["split"] == "train"]


def fit_chapman_richards_logged(cohort, split_type, cr_train_df, n_rows_fit, split_seed=SEED, maturity_age_min=MATURITY_AGE_MIN_DEFAULT, name_suffix=""):
    # Returns cr_params, or None if the fit failed/didn't converge -- callers
    # already handle a None result (average-by-age and the summary printers
    # skip it gracefully).
    #
    # name_suffix (2026-08-02 addition): "" for the default split_seed (SEED, every prior
    # result in this repo) -- writes to the exact same plain path as before, zero change.
    # Non-default split_seed appends "_splitseed<N>" so a robustness-check refit never
    # overwrites the primary CR anchor pinn_noenv/pinn_env_terrain's load_cr_params() reads by
    # default. See documentation/experiment_log.md's 2026-08-02 split-seed robustness entries --
    # this is the CR-anchor half of that check: pinn_noenv/pinn_env_terrain's own load_cr_params()
    # needs a split-seed-MATCHED anchor to be a valid comparison, not the default-seed one.
    timer = RunTimer().start()
    hyperparameters = {"seed": SEED, "split_seed": split_seed, "maturity_age_min": maturity_age_min}
    attempt_id = write_started_marker(
        model_name="chapman_richards", cohort=cohort, split_type=split_type, run_phase="fit",
        is_test_run=False, device=DEVICE, hyperparameters=hyperparameters,
    )
    try:
        cr_params = fit_chapman_richards(cr_train_df)
        cr_output_path = output_dir(f"chapman_richards{name_suffix}", cohort, "params.json", split_type=split_type)
        save_cr_params(cr_params, cohort, n_rows_fit, cr_output_path)
        print(f"  Chapman-Richards params saved -> {cr_output_path}")

        if cr_params["y_max"] <= 0 or cr_params["k"] <= 0 or cr_params["p"] <= 0:
            print("  WARNING: a fitted Chapman-Richards parameter is zero or negative — check this cohort's fit.")

        max_observed_height = cr_train_df["elev_percentile_95th"].max()
        if cr_params["y_max"] < max_observed_height:
            print(
                f"  WARNING: fitted y_max ({cr_params['y_max']:.2f} m) is below the "
                f"max observed training height ({max_observed_height:.2f} m) — "
                "this fit looks unstable, not just a low asymptote."
            )

        write_run_log(
            attempt_id=attempt_id,
            model_name="chapman_richards", cohort=cohort, split_type=split_type, run_phase="fit",
            status="success", is_test_run=False, device=DEVICE,
            hyperparameters=hyperparameters, metrics=None, error=None,
            output_dir=cr_output_path.parent, runtime_seconds=timer.elapsed_seconds(), n_rows_fit=n_rows_fit,
        )
        return cr_params
    except Exception as error:
        write_run_log(
            attempt_id=attempt_id,
            model_name="chapman_richards", cohort=cohort, split_type=split_type, run_phase="fit",
            status="failed", is_test_run=False, device=DEVICE,
            hyperparameters=hyperparameters, metrics=None, error=format_error(error),
            output_dir=None, runtime_seconds=timer.elapsed_seconds(), n_rows_fit=n_rows_fit,
        )
        print(f"  WARNING: Chapman-Richards fit failed for {cohort}: {error}")
        return None


def fit_average_by_age_logged(cohort, split_type, avg_train_df, n_rows_fit, split_seed=SEED, maturity_age_min=MATURITY_AGE_MIN_DEFAULT, name_suffix=""):
    timer = RunTimer().start()
    hyperparameters = {"seed": SEED, "split_seed": split_seed, "maturity_age_min": maturity_age_min}
    attempt_id = write_started_marker(
        model_name="average_by_age", cohort=cohort, split_type=split_type, run_phase="fit",
        is_test_run=False, device=DEVICE, hyperparameters=hyperparameters,
    )
    try:
        lookup_table, fallback_mean_height = fit_average_by_age(avg_train_df)
        avg_output_path = output_dir(f"average_by_age{name_suffix}", cohort, "lookup.json", split_type=split_type)
        save_lookup(lookup_table, fallback_mean_height, cohort, n_rows_fit, avg_output_path)
        print(f"  Average-by-age lookup saved -> {avg_output_path}")

        write_run_log(
            attempt_id=attempt_id,
            model_name="average_by_age", cohort=cohort, split_type=split_type, run_phase="fit",
            status="success", is_test_run=False, device=DEVICE,
            hyperparameters=hyperparameters, metrics=None, error=None,
            output_dir=avg_output_path.parent, runtime_seconds=timer.elapsed_seconds(), n_rows_fit=n_rows_fit,
        )
        return lookup_table, fallback_mean_height
    except Exception as error:
        write_run_log(
            attempt_id=attempt_id,
            model_name="average_by_age", cohort=cohort, split_type=split_type, run_phase="fit",
            status="failed", is_test_run=False, device=DEVICE,
            hyperparameters=hyperparameters, metrics=None, error=format_error(error),
            output_dir=None, runtime_seconds=timer.elapsed_seconds(), n_rows_fit=n_rows_fit,
        )
        print(f"  WARNING: average-by-age fit failed for {cohort}: {error}")
        return None, None


def fit_linear_baseline_logged(cohort, split_type, split_assignment, split_seed=SEED, maturity_age_min=MATURITY_AGE_MIN_DEFAULT, name_suffix=""):
    timer = RunTimer().start()
    hyperparameters = {"seed": SEED, "split_seed": split_seed, "maturity_age_min": maturity_age_min}
    attempt_id = write_started_marker(
        model_name="linear_baseline", cohort=cohort, split_type=split_type, run_phase="fit",
        is_test_run=False, device=DEVICE, hyperparameters=hyperparameters,
    )
    try:
        linear_train_df = load_train_rows(cohort, "linear_baseline", split_assignment, maturity_age_min=maturity_age_min)
        linear_params = fit_linear_baseline(linear_train_df)
        linear_output_path = output_dir(f"linear_baseline{name_suffix}", cohort, "params.json", split_type=split_type)
        save_linear_params(linear_params, cohort, len(linear_train_df), linear_output_path)
        print(f"  Linear baseline params saved -> {linear_output_path}")

        write_run_log(
            attempt_id=attempt_id,
            model_name="linear_baseline", cohort=cohort, split_type=split_type, run_phase="fit",
            status="success", is_test_run=False, device=DEVICE,
            hyperparameters=hyperparameters, metrics=None, error=None,
            output_dir=linear_output_path.parent, runtime_seconds=timer.elapsed_seconds(),
            n_rows_fit=len(linear_train_df),
        )
    except Exception as error:
        write_run_log(
            attempt_id=attempt_id,
            model_name="linear_baseline", cohort=cohort, split_type=split_type, run_phase="fit",
            status="failed", is_test_run=False, device=DEVICE,
            hyperparameters=hyperparameters, metrics=None, error=format_error(error),
            output_dir=None, runtime_seconds=timer.elapsed_seconds(),
        )
        print(f"  WARNING: linear baseline fit failed for {cohort}: {error}")


def fit_rf_baseline_logged(cohort, split_type, split_assignment, split_seed=SEED, maturity_age_min=MATURITY_AGE_MIN_DEFAULT, name_suffix=""):
    timer = RunTimer().start()
    hyperparameters = {"seed": SEED, "split_seed": split_seed, "maturity_age_min": maturity_age_min}
    attempt_id = write_started_marker(
        model_name="rf_baseline", cohort=cohort, split_type=split_type, run_phase="fit",
        is_test_run=False, device=DEVICE, hyperparameters=hyperparameters,
    )
    try:
        rf_train_df = load_train_rows(cohort, "rf_baseline", split_assignment, maturity_age_min=maturity_age_min)
        rf_model, rf_encoded_column_names, rf_feature_columns = fit_rf_baseline(rf_train_df)
        rf_output_dir = output_dir(f"rf_baseline{name_suffix}", cohort, split_type=split_type)
        rf_model_path = save_rf_model(
            rf_model, rf_encoded_column_names, cohort, len(rf_train_df), rf_output_dir,
            feature_columns=rf_feature_columns,
        )
        print(f"  RF baseline model saved -> {rf_model_path}")

        write_run_log(
            attempt_id=attempt_id,
            model_name="rf_baseline", cohort=cohort, split_type=split_type, run_phase="fit",
            status="success", is_test_run=False, device=DEVICE,
            hyperparameters=hyperparameters, metrics=None, error=None,
            output_dir=rf_output_dir, runtime_seconds=timer.elapsed_seconds(), n_rows_fit=len(rf_train_df),
        )
    except Exception as error:
        write_run_log(
            attempt_id=attempt_id,
            model_name="rf_baseline", cohort=cohort, split_type=split_type, run_phase="fit",
            status="failed", is_test_run=False, device=DEVICE,
            hyperparameters=hyperparameters, metrics=None, error=format_error(error),
            output_dir=None, runtime_seconds=timer.elapsed_seconds(),
        )
        print(f"  WARNING: RF baseline fit failed for {cohort}: {error}")


def run_for_cohort(
    cohort, split_type, split_seed=SEED, maturity_age_min=MATURITY_AGE_MIN_DEFAULT,
    k_folds=DEFAULT_K_FOLDS, held_out_fold=0,
):
    # name_suffix: "" only when split_seed/maturity_age_min are at their defaults AND split_type
    # isn't spatial_block_kfold -- every existing output then stays at its exact original plain
    # path, zero change. Each non-default piece appends its own tag ("_splitseed<N>",
    # "_agemin<N>", "_fold<i>"), same "only non-default gets a suffix" convention already used for
    # split_seed alone (2026-08-02) -- multiple can be non-default at once, the tags concatenate.
    # A fold tag is always added under spatial_block_kfold (never omitted), since held_out_fold=0
    # is still a genuinely different split from every other fold, not a "default" in the same
    # sense split_seed=SEED is.
    suffix_parts = []
    if split_seed != SEED:
        suffix_parts.append(f"_splitseed{split_seed}")
    if maturity_age_min != MATURITY_AGE_MIN_DEFAULT:
        suffix_parts.append(f"_agemin{maturity_age_min}")
    if split_type == "spatial_block_kfold":
        suffix_parts.append(f"_fold{held_out_fold}")
    name_suffix = "".join(suffix_parts)

    print(
        f"===== {cohort} ({split_type}, split_seed={split_seed}, maturity_age_min={maturity_age_min}"
        f"{f', fold={held_out_fold}/{k_folds}' if split_type == 'spatial_block_kfold' else ''}) ====="
    )
    filtered_df, split_assignment = build_split_for_cohort(
        cohort, split_type, split_seed=split_seed, maturity_age_min=maturity_age_min, name_suffix=name_suffix,
        k_folds=k_folds, held_out_fold=held_out_fold,
    )
    n_rows_fit = int((filtered_df["split"] == "train").sum())

    cr_train_df = filtered_df[filtered_df["split"] == "train"]
    avg_train_df = cr_train_df  # average-by-age uses the same age-only table as CR

    cr_params = fit_chapman_richards_logged(
        cohort, split_type, cr_train_df, n_rows_fit, split_seed=split_seed, maturity_age_min=maturity_age_min, name_suffix=name_suffix
    )
    lookup_table, fallback_mean_height = fit_average_by_age_logged(
        cohort, split_type, avg_train_df, n_rows_fit, split_seed=split_seed, maturity_age_min=maturity_age_min, name_suffix=name_suffix
    )
    fit_linear_baseline_logged(
        cohort, split_type, split_assignment, split_seed=split_seed, maturity_age_min=maturity_age_min, name_suffix=name_suffix
    )
    fit_rf_baseline_logged(
        cohort, split_type, split_assignment, split_seed=split_seed, maturity_age_min=maturity_age_min, name_suffix=name_suffix
    )

    print()
    return cr_params, lookup_table, fallback_mean_height


def print_chapman_richards_summary(results):
    # Printed with more decimals than usual so small differences between
    # cohorts (or against the prior dissertation's values) are not hidden by
    # rounding — the saved params.json files always keep full precision
    # regardless of how this is displayed.
    print("===== Chapman-Richards: fitted params vs prior dissertation =====")
    header = f"{'cohort':<10}{'y_max':>16}{'k':>16}{'p':>16}"
    print(header)
    print(
        f"{'prior':<10}"
        f"{PRIOR_CR_PARAMS['y_max']:>16.8f}"
        f"{PRIOR_CR_PARAMS['k']:>16.8f}"
        f"{PRIOR_CR_PARAMS['p']:>16.8f}"
    )
    for cohort in COHORTS:
        cr_params = results[cohort][0]
        if cr_params is None:
            print(f"{cohort:<10}  fit did not converge")
            continue
        print(
            f"{cohort:<10}"
            f"{cr_params['y_max']:>16.8f}"
            f"{cr_params['k']:>16.8f}"
            f"{cr_params['p']:>16.8f}"
        )
    print()


def print_average_by_age_summary(results):
    print("===== Average-by-age: lookup table summary =====")
    for cohort in COHORTS:
        lookup_table, fallback_mean_height = results[cohort][1], results[cohort][2]
        if lookup_table is None:
            print(f"{cohort}: fit failed, see outputs/run_logs/")
            continue
        ages_covered = sorted(lookup_table.keys())
        print(f"{cohort}:")
        print(f"  Ages covered: {ages_covered[0]} to {ages_covered[-1]} ({len(ages_covered)} distinct ages)")
        print(f"  Fallback mean height (used for ages not seen in training): {fallback_mean_height:.8f} m")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-type",
        choices=["plot_level", "spatial_block", "spatial_block_kfold", "temporal", "temporal_narrow_gap"],
        default="plot_level",
        help=(
            "plot_level (default, easy/established), spatial_block (harder, "
            "unseen-compartment test), spatial_block_kfold (rotates which compartments are held "
            "out across --n-folds folds -- use with --fold-index to fit a fold-matched CR anchor "
            "for pinn_env_terrain/pinn_env_terrain_k's k-fold runs), temporal (predict a real "
            "future survey, 11-year train-to-test gap), or temporal_narrow_gap (same idea, 2-year "
            "gap)."
        ),
    )
    parser.add_argument(
        "--n-folds", type=int, default=DEFAULT_K_FOLDS,
        help=f"Number of folds for --split-type spatial_block_kfold (default {DEFAULT_K_FOLDS}). "
             "Ignored for every other split type.",
    )
    parser.add_argument(
        "--fold-index", type=int, default=0,
        help="Which fold to hold out as test, for --split-type spatial_block_kfold (0-indexed, "
             "must be < --n-folds). Ignored for every other split type. Run once per fold "
             "(0 .. n-folds-1) to fit a CR anchor matching every fold of the DNN/PINN k-fold sweep.",
    )
    parser.add_argument(
        "--split-seed", type=int, default=SEED,
        help=f"Seed for spatial_block_split's/plot_level_split's own shuffle (default {SEED}, "
             "every existing baseline output). Non-default writes to a "
             "'_splitseed<N>'-suffixed path, never overwriting the primary outputs -- see "
             "documentation/experiment_log.md's 2026-08-02 split-seed robustness entries. "
             "pinn_noenv/pinn_env_terrain's load_cr_params() needs a matching CR anchor refit "
             "here before a non-default --split-seed on those models is a valid comparison.",
    )
    parser.add_argument(
        "--maturity-age-min", type=int, default=MATURITY_AGE_MIN_DEFAULT,
        help=f"Passed straight through to filter_data()'s maturity_age_min (default "
             f"{MATURITY_AGE_MIN_DEFAULT}, every existing baseline output). A plot whose Age at "
             "the 2023 survey is below this is dropped ENTIRELY (every one of its survey years, "
             "not just the young one) -- see models/common/data.py::filter_data(). Pass 0 to "
             "disable the gate and keep every plot regardless of maturity, to check whether this "
             "filter is helping or hurting model accuracy. Non-default writes to a "
             "'_agemin<N>'-suffixed path, never overwriting the primary outputs.",
    )
    args = parser.parse_args()

    if args.split_type == "spatial_block_kfold" and not (0 <= args.fold_index < args.n_folds):
        raise ValueError(f"--fold-index must be in [0, {args.n_folds}), got {args.fold_index}")

    results = {}
    for cohort in COHORTS:
        results[cohort] = run_for_cohort(
            cohort, args.split_type, split_seed=args.split_seed, maturity_age_min=args.maturity_age_min,
            k_folds=args.n_folds, held_out_fold=args.fold_index,
        )

    print_chapman_richards_summary(results)
    print_average_by_age_summary(results)


if __name__ == "__main__":
    main()

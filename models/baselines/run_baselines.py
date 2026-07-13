# Run as: python -m models.baselines.run_baselines
# Fits Chapman-Richards, average-by-age, linear regression, and random
# forest baselines for both cohorts. This pass only fits and saves
# parameters/models -- evaluation happens separately in evaluate_baselines.py.

from pathlib import Path

from models.average_by_age.average_by_age import fit as fit_average_by_age
from models.average_by_age.average_by_age import save_lookup
from models.chapman_richards.chapman_richards import fit as fit_chapman_richards
from models.chapman_richards.chapman_richards import save_params as save_cr_params
from models.common.data import filter_data, load_cohort_data, load_model_table
from models.common.splits import plot_level_split
from models.linear_baseline.linear_baseline import fit as fit_linear_baseline
from models.linear_baseline.linear_baseline import save_params as save_linear_params
from models.rf_baseline.rf_baseline import fit as fit_rf_baseline
from models.rf_baseline.rf_baseline import save_model as save_rf_model

PROJECT_ROOT = Path(__file__).resolve().parents[2]

COHORTS = ["4survey", "6survey"]
SEED = 42

# The prior dissertation's Chapman-Richards fit, for a quick sanity check.
PRIOR_CR_PARAMS = {"y_max": 46.1126, "k": 0.01866979, "p": 1.0175}


def build_split_for_cohort(cohort):
    # The split is computed ONCE per cohort, from the smallest shared table
    # (identification, LiDAR_year, blk, Age, yldc, Top_Height99 -- everything
    # Chapman-Richards and average-by-age need), then saved. linear_baseline
    # and rf_baseline reuse this exact split by merging onto it (in
    # load_train_rows), rather than recomputing plot_level_split()
    # separately -- that guarantees all four baselines share identical
    # train/val/test membership, even though they load different source
    # files with possibly different row orders.
    #
    # Note: cr_age.csv.gz itself has no yldc column, so this filtered table
    # (not a fresh read of cr_age.csv.gz) is what CR and average-by-age are
    # actually fitted on below.
    df = load_cohort_data(cohort)
    filtered_df = filter_data(df)

    # plot_level_split returns a three-way train/val/test split (60/20/20).
    # Neither Chapman-Richards, average-by-age, nor linear regression has
    # anything to tune (no hyperparameters, no early stopping), and the RF
    # baseline isn't tuned yet either, so val is saved here for schema
    # consistency with later models only -- it is not read by any model
    # fitted in this script. Only train is used for fitting.
    filtered_df["split"] = plot_level_split(filtered_df, seed=SEED)

    split_path = PROJECT_ROOT / "outputs" / "splits" / cohort / "split_assignment.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df[["identification", "LiDAR_year", "split"]].to_csv(split_path, index=False)
    print(f"  Saved split assignment -> {split_path}")

    val_row_count = (filtered_df["split"] == "val").sum()
    test_row_count = (filtered_df["split"] == "test").sum()
    train_row_count = (filtered_df["split"] == "train").sum()
    print(
        f"  Training rows: {train_row_count:,}  "
        f"(val rows saved but unused by these models: {val_row_count:,}; "
        f"test rows held out, not touched: {test_row_count:,})"
    )

    split_assignment = filtered_df[["identification", "LiDAR_year", "split"]]
    return filtered_df, split_assignment


def load_train_rows(cohort, table_name, split_assignment):
    # Load one model's own full table, apply the same filters, then attach
    # the shared split by merging on identification + LiDAR_year (not by
    # recomputing the split).
    table = load_model_table(cohort, table_name)
    filtered_table = filter_data(table)

    merged = filtered_table.merge(split_assignment, on=["identification", "LiDAR_year"], how="inner")
    assert len(merged) == len(filtered_table), (
        f"Row count changed after merging {table_name} with the shared split assignment -- "
        "the source tables may no longer agree on which rows survive filtering."
    )

    return merged[merged["split"] == "train"]


def run_for_cohort(cohort):
    print(f"===== {cohort} =====")
    filtered_df, split_assignment = build_split_for_cohort(cohort)
    n_rows_fit = int((filtered_df["split"] == "train").sum())

    cr_train_df = filtered_df[filtered_df["split"] == "train"]
    avg_train_df = cr_train_df  # average-by-age uses the same age-only table as CR

    # --- Chapman-Richards ---
    cr_params = None
    try:
        cr_params = fit_chapman_richards(cr_train_df)
    except RuntimeError as error:
        print(f"  WARNING: Chapman-Richards fit did not converge for {cohort}: {error}")

    if cr_params is not None:
        cr_output_path = PROJECT_ROOT / "outputs" / "chapman_richards" / cohort / "params.json"
        save_cr_params(cr_params, cohort, n_rows_fit, cr_output_path)
        print(f"  Chapman-Richards params saved -> {cr_output_path}")

        if cr_params["y_max"] <= 0 or cr_params["k"] <= 0 or cr_params["p"] <= 0:
            print("  WARNING: a fitted Chapman-Richards parameter is zero or negative — check this cohort's fit.")

        max_observed_height = cr_train_df["Top_Height99"].max()
        if cr_params["y_max"] < max_observed_height:
            print(
                f"  WARNING: fitted y_max ({cr_params['y_max']:.2f} m) is below the "
                f"max observed training height ({max_observed_height:.2f} m) — "
                "this fit looks unstable, not just a low asymptote."
            )

    # --- Average-by-age ---
    lookup_table, fallback_mean_height = fit_average_by_age(avg_train_df)
    avg_output_path = PROJECT_ROOT / "outputs" / "average_by_age" / cohort / "lookup.json"
    save_lookup(lookup_table, fallback_mean_height, cohort, n_rows_fit, avg_output_path)
    print(f"  Average-by-age lookup saved -> {avg_output_path}")

    # --- Linear baseline ---
    linear_train_df = load_train_rows(cohort, "linear_baseline", split_assignment)
    linear_params = fit_linear_baseline(linear_train_df)
    linear_output_path = PROJECT_ROOT / "outputs" / "linear_baseline" / cohort / "params.json"
    save_linear_params(linear_params, cohort, len(linear_train_df), linear_output_path)
    print(f"  Linear baseline params saved -> {linear_output_path}")

    # --- RF baseline ---
    rf_train_df = load_train_rows(cohort, "rf_baseline", split_assignment)
    rf_model = fit_rf_baseline(rf_train_df)
    rf_output_dir = PROJECT_ROOT / "outputs" / "rf_baseline" / cohort
    rf_model_path = save_rf_model(rf_model, cohort, len(rf_train_df), rf_output_dir)
    print(f"  RF baseline model saved -> {rf_model_path}")

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
        ages_covered = sorted(lookup_table.keys())
        print(f"{cohort}:")
        print(f"  Ages covered: {ages_covered[0]} to {ages_covered[-1]} ({len(ages_covered)} distinct ages)")
        print(f"  Fallback mean height (used for ages not seen in training): {fallback_mean_height:.8f} m")
    print()


def main():
    results = {}
    for cohort in COHORTS:
        results[cohort] = run_for_cohort(cohort)

    print_chapman_richards_summary(results)
    print_average_by_age_summary(results)


if __name__ == "__main__":
    main()

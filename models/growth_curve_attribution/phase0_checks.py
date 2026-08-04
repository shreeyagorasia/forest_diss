# Purpose: the four cheap, no-model-fitting checks agreed before committing engineering time to
# any Stage 2 architecture (see documentation/experiment_log.md's Stage 2 planning notes):
#   1. How many distinct survey years does each plot actually have -- gates whether a per-plot
#      curve fit (Candidate A) or per-plot random effect (Candidate B) is even identifiable.
#   2. Does the deviation from the yldc curve have real, non-degenerate spread -- if it's tiny
#      and unstructured, that already answers "is there anything here to explain" cheaply.
#   3. Does a big deviation line up with a recorded thinning event -- a real confound to rule
#      out before crediting anything to environment/terrain.
#   4. Would a distance-based Stage 2 model (GNNWR/GRF/GP-kriging) even have legitimate
#      train-set neighbours to work with under spatial_block_split -- this project already found
#      the same check kills a fixed-radius neighbour FEATURE (2026-08-02 log entry); re-running
#      it here checks whether the same failure mode is likely for a distance-based MODEL too,
#      before any of them get built.
#
# Plain functions only, no plotting here -- this project's own convention (see
# models/spatial_attribution/nlme.py's header) is to keep plotting logic in the calling
# notebook/script, real logic in the module.

from scipy.spatial import cKDTree

from models.common.splits import SPATIAL_BLOCK_COL, SPATIAL_BUFFER_METRES, SPLIT_SEED, spatial_block_split


def distinct_timestamp_counts(df):
    # counts_per_plot: one row per plot, how many distinct survey years it has real data for.
    # summary: how many plots have exactly 1, 2, 3, ... distinct years -- the actual identifiability
    # picture (a 1-timestamp plot can still get an algebraic/GADA-style y_max, but not a real
    # least-squares fit; more timestamps make that fit more trustworthy, not just possible).
    counts_per_plot = df.groupby("identification")["LiDAR_year"].nunique()
    summary = counts_per_plot.value_counts().sort_index()
    return counts_per_plot, summary


def yldc_deviation_summary(df):
    # deviation = observed Top_Height95 minus the plot's OWN yldc-implied curve prediction at
    # that row's Age -- both columns already computed once by
    # data_processing/export_growth_curve_tables.py, not recomputed here.
    valid = df.dropna(subset=["top_height95_yldc_predicted"]).copy()
    valid["yldc_deviation"] = valid["Top_Height95"] - valid["top_height95_yldc_predicted"]

    overall_stats = {
        "n_rows": len(valid),
        "mean": valid["yldc_deviation"].mean(),
        "std": valid["yldc_deviation"].std(),
        "skew": valid["yldc_deviation"].skew(),
        "excess_kurtosis": valid["yldc_deviation"].kurtosis(),
    }

    # age_band_5yr already exists in the growth_curve_table (built by
    # data_processing/clean_master_data.py's add_export_metrics()) -- reused here to see WHERE
    # the deviation's spread comes from (e.g. a suspiciously wide/biased young-age band would
    # point at the known low-Age numerical instability in this formula, not a real site effect).
    by_age_band = valid.groupby("age_band_5yr")["yldc_deviation"].agg(["count", "mean", "std"])

    return valid, overall_stats, by_age_band


def thinning_confound_check(valid_with_deviation):
    # Cross-checks whether large yldc deviations line up with a RECORDED thinning event, rather
    # than being credited to environment by default -- same confounder-checking discipline
    # already used elsewhere in this project (env_deviation, the Stage 1 confound checks).
    return valid_with_deviation.groupby("thinning_status")["yldc_deviation"].agg(["count", "mean", "std"])


def neighbour_coverage_check(df, radius_metres=75, seed=None):
    # Re-runs the same shape of check that already found (2026-08-02 log entry) that a
    # fixed-radius neighbour FEATURE is structurally incompatible with spatial_block_split's
    # whole-compartment holdout -- here applied to whether a distance-based Stage 2 MODEL would
    # face the same problem, before any of them (GNNWR/GRF/GP-kriging) get built.
    seed = seed if seed is not None else SPLIT_SEED

    one_row_per_plot = df.drop_duplicates("identification").copy()
    one_row_per_plot["split"] = spatial_block_split(
        one_row_per_plot,
        block_col=SPATIAL_BLOCK_COL,
        buffer_distance=SPATIAL_BUFFER_METRES,
        seed=seed,
    )

    train_coordinates = one_row_per_plot.loc[one_row_per_plot["split"] == "train", ["x", "y"]].values
    test_coordinates = one_row_per_plot.loc[one_row_per_plot["split"] == "test", ["x", "y"]].values

    if len(train_coordinates) == 0 or len(test_coordinates) == 0:
        raise ValueError("No train or test plots after spatial_block_split -- check the split ran correctly.")

    train_tree = cKDTree(train_coordinates)
    distances_to_nearest_train_plot, _ = train_tree.query(test_coordinates)

    fraction_with_no_train_neighbour = float((distances_to_nearest_train_plot > radius_metres).mean())
    mean_distance = float(distances_to_nearest_train_plot.mean())

    return {
        "n_test_plots": len(test_coordinates),
        "radius_metres": radius_metres,
        "fraction_of_test_plots_with_no_train_neighbour_within_radius": fraction_with_no_train_neighbour,
        "mean_distance_to_nearest_train_plot_metres": mean_distance,
    }

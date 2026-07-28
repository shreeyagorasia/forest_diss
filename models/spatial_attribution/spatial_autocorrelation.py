    # Purpose: Test whether the CR residuals are spatially clustered, or just random noise.
# Key logic: two tools, meant to be run TOGETHER, not independently. Run
# semivariogram_range() first -- it scans a whole range of distances and finds the one where
# the spatial pattern actually levels off. Then feed that distance into global_morans_i() as
# its `distance` argument. Picking a neighbour-count (like "8 nearest plots") instead of a real
# physical distance is a trap here: plots sit on a dense 20m/40m grid, so a small neighbour
# count only reaches a few tens of metres -- testing "is this plot like the plot right next to
# it", not "is there real regional clustering across the forest". Using an actual distance in
# metres avoids that trap.

import warnings

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree


def global_morans_i(x, y, values, distance, sample_size=5000, seed=42):
    # Moran's I answers one question: are plots within `distance` of each other more similar
    # to each other than far-apart plots are, more than random chance alone would explain? A
    # value near 0 means no real pattern (random noise). A value clearly above 0 (with a small
    # p-value) means real clustering exists -- similar residuals really do sit near each other.
    #
    # `distance` should come from semivariogram_range()'s result, run on the same data first --
    # not guessed. See the note at the top of this file for why.
    import esda
    import libpysal

    rng = np.random.default_rng(seed)
    mask = ~np.isnan(values)
    x, y, values = np.asarray(x)[mask], np.asarray(y)[mask], np.asarray(values)[mask]

    # Moran's I gets slow on very large point sets, so a random sample keeps this quick. 5,000
    # points is already far more than enough to detect real spatial structure if it's there.
    if len(values) > sample_size:
        sample_index = rng.choice(len(values), size=sample_size, replace=False)
        x, y, values = x[sample_index], y[sample_index], values[sample_index]

    coordinates = np.column_stack([x, y])

    # Distance-band weights: every plot within `distance` metres counts as a neighbour, all
    # weighted equally. Unlike a fixed neighbour count, this directly controls the physical
    # scale being tested, no matter how densely-packed plots happen to be in a given area.
    with warnings.catch_warnings():
        # Aberfoyle is genuinely several separate forest blocks, not one connected shape (see
        # the boundary work in spatial_viz_comparison_scratch.ipynb), and a plot with no other
        # plot within `distance` of it has no neighbours at all -- both are expected here, not
        # a bug, so this specific warning is silenced on purpose.
        warnings.simplefilter("ignore", category=UserWarning)
        weights = libpysal.weights.DistanceBand.from_array(coordinates, threshold=distance, binary=True)
    weights.transform = "r"

    moran_result = esda.moran.Moran(values, weights, permutations=199)
    return moran_result.I, moran_result.p_sim


def semivariogram_range(x, y, values, max_distance=5000, n_bins=15, sample_size=5000, seed=42):
    # A semivariogram asks a different, more detailed question than Moran's I: not just "is
    # there spatial structure", but "how far does it reach". It measures how different two
    # plots' residuals tend to be, as a function of the distance between them, then fits a
    # curve to that relationship. Three numbers come out of that curve:
    #   - nugget:  how different two plots are even at distance ~0 (this is "noise" -- error
    #              that isn't explained by location at all, e.g. measurement error)
    #   - sill:    how different two plots are once they're far enough apart to be unrelated
    #   - range:   the distance at which the curve levels off from nugget to sill -- this is
    #              the answer to "how far does the spatial pattern actually reach"
    rng = np.random.default_rng(seed)
    mask = ~np.isnan(values)
    x, y, values = np.asarray(x)[mask], np.asarray(y)[mask], np.asarray(values)[mask]

    if len(values) > sample_size:
        sample_index = rng.choice(len(values), size=sample_size, replace=False)
        x, y, values = x[sample_index], y[sample_index], values[sample_index]

    coordinates = np.column_stack([x, y])
    tree = cKDTree(coordinates)

    # Every pair of plots within max_distance of each other, found efficiently with a KD-tree
    # instead of comparing every plot against every other plot (which would be far too slow).
    pairs = tree.query_pairs(r=max_distance, output_type="ndarray")
    if len(pairs) == 0:
        return np.nan, "no_pairs"

    pair_distances = np.linalg.norm(coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]], axis=1)
    squared_differences = (values[pairs[:, 0]] - values[pairs[:, 1]]) ** 2

    # Group pairs into distance bins (e.g. "0-333m apart", "333-666m apart", ...), then average
    # the squared difference within each bin -- this is the standard semivariogram calculation.
    bin_edges = np.linspace(0, max_distance, n_bins + 1)
    bin_index = np.digitize(pair_distances, bin_edges) - 1

    bin_centres = []
    semivariance = []
    for bin_number in range(n_bins):
        in_this_bin = bin_index == bin_number
        if in_this_bin.sum() < 30:  # too few pairs in this bin to trust its average
            continue
        bin_centres.append((bin_edges[bin_number] + bin_edges[bin_number + 1]) / 2)
        semivariance.append(0.5 * squared_differences[in_this_bin].mean())

    bin_centres = np.array(bin_centres)
    semivariance = np.array(semivariance)
    if len(bin_centres) < 4:
        return np.nan, "insufficient_bins"

    # Two cases where fitting a curve doesn't make sense, checked before trying:
    #   - the values are already flat from the very first bin (no rise at all -> no real
    #     spatial structure at any distance tested)
    #   - the values are still rising at the last bin (the real range is bigger than
    #     max_distance, so this window wasn't wide enough to see it level off)
    nugget_estimate = semivariance[0]
    sill_estimate = semivariance[-3:].mean()
    if sill_estimate <= nugget_estimate * 1.3:
        return 0.0, "no_structure"

    def exponential_model(distance, nugget, sill, range_param):
        return nugget + sill * (1 - np.exp(-distance / range_param))

    try:
        fitted_params, _ = curve_fit(
            exponential_model, bin_centres, semivariance,
            p0=[semivariance.min(), semivariance.max() - semivariance.min(), max_distance / 3],
            bounds=([0, 0, 1], [semivariance.max() * 2, semivariance.max() * 3, max_distance * 5]),
            maxfev=5000,
        )
        # The "practical range" (where the curve is 95% of the way to the sill) is a standard
        # geostatistics convention -- about 3x the fitted range_param for this exponential shape.
        practical_range = 3 * fitted_params[2]
        if practical_range >= max_distance:
            return max_distance, "exceeds_window"
        return practical_range, "resolved"
    except RuntimeError:
        return np.nan, "fit_failed"

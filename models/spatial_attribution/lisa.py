# Purpose: Find WHICH specific plots are part of a real, statistically significant spatial
# cluster. Not just "is there clustering somewhere" (that's what global Moran's I already
# answered), but "which plots, exactly".
# Key logic: esda already has a ready-made "Local Moran's I" tool for this. Unlike the plain
# Moran's I and semivariogram earlier, there's no need to hand-write this one, it's a direct,
# well-tested library call. Two things are different here from the global test, on purpose:
#   1. Neighbours are picked by k-nearest-neighbours, not a fixed distance. A fixed distance
#      (like the semivariogram's 3,956m) gives wildly different neighbour COUNTS depending on
#      local plot density. Checked directly on this data: 681 to 7,141 neighbours per plot at
#      3,956m. K-nearest-neighbours instead guarantees every plot is compared against the same
#      NUMBER of neighbours, adapting automatically to how dense or sparse that area is.
#   2. Significance is corrected for running thousands of tests at once (one per plot), using
#      the Benjamini-Hochberg false discovery rate method. Without this, roughly 5% of all
#      plots would show up as "significant" purely by chance, even if there were no real local
#      clusters anywhere at all.

import warnings

import esda
import libpysal
import numpy as np
from statsmodels.stats.multitest import multipletests

# esda's quadrant codes (1-4) mean the same thing every time, so they're translated into plain
# English labels once here, rather than every caller having to remember what "3" means.
QUADRANT_LABELS = {
    1: "High-High",  # this plot AND its neighbours are both above average. A real hot-spot
    2: "Low-High",   # this plot is below average, but its neighbours are above. An outlier
    3: "Low-Low",    # this plot AND its neighbours are both below average. A real cold-spot
    4: "High-Low",   # this plot is above average, but its neighbours are below. An outlier
}


def local_morans_i(x, y, values, k, significance_level=0.05, permutations=999, seed=42):
    # Returns one label per plot: "High-High", "Low-Low", "High-Low", "Low-High", or
    # "Not significant". Only the first four mean a real, statistically confirmed local
    # pattern. "Not significant" means this specific plot's local neighbourhood isn't
    # distinguishable from random, even accounting for how many plots were tested at once.
    mask = ~np.isnan(values)
    x, y, values = np.asarray(x)[mask], np.asarray(y)[mask], np.asarray(values)[mask]
    coordinates = np.column_stack([x, y])

    with warnings.catch_warnings():
        # Aberfoyle is genuinely several separate forest blocks. A nearest-neighbour graph
        # over disconnected blocks is expected here, not a bug.
        warnings.simplefilter("ignore", category=UserWarning)
        weights = libpysal.weights.KNN.from_array(coordinates, k=k)
    weights.transform = "r"

    local_result = esda.moran.Moran_Local(values, weights, permutations=permutations, seed=seed)

    # Benjamini-Hochberg correction: sorts all the p-values and compares each one against a
    # rank-scaled threshold instead of the same fixed 0.05 for every single plot. This is
    # what keeps thousands of simultaneous tests from producing a flood of false positives.
    is_significant, _, _, _ = multipletests(local_result.p_sim, alpha=significance_level, method="fdr_bh")

    labels = []
    for quadrant_code, significant in zip(local_result.q, is_significant):
        if significant:
            labels.append(QUADRANT_LABELS[quadrant_code])
        else:
            labels.append("Not significant")

    return np.array(labels), local_result.Is, local_result.p_sim

# Progress Notes

A running log of what's been built, key decisions and why, and loose ideas worth revisiting.
Separate from `plans_md/Dissertation Plan v5 - 7th June.md`, which is the formal
chapter-by-chapter plan — this is the informal working log.

***

## Repo structure (as of 13 July 2026)

```
models/
├── common/            # metrics.py, splits.py, data.py, geo.py, plotting.py, saving.py
├── chapman_richards/
├── average_by_age/
├── linear_baseline/
├── rf_baseline/
└── baselines/         # run_baselines.py + evaluate_baselines.py orchestrate all four

data_processing/
└── export_model_tables.py   # standalone, no notebook dependency (see below)

outputs/<model_name>/<cohort>/   # gitignored, regenerate via models/baselines/
results_notebooks/
└── baseline_results.ipynb       # reads outputs/ only, never refits

data_exploration_gpkg/notebooks/
├── lidar_years_all_data_cleaning.ipynb        # the cleaning funnel, still notebook-only
└── spatial_temporal_split_visualisation.ipynb # maps/visualises the three split types
```

## Data pipeline: two steps, notebook-independent after step 1

1. **Cleaning notebook** (`EXPORT_FILES = True`) — the actual cleaning funnel (species filter,
   age/height validity, deduplication, cohort balancing) runs here and writes `data/processed/master/`.
   This is the only step that needs the notebook.
2. **`python -m data_processing.export_model_tables`** — reads `master/`, derives every
   per-model table (`current_state/`) and the transition tables. Pure column selection on
   already-cleaned data, so it works even if the notebook is mid-edit or broken.

Both steps write **Parquet**, not CSV — CSV was round-tripping booleans as `"True"`/`"False"`
strings and losing categorical dtype fidelity. Only `predictions.csv`, `split_assignment.csv`,
and `plot_coordinates.csv.gz` stay CSV (small, human-facing files).

**Target**: `Top_Height99` primary, `Top_Height95` fallback — confirmed against the cleaning
notebook's own section 11.2 decision, despite an earlier exploratory finding that
`Top_Height99 < Top_Height95` in ~64% of rows (known, unresolved caveat, not a blocker).

## Baseline models: all four fitted and evaluated

`plot_level_split` (60/20/20), filters `Age >= 20` + yield class 2–50, no environmental features yet.

```
4survey: CR 68.5% acc → avg-by-age 69.3% → linear 77.1% → RF 81.6%
6survey: CR 83.9% acc → avg-by-age 85.6% → linear 83.1% → RF 86.9%
```

- RF wins on every metric (expected — it's the most flexible of the four).
- Linear regression has a large one-signed bias in the 80+ age band (extrapolation failure).
- CR's fitted `y_max` is bound-constrained (`>= max observed training height`) — an earlier
  unconstrained fit landed *below* the observed max, which is physically impossible for an
  asymptote. Fixed by widening `curve_fit` bounds and trying multiple starting guesses.
- CR shows a large bias in the oldest age bands too (6survey 60-80yr: bias ≈ -11.2m) — treated as
  a genuine limitation of a 3-parameter curve with a thin old-growth sample, not a bug. Not a
  reason to scrub old-age data from the master dataset, since a more flexible future model might
  handle that range fine — flagged via age-banded metrics instead of filtered away.
- No loss-function/training-curve plots for any of these four — none use gradient descent
  (CR/linear are single least-squares solves, avg-by-age is a groupby-mean, RF grows trees by
  greedy variance reduction). Only relevant once DNN/PINN work starts.

## Split infrastructure: built and tested, only one wired in yet

- `plot_level_split()` — used by all four baselines above.
- `spatial_block_split()` — compartment (`cpmt`, not `blk`) based, size-aware block assignment.
  `buffer_distance` is **asymmetric**: only excludes train plots near a val/test boundary, val/test
  are never touched (they always keep every plot the block assignment gave them, regardless of
  compartment shape). Current default 60m (3× the ~20m grid cell width — 50m was an awkward 2.5×,
  100m excluded 2-3x more data for the same close-range leakage protection). Verified
  programmatically (KDTree nearest-neighbour re-check after buffering), not just asserted.
- `temporal_split()` — year-based, ignores location entirely on purpose (a different question to
  the spatial split — kept separate so a spatial-vs-temporal failure can't be conflated).

Neither `spatial_block_split` nor `temporal_split` is wired into a model yet — no model uses
terrain/wind features or does temporal generalisation testing yet. Visualised in
`spatial_temporal_split_visualisation.ipynb`.

***

## Ideas worth revisiting

### Neighbouring-plot canopy cover as a shelter/exposure proxy

Raised as a fallback in case WASP/OS Terrain 50/Global Wind Atlas is delayed or insufficient: use
the same centroid + KDTree infrastructure already built (`models/common/geo.py`) to compute, for
each plot, an aggregate of *neighbouring* plots' `CanopyCover` within some radius — a cheap
"open edge vs. sheltered mid-stand" signal derivable from data already on hand, no external
terrain/wind source needed.

Two things to watch if this gets built:
- **Leakage risk** — a neighbour-derived feature pulls information across the plot boundary the
  same way a raw measurement would. Any such feature must only aggregate same-split neighbours,
  or it reopens exactly the leakage `buffer_distance` exists to prevent.
- **It's not a direct exposure measurement** — canopy cover is downstream of the same growth
  process being predicted (see the plan's DAG: LAI/canopy cover/volume are consequences of height,
  not causes), so it's weaker evidence than an actual terrain/wind covariate.

Not implemented. Candidate feature for later baseline/PINN work if terrain/wind data is delayed.

### Age-filter investigation — explicitly deprioritized, not a pending TODO

The cleaning funnel's "plausible age" stage exists but is currently a no-op
(`AGE_MIN, AGE_MAX = 0, 200`), and the height check only validates `Top_Height95`, not
`Top_Height99` (empirically fine right now — max `Top_Height99` is 53.5m/46.5m, both under 60 —
but not formally enforced for the actual target column).

A candidate rule was discussed (keep a plot only if it's ≥30 years old by 2023, i.e.
`plyr <= 1993`) and its impact computed against the real data:

| | 4survey | 6survey |
|---|---|---|
| Plots removed | 13,654 / 71,766 (19.0%) | 128 / 13,897 (0.9%) |
| Rows removed | 54,616 | 768 |
| Min age in earliest survey year after the cut | 15 (2008) | 9 (2002) |

**Decided not to implement this now.** If revisited: open questions were (a) the exact
biological/data-quality justification for each bound — worth asking an expert rather than picking
a round number, (b) whether it belongs in the master cleaning funnel (permanent, affects every
model) or as a per-model filter like the existing `Age >= 20`, (c) implement in the notebook or as
a new `data_processing/` script.

### RF model artifacts and Git LFS

Decided **not** to use Git LFS for `rf_baseline`'s `model.joblib` (can be 300MB–1.3GB+ with
sklearn's default unlimited tree depth). It's fast and deterministic to regenerate
(`models/baselines/run_baselines.py`, fixed seeds throughout), nothing downstream depends on the
exact binary (only its `metrics.json`/`predictions.csv`, which are tiny and tracked), and GitHub's
free LFS tier (1GB storage/bandwidth) would already be exceeded by these two files alone. `outputs/`
is gitignored entirely for this reason — regenerate on demand, don't try to version-control fitted
model binaries.

***

## What's not started yet

DNN, PINN (any version), terrain/wind feature extraction, XGBoost + SHAP, spatial/temporal
generalisation testing (the two split types built but unused above).

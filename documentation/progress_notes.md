# Progress Notes

A running log of what's been built, key decisions and why, and loose ideas worth revisiting.
Separate from `plans_md/Dissertation Plan v5 - 7th June.md`, which is the formal
chapter-by-chapter plan — this is the informal working log.

***

## Repo structure (as of 15 July 2026)

```
models/
├── common/            # metrics.py, splits.py, data.py, geo.py, plotting.py, saving.py,
│                       # torch_data.py, torch_model.py, run_logging.py
├── chapman_richards/
├── average_by_age/
├── linear_baseline/
├── rf_baseline/
├── baselines/          # run_baselines.py + evaluate_baselines.py orchestrate all four sklearn/CR models
├── dnn_noenv/           # dnn_noenv.py (network+fit/predict) + run_dnn_noenv.py (FIT, cluster)
│                        # + evaluate_dnn_noenv.py (EVALUATE, local)
├── pinn_noenv/          # pinn_noenv.py (network+physics losses+fit/predict) + run_pinn_noenv.py (FIT)
│                        # + evaluate_pinn_noenv.py (EVALUATE, local)
├── dnn_env_terrain/     # scaffolded 21 July 2026, empty __init__.py only -- not implemented yet,
│                        # blocked on the terrain/wind feature extraction step below
└── pinn_env_terrain/    # scaffolded 21 July 2026, same status as dnn_env_terrain/

data_processing/
└── export_model_tables.py   # standalone, no notebook dependency (see below)

outputs/
├── <model_name>/<cohort>/   # gitignored, regenerate by running the model
└── run_logs/                # gitignored, one JSON per run attempt — see "Run logging" below

notebooks/    # restructured 22 July 2026 from three separate top-level notebook folders
              # (data_exploration_gpkg/notebooks/, environmental_data_exploration/,
              # results_notebooks/) into one tree, grouped by purpose rather than by when the
              # notebook was created. Filenames unchanged, only their folder moved -- any doc
              # or notebook comment still citing the old paths is stale.
├── data_exploration/
│   ├── lidar_years_all_data_cleaning.ipynb        # the cleaning funnel, still notebook-only
│   └── lidar_years_all_data_understanding.ipynb
├── environmental_data/
│   ├── environmental_data_sources_survey.ipynb   # Tier 1 data-source survey, see section below
│   ├── aux_data_resolution_check.ipynb           # empirical re-check of every source (real
│   │                                              # extraction + statistical screen), see below
│   └── figures/   # aux_data_resolution_check_results.csv lives here now
├── spatial_analysis/
│   ├── spatial_temporal_split_visualisation.ipynb # maps/visualises the three split types
│   └── spatial_residual_autocorrelation.ipynb     # PLANNED, not built yet -- Moran's I / LISA
│                                                   # step of the spatial-question plan below
├── model_results/
│   ├── baseline_results.ipynb       # reads outputs/ only, never refits — sklearn/CR baselines
│   │                                 # only, covers all three split types
│   └── baseline_models_parameter_tuning.ipynb
├── scratch/
│   └── spatial_viz_comparison_scratch.ipynb   # gitignored — throwaway, re-grows large on
│                                               # every re-run, see charting decision below
└── archive/    # empty for now -- for notebooks that get superseded later, none yet

jobs/                  # SLURM submission scripts (user-maintained)
├── baselines/run_baselines.sh
├── baselines/evaluate_baselines.sh
├── dnn_noenv/run_dnn_noenv.sh
├── dnn_noenv/evaluate_dnn_noenv.sh
├── pinn_noenv/run_pinn_noenv.sh
├── pinn_noenv/evaluate_pinn_noenv.sh
└── data_processing/export_model_tables.sh

documentation/
├── progress_notes.md      # this file — informal running log, status, runbook, checklist
├── experiment_log.md      # formal experiment table: what's been run, what's planned, why
└── model_instructions/    # per-model-family setup/usage notes
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

All four sklearn/CR baselines have now been run under all three split types (see
`baseline_results.ipynb` section 10 for the plot_level vs spatial_block vs temporal comparison).
DNN/PINN can run under either `temporal_split` or `spatial_block_split` (both wired into
`models/common/torch_data.py::load_split_table()` as of 16 July 2026) — see "DNN and PINN
(no-environment)" below. Note: `spatial_block_split`, not `temporal_split`, is the split that
actually matches the dissertation's central research question (spatial/environmental
attribution) — `temporal_split` was run first because it happened to be the split most useful for
proving the physics-loss machinery works at all (see `documentation/experiment_log.md`'s Findings
log), not because it's the primary result. Visualised in
`spatial_temporal_split_visualisation.ipynb`.

***

## DNN and PINN (no-environment)

Both run under either `temporal_split` (train on early years → early-stop on 2021 → held-out test
on 2023) or `spatial_block_split` (whole held-in compartments → held-out val/test compartments),
selected via `--split-type` on the fit/evaluate scripts (defaults to `temporal`). Age +
thinning-status features only, no terrain/wind covariates yet (matches the baselines' current
feature set). Shared architecture: `NoEnvNetwork`, 3 hidden layers, 128 neurons, LeakyReLU
(`models/common/torch_model.py`).

The PINN's trajectory-consistency loss needs pairs of consecutive-survey rows of the same plot
that are both in the training split — under `temporal_split` that meant "both years in
`train_years`", but under `spatial_block_split` a whole plot (all its survey years) moves to
train/val/test together, so the correct general rule is "both endpoints are themselves labelled
`train`" (`models/common/torch_data.py::load_trajectory_pairs()`) — this one rule is correct under
either split type without branching on which one is active.

- **DNN** (`models/dnn_noenv/`) — plain regression, MSE loss only.
- **PINN** (`models/pinn_noenv/`) — MSE loss plus two Chapman-Richards physics terms: a
  derivative-consistency loss (instantaneous growth rate vs CR's derivative, via
  `torch.autograd.grad`) and a trajectory-consistency loss (finite-difference growth across real
  consecutive-survey pairs of the same plot vs CR's implied growth). CR's `y_max`/`k`/`p` are read
  from `outputs/chapman_richards/<cohort>/params.json` and treated as **frozen constants** — never
  refit inside the PINN.

**Each model is split into a FIT script and an EVALUATE script**, mirroring the existing
`run_baselines.py` / `evaluate_baselines.py` pattern:

- `run_dnn_noenv.py` / `run_pinn_noenv.py` — trains only. Never touches the test split (2023) or
  computes accuracy metrics. Meant to run on the SLURM cluster (GPU). Saves checkpoints, scalers,
  `training_history.csv` (per-epoch loss), `run_metadata.json`. Prints progress every 10 epochs.
- `evaluate_dnn_noenv.py` / `evaluate_pinn_noenv.py` — loads the checkpoint the fit script saved,
  makes predictions on the test split only, computes real MAE/RMSE/R2/Bias, saves
  `predictions.csv` + `metrics.json`. Cheap CPU forward pass — meant to run locally after copying
  the checkpoint down from the cluster (`rsync`).

This split exists because training happens on the cluster (GPU, minutes-to-hours) but computing
test metrics is a single cheap forward pass — no reason to wait for/pay for cluster time twice, or
to touch the test split from a script that might get re-run mid-experimentation.

**`--seed`/`--run-name`, both scripts, both models (as of 20 July 2026)**: `run_dnn_noenv.py`
didn't have `--run-name` until now — only `run_pinn_noenv.py` did (added for the physics-weight
sweep). Added it to DNN too, same handling as PINN's: `--run-name` only changes where results are
saved, never which data table is loaded, so a reseed (`--seed 43 --run-name dnn_noenv_seed43`)
never overwrites the primary seed-42 checkpoint. This was blocking a DNN reseed check — see
"What's not started yet" below.

**`grad_norm` now logged per epoch** (both models' `training_history.csv`, as of 20 July 2026):
the pre-clip value `clip_grad_norm_()` already computed internally but never saved. Free diagnostic
on any future run; not retroactive on existing checkpoints, so older `training_history.csv` files
won't have this column.

## Run logging (`models/common/run_logging.py`)

Every fit/evaluate script (baselines included) writes one JSON file per run attempt to
`outputs/run_logs/` (gitignored — these are meant to be pruned locally, not versioned). Two log
entries share an `attempt_id`: a `"started"` entry written *before* any real work happens, and a
`"success"`/`"failed"` entry written at the end. A `"started"` entry with no matching
completion entry means the job died without a normal Python exception (SLURM OOM-kill, walltime
limit, node crash) — that gap is itself the signal.

Each entry records: timestamp, model/cohort/split_type/run_phase, status, `is_test_run` (true
when `max_epochs` is below `TEST_RUN_MAX_EPOCHS_THRESHOLD` — auto-tagged, no manual flag needed),
device (cpu/mps/cuda), hostname, SLURM job id/name/nodelist (if running under SLURM), runtime,
git commit, package versions, hyperparameters, metrics, and full error traceback on failure.

**Important caveat**: `is_test_run` only describes the *fit* phase. An `evaluate` log entry has no
way of knowing whether the checkpoint it evaluated came from a real 500-epoch run or a 2-epoch
sanity check — check the corresponding `fit` entry (same cohort/model, look at `hyperparameters.
max_epochs`) before trusting an evaluate result as real.

***

## Runbook: what to run, in what order

1. **Push cleaned data to the cluster** (`data/processed/` and `data/interim/` are both
   gitignored, so neither travels via git):
   - `rsync` `data/processed/master/*.parquet` to the cluster, then on the cluster run
     `python -m data_processing.export_model_tables` to regenerate `data/processed/current_state/`
     there (proven byte-for-byte deterministic — safe to regenerate on a different machine).
   - `rsync` `data/interim/plot_coordinates.csv.gz` directly (small, ~3MB, no need to regenerate --
     it doesn't even need `geopandas` on the cluster). Only needed for `spatial_block_split`
     specifically — `temporal_split` never touches plot coordinates, which is why this was missed
     the first time `spatial_block_split` was wired into DNN/PINN: a job failing with
     `No such file or directory: .../data/interim/plot_coordinates.csv.gz` means this step was
     skipped.
2. **Submit real training jobs** on the cluster: `jobs/dnn_noenv/run_dnn_noenv.sh` and
   `jobs/pinn_noenv/run_pinn_noenv.sh` (these call `run_dnn_noenv.py`/`run_pinn_noenv.py` with the
   real `--max-epochs 500`, not a quick sanity check). Both take a `split_type` argument
   (`temporal` or `spatial_block`, e.g. `sbatch jobs/dnn_noenv/run_dnn_noenv.sh 4survey 500 20 spatial_block`)
   — defaults to `temporal` if omitted. Cluster logs land in the matching `logs/dnn_noenv/` or
   `logs/pinn_noenv/` folder.
3. **Pull results back**: `rsync` `outputs/temporal/`, `outputs/spatial_block/`, and
   `outputs/run_logs/` down from the cluster to the laptop.
4. **Evaluate** (cheap, CPU): either locally with
   `python -m models.dnn_noenv.evaluate_dnn_noenv --split-type <temporal|spatial_block>`
   / `python -m models.pinn_noenv.evaluate_pinn_noenv --split-type <temporal|spatial_block>`, or on
   the cluster with `jobs/dnn_noenv/evaluate_dnn_noenv.sh` / `jobs/pinn_noenv/evaluate_pinn_noenv.sh`
   (same `split_type` argument). Omit `--cohort` (or the SLURM script's cohort argument) to run both
   4survey and 6survey. This writes the real `metrics.json`/`predictions.csv` and its own run-log
   entry, at `outputs/<split_type>/<model>/<cohort>/` — matching whichever `split_type` you fit
   under; using the wrong one here loads no checkpoint and fails loudly rather than silently
   comparing mismatched results.

## Checking a run is good, not just present

- Look in `outputs/run_logs/` for any `STARTED` filename tag with no matching `success`/`failed`
  entry sharing the same `attempt_id` → the job died silently, treat as failed.
- Check for `FAILED` filename tags → read the `error.traceback` field in that JSON directly.
- Check `is_test_run` on the **fit** entry, not just the evaluate entry (see caveat above) —
  a non-test evaluate result can still be built on a toy checkpoint if the fit was a sanity check.
- Open `training_history.csv` — did loss actually decrease and roughly plateau, or does it look
  like it stopped after 1-2 epochs (config bug) or never improved (learning rate / data bug)?
- Compare the evaluate-phase metrics against the sklearn/CR baselines in `baseline_results.ipynb`
  section 10 (temporal split numbers specifically) — a DNN/PINN doing much worse than CR or RF
  under the same split is a signal to check the run, not necessarily a real result.
- For PINN specifically: confirm `frozen_cr_params` in `run_metadata.json` matches
  `outputs/chapman_richards/<cohort>/params.json` — it should, since the PINN reads that file
  directly, but worth a sanity check if params.json was ever regenerated after the PINN ran.

## Reading order through the code (rough guide)

1. `models/common/data.py`, `models/common/splits.py` — how the master table becomes train/val/test.
2. `models/common/torch_data.py` — scalers, tensor building, `load_trajectory_pairs()`.
3. `models/common/torch_model.py` — `NoEnvNetwork`, `chapman_richards_derivative()`, `select_device()`.
4. `models/dnn_noenv/dnn_noenv.py` then `run_dnn_noenv.py` then `evaluate_dnn_noenv.py` — simplest
   full fit→evaluate path before adding the physics terms.
5. `models/pinn_noenv/pinn_noenv.py` — same shape as the DNN, plus `compute_physics_loss()` and
   `compute_trajectory_loss()`.
6. `models/common/run_logging.py` — the started/success/failed logging pattern used by every script.
7. `notebooks/model_results/baseline_results.ipynb` — how metrics get read back and compared across splits.

***

## Environmental data sources (Tier 1, started 21 July 2026)

`notebooks/environmental_data/environmental_data_sources_survey.ipynb` — the dissertation
plan's Tier 1, step 2 ("research and get access to terrain/wind data sources"). Live
access/resolution tests, not per-plot extraction yet (that's step 3, next).

- **OS Terrain 50 DTM — confirmed working, no API key needed.** Public OS Downloads API;
  downloaded the whole-GB ASCII Grid bundle (~160MB, one-time, cached in
  `data/raw/environmental/`, gitignored) and read a real tile (`NN40`) covering Aberfoyle's grid
  centre. Confirmed 50m resolution, plausible elevation range (21.6-726.1m on that tile).
  Verdict: include — source for elevation, slope, northness, eastness, TWI; TOPEX derives from
  it too (no separate access needed for TOPEX).
- **Global Wind Atlas — confirmed working, no API key needed**, public REST API, whole-GB
  GeoTIFF per height (10/50/100/150/200m; used 10m per the plan's "lowest suitable height"
  guidance). Real finding worth keeping: the advertised "250m" resolution is only true
  north-south — at Aberfoyle's latitude (~56N) the actual ground resolution came out ~155m
  east-west / ~278m north-south (a longitude-degree isn't a fixed distance), confirmed by
  reading real wind-speed values over the study bbox (0.07-18.77 m/s, mean 3.60 m/s at 10m).
  Verdict: include now, as the plan's own designated WASP fallback.
- **WASP wind atlas — not testable, access not yet requested.** No public API; access is via Dr
  Suárez-Minguez per the plan. **Action item, not yet done: email to request it** (plan's Day
  1-2 timeline; this is currently the blocking step for the "wind" side of Env-PINN v3).
- **HadUK-Grid, ERA5-Land, CEH soil, James Hutton soil map** — **correction, superseded below.**
  This line originally said these were ruled out on resolution grounds without live-testing them.
  That was wrong — see `aux_data_resolution_check.ipynb` below, which actually extracted and
  statistically screened all four (plus more) and found real, significant plot-level structure in
  most of them. Left here so the correction is visible, not silently dropped.

**Next (not started yet):** extract OS Terrain 50 + Global Wind Atlas features at every plot
centroid (needs every `NN`/`NS` DTM tile intersecting the study bbox, not just the one spot-check
tile above) — this is Tier 1 step 3. `models/dnn_env_terrain/` and `models/pinn_env_terrain/` are
scaffolded (empty `__init__.py`, same convention `linear_baseline`/`rf_baseline` used before they
were implemented) but have no real code yet — blocked on this extraction step.

***

## Environmental data sources, re-checked empirically (21 July 2026, `aux_data_resolution_check.ipynb`)

Settles the "resolution label alone" problem above with real extraction + a statistical screen
(CV, variogram range, Moran's I, within/between-compartment ICC, Spearman vs. the CR residual) on
the full 4survey plot set (71,766 plots), not a spot-check tile. A `trust_score` (0-1) combines
these plus provenance and temporal match into one comparable number per source; full table in
`notebooks/environmental_data/figures/aux_data_resolution_check_results.csv`.

**Climate and soil are NOT ruled out — the opposite of what this doc said above.** HadUK-Grid
(1km temperature) has the *strongest* Spearman correlation with the CR residual of anything
tested (0.291, p<0.001), Moran's I=0.94. CEH 50m soil (pedotope class) is also real and
significant (Spearman -0.145). James Hutton's 1:250,000 soil map showed 6 distinct WRB soil
groups in a 150-plot sample, more than the plan's assumed "~3-5 polygons total."

**GEE is fixed** (a real Google Cloud project ID was supplied mid-session) — this unblocked both
ERA5-Land and AlphaEarth. Neither is excluded for access reasons anymore:
- **HadUK-Grid**: real and strongest signal found, but only one year (2021) downloaded so far —
  needs multi-year averaging before it's trustworthy, not an access issue.
- **ERA5-Land**: real (~11.1km resolution via GEE, `ECMWF/ERA5_LAND/MONTHLY_AGGR`), but weak
  (Spearman 0.106) and coarse (8 distinct values over the whole forest) — not worth it next to
  CHELSA/HadUK-Grid on information-content grounds.
- **AlphaEarth**: access fixed, but 2017-onward coverage only — can't support the 2002-2023 study
  span regardless of access.
- **WASP** remains the only genuinely blocked source (people-dependency, email to Dr
  Suárez-Minguez still not sent).

**Ready to use now** for `dnn_env_terrain`/`pinn_env_terrain`: OS Terrain 50 terrain group
(elevation, slope, northness, eastness, TWI, TOPEX), Global Wind Atlas, SoilGrids pH, CHELSA
bio1, CEH pedotope class, distance-to-compartment-boundary, distance-to-forest-perimeter,
elevation roughness.

**Held back, not a blocked source but a real modelling risk**: neighbour mean height / height
differential (from nearby plots) have the *single strongest* raw correlation with the CR residual
found (Spearman 0.653) — but this is a spatial-lag feature (derived from the target itself via
neighbouring plots), likely to dominate a naive SHAP run without being a genuine environmental
driver. Needs a with/without SHAP check before use, not optional.

**TOPEX finding worth flagging directly**: TOPEX's raw correlation with the CR residual
(Spearman +0.098) is **entirely explained by elevation** — controlling for elevation drops it to
0.001 (p=0.836, not significant). Global Wind Atlas wind speed keeps about half its signal after
the same control (-0.213 raw -> -0.095 elevation-controlled, still p<0.001). A windward-only
(south-west-facing, Scotland's prevailing wind) variant of TOPEX was also built and tested — no
better than the omnidirectional version (Spearman 0.077 vs. GWA, 0.077 vs. residual, both
slightly *weaker* than omnidirectional) — a real null result, not just unexplored. Sign
convention, checked directly in the code rather than assumed: **positive TOPEX = sheltered,
negative = exposed** (validated on a synthetic conical hill, summit scored -133).

**New candidates identified, not yet built**: Global Wind Atlas also serves Ruggedness Index
(RIX), power-density, air-density, and capacity-factor layers via the same already-working API —
RIX is the most promising (flags "complex terrain" where standard wind models break down,
conceptually different from TOPEX/wind speed). DAMS (the UK forestry-standard windiness score,
validated on Sitka spruce anchorage specifically) was checked and explicitly set aside — its
underlying Wind Zone base map has no bulk GIS download, would need a formal Forest Research data
request. WASP's separate "Extreme Wind Atlas" product is self-service downloadable (no formal
request), but its own documentation explicitly warns it's unsuitable for mountainous terrain
(Aberfoyle's actual terrain), needs further WAsP Engineering software to become site-specific
rather than a simple raster, and measures rare extreme-gust events, not the chronic wind exposure
this study is centred on — not pursued for the general covariate set on these grounds.

`whcl` (raw GPKG windthrow hazard class) has only been visually checked
(`lidar_years_all_data_understanding.ipynb`), not run through the same real-extraction +
statistical screen as everything else here — still an open item, not resolved either way. Worth
checking its correlation against TOPEX/elevation specifically before treating it as independent,
since it may be a DAMS-like composite of the same underlying terrain inputs.

***

## Storm/windthrow diagnostic idea (raised 21 July 2026, not yet built)

A way to distinguish *chronic* wind exposure (gradually suppresses growth rate, what
TOPEX/Global Wind Atlas above are testing) from a *discrete* storm/windthrow event (sudden
damage, different signature) using data already on hand: chronic exposure should show as a
persistent residual offset across every survey year for a plot; a storm event should show as a
sudden drop at one specific survey-year transition, not before. No new external data needed —
this is a shape-of-trajectory check on the existing per-plot CR residuals across survey years.

Known storms mapped against the actual survey-year transitions:

| Transition | Cohort | Known storm(s) in the gap |
|---|---|---|
| 2002 -> 2006 | 6survey only | Storm Erwin/Gudrun, 7-9 Jan 2005 -- clean, one candidate |
| 2006 -> 2008 | 6survey only | None found -- useful as a "quiet" control transition |
| 2008 -> 2012 | both | Hurricane Bawbag/Cyclone Friedhelm, 7-8 Dec 2011 -- plausibly clean |
| 2012 -> 2021 | both | **Confounded** -- Dec 2013 (possibly the stormiest December on record), the 2014-15 season also flagged as stormy, and Storm Arwen (26 Nov 2021) all fall inside this one 9-year gap. Too many candidates to isolate one -- weakest transition for this diagnostic despite Arwen being the most famous. |
| 2021 -> 2023 | both | Storm Eunice/Storm Franklin, Feb 2022 -- clean, 2-year gap |

**Open item**: the exact month the 2021 LiDAR survey was flown isn't documented anywhere in this
repo, only the year — matters a lot, since Storm Arwen hit 26 November 2021 and whether the
survey flew before or after that date changes which transition (2012->2021 vs. 2021->2023) its
damage would actually show up in. Check Forest Research's flight records before trusting any
Arwen-specific before/after comparison.

Corroborating checks, not yet run: spatial clustering of "sudden-drop" plots (ties into the LISA
work below — a real storm should hit specific compartments, not scatter randomly), and
cross-referencing against `whcl` (a plot with high documented windthrow hazard class *and* a
sudden decline at the right transition is converging evidence, not proof either way alone).

***

## Spatial-question analysis plan (21 July 2026)

Full plan built via Ultraplan, saved at `.claude/plans/optimized-wiggling-catmull.md` (outside
this repo) — summarised here so the decision isn't only recorded somewhere ephemeral.

**Scope decision**: focusing on the spatial question only for now (attribution: why do plots
deviate from the CR/average growth curve, and prediction: does a model generalise to unseen
locations — the latter already done, see spatial_block_split results elsewhere in this doc). The
temporal-attribution question (why did a plot's deviation change over time) is data-starved, not
just unstarted — it needs time-varying environmental covariates, and the available terrain/wind
sources are all static present-day snapshots; HadUK-Grid/ERA5-Land are the only time-varying
options and were already deprioritised/found weak above.

**Cohort decision**: sticking with the existing balanced 4survey/6survey cohorts for the spatial
work too, not building a bigger "unbalanced" dataset. Reasoning: a plot with only 1-2 survey years
makes curve fitting harder regardless of whether the panel is balanced, so relaxing the
"present-in-every-year" requirement doesn't net-remove noise, just trades one kind for another —
not worth the pipeline churn for that trade.

**Planned sequence**: (1) Moran's I on CR residuals — the gate, confirms real spatial structure
exists before attributing it to anything; (2) LISA cluster maps — the rigorous version of
eyeballing a colour map, reuses the plotly machinery from
`spatial_viz_comparison_scratch.ipynb`; (3) XGBoost + SHAP on the "ready now" covariate list
above; (4) **NLME, sequenced right after SHAP, not after Env-PINN** — SHAP's confirmed continuous
covariates (TOPEX, wind speed, elevation, etc.) become NLME's fixed effects directly on `y_max`,
plot/compartment as the random effect (the Slovakia beech paper's structure, not the Larch
paper's categorical-site-type structure — worth stating that distinction plainly when this gets
written up, since it's adapting the two-stage template, not copying it exactly). NLME needs to
come before/alongside Env-PINN architecture decisions, not after — it gives an interpretable
"how much does terrain/wind explain" number that should inform how much complexity the PINN's
sub-network actually needs to earn its place, and building it after Env-PINN risks retrofitting it
to match whatever the PINN happened to use rather than being an independent check; (5) GAMLSS and
GWR — worth a paragraph each in the methods/limitations discussion regardless of whether they get
built; GWR is moderate effort and worth doing, GAMLSS's full implementation is a bigger lift than
the rest of this plan combined (thin Python tooling) and shouldn't be prioritised unless there's
clearly spare time.

**Charting decision**: built `notebooks/scratch/spatial_viz_comparison_scratch.ipynb`
comparing 5 spatial-plotting approaches (matplotlib hexbin, ipympl, HoloViews+Datashader,
Lonboard, Plotly) on the same RF/spatial_block/4survey residual data. **Plotly is the standard
going forward** for every future spatial figure (residual maps, learned `y_max` maps, attribution
maps) — Lonboard was dropped after real rendering failures in the user's Jupyter environment, on
top of already being the less mature/supported option of the two on paper (19x fewer GitHub stars
than plotly, no built-in static export). Follow-up task, not done yet: turn the winning approach
into a shared `spatial_plot.py` utility so this decision only gets made once.

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

### Age filter — resolved 15 July 2026, forester-consulted decision

The cleaning funnel's "plausible age" stage exists but is currently a no-op
(`AGE_MIN, AGE_MAX = 0, 200`), and the height check only validates `Top_Height95`, not
`Top_Height99` (empirically fine right now — max `Top_Height99` is 53.5m/46.5m, both under 60 —
but not formally enforced for the actual target column).

**What the forester said** (asked directly about the young-age vs old-age tails):
- **Young trees are unreliable**, not just noisy: top height is "unrelated to age and competition"
  at young ages, which is exactly why surveyors are told to estimate Yield Class instead of
  measuring top height around that age. This is a *measurement*-reliability problem, not just a
  modelling-difficulty one.
- **Old trees (70-80+) are fine to keep.** The increasing spread at the old end is expected
  allometric heteroskedasticity (real biological variability as trees approach senescence/
  self-thinning), not a data-quality artefact. Consistent with what's already decided below for CR's
  old-age bias — flag via age-banded metrics, don't truncate.
- The forester's number for the young-end cutoff is **30**, tied specifically to the age at which
  UK forestry practice switches from measuring top height to estimating Yield Class — a
  mechanistic reason, not a round-number guess.

**Decision: no upper age cap; lower bound implemented as a *plot-level* rule — every plot's age
must be ≥30 by 2023** (`age_2023 = 2023 - plyr >= 30`, i.e. `plyr <= 1993`), which is equivalent to
a sliding per-survey-year minimum age: ≥15 in 2008, ≥9 in 2002, ≥22 in 2021, etc. Real impact on
the data:

| | 4survey | 6survey |
|---|---|---|
| Plots removed | 13,654 / 71,766 (19.0%) | 128 / 13,897 (0.9%) |
| Rows removed | 54,616 / 287,064 (19.0%) | 768 / 83,382 (0.9%) |
| Rows with Age ≥ 80 kept (no upper cap) | 5,170 (1.8% of rows) | 520 (0.6% of rows) |

**Pros of the plot-level (age-in-2023 ≥ 30) rule, vs. filtering every row's own Age ≥ 30:**
- Removes far less data. A strict row-level `Age >= 30` filter (drop any row below age 30
  regardless of year) would remove 36.7% of 4survey rows and 44.8% of 6survey rows — the
  plot-level version only removes 19.0% / 0.9%, because 6survey's plots are mostly long-tracked,
  already-mature stands.
- Preserves complete plot trajectories, which matters specifically for the PINN's
  trajectory-consistency loss — it needs consecutive-survey pairs from the *same* plot; discarding
  a plot's early-age rows would shrink the usable pair pool more than a row-level filter would.
- One simple, reproducible, deterministic condition (`plyr <= 1993`), easy to apply identically
  across baselines/DNN/PINN.
- By 2023 — the year used as the held-out test set for `temporal_split()` — every test-time
  prediction is guaranteed to be on a tree ≥30, which is exactly where the forester's concern was
  weakest (their comment was about *young* trees being unreliable; 2023 is never young under this
  rule).

**Cons / open risk, worth stating explicitly in the write-up:**
- **Does not implement the forester's stated mechanism precisely.** The forester's concern is about
  the reliability of an individual LiDAR survey at the age it was taken, not about whether the same
  plot eventually matures. This rule keeps young-age rows (e.g. age 15-29) from plots that mature
  by 2023, which are exactly the kind of low-reliability measurements the forester flagged —
  they're retained here only because their *parent plot* happens to be resurveyed for long enough.
- **Possible selection bias.** The surviving young-age (<30) observations are conditioned on the
  plot being resurveyed again and eventually reaching 30+ by 2023 — not a random sample of young
  stands. Plots that were felled, not resurveyed, or belong to a different management trajectory
  are disproportionately excluded, and it's untested whether that correlates with anything
  height-relevant (e.g. yield class, thinning regime).
- **Very asymmetric bite across cohorts** (19.0% of 4survey rows removed vs 0.9% of 6survey) means
  "the same rule" has very different practical effect depending on cohort survey-year structure —
  worth flagging when comparing 4survey vs 6survey results, since some of the difference could
  reflect this rather than a genuine cohort difference.
- **No upper cap** means old-age heteroskedasticity is fully present in training data — correct
  per the forester, but means age-banded reporting (already used for CR) stays necessary rather
  than optional for interpreting old-age results.

If the mismatch above turns out to matter (e.g. training curves look meaningfully different once
young-but-eventually-mature rows are excluded), the row-level `Age >= 25`/`Age >= 30` variant
computed earlier is the fallback robustness check — see `documentation/experiment_log.md`.

Not yet implemented in `data_processing/` — still to decide: master cleaning funnel (permanent,
affects every model) vs. a per-model filter like the existing `Age >= 20`.

### RF model artifacts and Git LFS

Decided **not** to use Git LFS for `rf_baseline`'s `model.joblib` (can be 300MB–1.3GB+ with
sklearn's default unlimited tree depth). It's fast and deterministic to regenerate
(`models/baselines/run_baselines.py`, fixed seeds throughout), nothing downstream depends on the
exact binary (only its `metrics.json`/`predictions.csv`, which are tiny and tracked), and GitHub's
free LFS tier (1GB storage/bandwidth) would already be exceeded by these two files alone. `outputs/`
is gitignored entirely for this reason — regenerate on demand, don't try to version-control fitted
model binaries.

Update 2026-07-21: future RF saves now use `joblib.dump(..., compress=3)` and record
`joblib_compress` in `model_metadata.json`. This does not change predictions or notebook outputs,
but it only affects models saved after this change. Existing `model.joblib` files stay large unless
the RF baselines are rerun.

If storage pressure later justifies a smaller RF itself, use an explicit new rerun/config rather
than silently replacing the current baseline. Candidate modest-RF controls: keep `n_estimators=100`,
set a finite `max_depth` (for example 20-24), and use `min_samples_leaf=2` or higher. That would
change the fitted trees and may change RMSE/MAE, so update `model_metadata.json`, `metrics.json`,
`predictions.csv`, and the results notebooks together, and describe it as a new RF baseline
configuration rather than a pure storage-only change. Making model saving optional is a separate
pipeline change because `evaluate_baselines.py` currently expects `model.joblib` to exist.

***

## What's not started yet

**Status update, 18 July 2026** (supersedes the paragraph below, kept for the historical record):
DNN/PINN are now built, tuned, and evaluated under both `spatial_block_split` (primary) and
`temporal_split` (secondary) — genuine convergence confirmed for both (a hyperparameter-tuning pass
barely moved final RMSE, see `experiment_log.md`'s 2026-07-17 Findings entry), except DNN's
`temporal`/4survey run, which still shows an overfitting-adjacent pattern tuning only softened, not
fixed (a data-design limit: only 2 distinct training years, not a tuning problem). The PINN's
`physics_weight`/`trajectory_weight` had been left at an untested default of `1.0` since it was
first built — a weight sweep (7 values, 0.0-5.0) run under both splits found this was actively
hurting accuracy; **`pw=tw=0.05` is now the shared default for both `spatial_block` and
`temporal`** (see `experiment_log.md`'s Decisions log for the full reasoning, including the honest
nuance that the per-cohort optimum isn't identical to `0.05` under every split). Both results now
live in `notebooks/model_results/baseline_results.ipynb` (organized by split type, sections 5 and 6).
Not yet started: terrain/wind feature extraction, XGBoost + SHAP, and the PINN's `cr_matched`
anchor variant (see `experiment_log.md`'s naming glossary for what these are).

**`temporal_narrow_gap`: code added 20 July 2026, no jobs run yet.** `--split-type
temporal_narrow_gap` now works end-to-end (smoke-tested locally for baselines, DNN, and PINN,
all cohorts) on `run_baselines.py`/`run_dnn_noenv.py`/`run_pinn_noenv.py` and their matching
`evaluate_*` scripts — see `TEMPORAL_YEARS_NARROW_GAP` in `models/common/splits.py`. Worth knowing
if extending this split further: the year assignment isn't simply "move `temporal`'s val year
(2021) into train" — the PINN's trajectory loss needs a pair of chronologically ADJACENT real
surveys both labelled train, and holding out a YEAR IN THE MIDDLE of the sequence (rather than the
earliest) silently starves it to 0% usable pairs. A first attempt at this dict did exactly that and
PINN failed outright, while DNN and the baselines ran "successfully" on the same years since
neither uses trajectory pairs — easy to miss without a real smoke test of all three model types.

**Status update, 20 July 2026**: every result above (both weight sweeps, DNN-vs-PINN on every
split/cohort) is a SINGLE random seed per configuration — never tested for run-to-run variance. An
audit of every close-margin claim found DNN-vs-PINN on 6survey (0.6-1.0% RMSE apart) is tighter than
any weight-sweep gap, and it's the basis for the "PINN's edge is stability, not accuracy" framing.
**A reseed batch (40 cluster jobs: PINN's low-weight band + plain DNN, both cohorts, both splits,
seeds 43/44) has been submitted but not yet returned** — this is a submitted-job status, not a
result, so no claim above should be revised on the strength of it yet. Once it lands: evaluate
locally, fill in `baseline_models_parameter_tuning.ipynb` section 10 (already wired to read
whichever of the 40 results exist and report on them), and update this file / `experiment_log.md`
with whatever the actual variance turns out to be — do not treat this status update itself as
confirming or refuting anything, it only records that the check is running.

**Original entry (15-16 July 2026, now historical):** DNN/PINN under `spatial_block_split` now has
a first real result (16 July 2026) — both beat every
baseline on both cohorts, and beat their own `temporal_split` numbers too — but it's not yet
confirmed whether that reflects genuine convergence or an early-stopping artefact (see
`experiment_log.md`'s hyperparameter-tuning plan; each candidate fix is cheap to test at only
21-30 epochs per run). All 8 real runs so far stop well short of the 500-epoch budget, and DNN
specifically shows unstable behaviour across settings (healthy on one cohort/split, a textbook
overfitting collapse on another) — tuning effort is prioritized on PINN, the more consistent model
and the actual research focus, going forward (see `experiment_log.md`). Also not started:
terrain/wind feature extraction, XGBoost + SHAP, `temporal_narrow_gap`, and the PINN's `cr_matched`
anchor variant (see `experiment_log.md`'s naming glossary for what these are). A DNN/PINN results
notebook now exists
(`notebooks/model_results/baseline_models_parameter_tuning.ipynb`) covering loss curves and metrics for
both split types, updated automatically as real runs land.

**Deferred, 2026-07-21 (user's call — revisit later or fold into Env-PINN, not before)**:
- No formal statistical test on the reseed results yet — current analysis is informal seed-counting
  and eyeballing spread vs. mean gap. A bootstrap CI or Cohen's d on the DNN-vs-PINN gap would be
  more defensible (n=3 seeds limits precision either way).
- Not checked whether no-env DNN/PINN errors come from underfitting (not learning real patterns) or
  over-reliance on one feature (e.g. mostly `Age`, ignoring the rest) — a permutation-importance or
  partial-dependence check would answer this, and the same check applies again once terrain/wind
  features exist.

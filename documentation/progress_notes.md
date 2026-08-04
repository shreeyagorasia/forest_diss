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
│   ├── environmental_data_sources_survey_SUPERSEDED_2026-07-30.ipynb   # Tier 1 data-source survey, see section below
│   ├── av1_aux_data_resolution_check.ipynb           # empirical re-check of every source (real
│   │                                              # extraction + statistical screen), see below
│   └── figures/   # aux_data_resolution_check_results.csv lives here now
├── spatial_analysis/
│   ├── spatial_temporal_split_visualisation.ipynb # maps/visualises the three split types
│   └── av1_spatial_autocorrelation_terrain.ipynb      # built 22 July 2026 -- Moran's I + semivariogram
│                                                   # done (section 1); SHAP/NLME/GWR terrain
│                                                   # attribution (section 2) blocked on the
│                                                   # per-plot feature extraction step, not built yet
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

`notebooks/environmental_data/environmental_data_sources_survey_SUPERSEDED_2026-07-30.ipynb` — the dissertation
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
  That was wrong — see `av1_aux_data_resolution_check.ipynb` below, which actually extracted and
  statistically screened all four (plus more) and found real, significant plot-level structure in
  most of them. Left here so the correction is visible, not silently dropped.

**Next (not started yet):** extract OS Terrain 50 + Global Wind Atlas features at every plot
centroid (needs every `NN`/`NS` DTM tile intersecting the study bbox, not just the one spot-check
tile above) — this is Tier 1 step 3. `models/dnn_env_terrain/` and `models/pinn_env_terrain/` are
scaffolded (empty `__init__.py`, same convention `linear_baseline`/`rf_baseline` used before they
were implemented) but have no real code yet — blocked on this extraction step.

***

## Environmental data sources, re-checked empirically (21 July 2026, `av1_aux_data_resolution_check.ipynb`)

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
  needs multi-year averaging before it's trustworthy, not an access issue. **Correction,
  2026-07-30: this caveat is fixed, not still open** — `tas`/`groundfrost` are now genuinely
  averaged across all 6 survey years, cohort-aware (`models/common/download_haduk_multi_year.py`),
  and `tas`'s correlation held up under the real multi-year check (Spearman 0.273, 4survey).
  HadUK-Grid has moved from "held back" to "in the main feature set" — see
  `av1_aux_data_resolution_check.ipynb`'s HadUK-Grid section and `handover_2026-07-18.md` for the
  full numbers. Left the original line above unedited so the correction is visible, not silently
  dropped, per this doc's own convention just above.
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
(Spearman +0.093) is **entirely explained by elevation** — controlling for elevation drops it to
-0.001 (p=0.804, not significant). (2026-07-30: corrected to match the notebook's actual printed
output -- this entry previously said 0.098/0.001/0.836, copied from a markdown cell in
`av1_aux_data_resolution_check.ipynb` that itself didn't match its own cell's real output, now also
fixed.) Global Wind Atlas wind speed keeps about half its signal after
the same control (-0.206 raw -> -0.091 elevation-controlled, still p<0.001). A windward-only
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

## Spatial attribution: `models/spatial_attribution/` built, steps 1-2 done (22-23 July 2026)

New branch: `spatial_attribution`. New folder `models/spatial_attribution/` holds the real,
reusable code — the notebook (`notebooks/spatial_analysis/av1_spatial_autocorrelation_terrain.ipynb`)
only calls it and adds plots, per the user's explicit "keep plotting logic in the notebook, not
the module" correction this session.

**`data.py`** — generalised beyond Chapman-Richards on purpose, so any model/dataset can be
plugged in:
- `load_residuals(model_name, cohort, split_type)` / `load_residuals_by_year(...)` — `model_name`
  is a real parameter now (any model's `predictions.csv` folder name), not hardcoded. Both
  filter to `split == "test"` only — see the "why test-only" reasoning below, it's load-bearing,
  not just convention.
- `load_master(cohort)` — the full `data/processed/master/clean_master_{cohort}.parquet`.
- `join_by_plot(base_df, other_df, on="identification")` — a merge wrapper that reports how many
  rows from each side failed to match, so a bad join fails loudly instead of silently dropping
  rows.
- `LEAKAGE_RISK_COLUMNS` — the Dissertation Plan's "Height-derived predictors" list
  (`Vol95`/`99`/`RM95`, `GYCspec95`/`99`, raw height percentiles, `Top_Height95`) as one named,
  shared constant, so every future join excludes them the same way instead of each analysis
  re-deciding from scratch.

**`spatial_autocorrelation.py`** — `global_morans_i(x, y, values, distance, ...)` and
`semivariogram_range(x, y, values, ...)`. Both fully generic (just take arrays), not tied to any
one model's residual — confirmed useful immediately, since re-running them on DNN/PINN needed
zero module changes, only different data passed in.

**`lisa.py`** — `local_morans_i(x, y, values, k, ...)`. Uses k-nearest-neighbours (not a fixed
distance) plus Benjamini-Hochberg correction — both real fixes, not defaults kept out of
laziness: a fixed distance gave wildly uneven neighbour counts (681-7,141 depending on local
density, checked directly), and ~11,000 simultaneous per-plot significance tests need FDR
correction or ~5% would show "significant" by chance alone.

### Why test-split only, and why this isn't just "the convention"

Confirmed directly (not assumed): `predictions.csv` only ever contains `'test'` rows, for every
model — train/val predictions are never saved anywhere. Worked through why this matters
specifically for the spatial analysis, not just generally:
1. Training residuals are artificially small (the model was directly optimised to minimise them).
2. **Mixing splits would manufacture a fake result, not just a less-clean one** —
   `spatial_block_split` assigns whole contiguous compartments to train/val/test, so the split
   itself has real spatial structure. If train residuals are systematically smaller and
   train/test assignment is spatially clustered by construction, Moran's I on a mixed sample
   would detect that split-assignment artefact as "spatial autocorrelation" — indistinguishable
   from a genuine finding without knowing this.
3. **A separate, real gap this surfaced**: validation predictions are computed every epoch during
   DNN/PINN training (`evaluate_on_validation_set()` genuinely runs the model forward on val
   rows) but only the aggregate `val_loss` is kept (`training_history.csv`) — individual per-plot
   val predictions are never saved, for any model, because nothing needed them before now.
   Building that (straightforward for CR — just its 3 params; needs the saved checkpoint+scalers
   for DNN/PINN) is required before Step 3 can run properly: **SHAP directly selects Env-PINN's
   covariates, so running it on test residuals would leak test information into a design
   decision, compromising Env-PINN's eventual test evaluation.** Steps 1-2 (Moran's I/LISA, "is
   there a phenomenon at all") don't have this problem — they don't select anything about
   Env-PINN's architecture, so test is fine there. **Not built yet** — next real task before
   Step 3.

### Results, Step 1 (Chapman-Richards) — real numbers, verified in the notebook

Semivariogram range = 3,956m (status: resolved, genuinely flattened within the 5,000m window).
Global Moran's I = 0.130 at that distance, 999 permutations: p_sim=0.001 (floor value — beat all
999 shuffles), p_norm≈0.000000 (not floor-limited, confirms the same conclusion independently),
z=270-285 depending on the random sample draw (huge z is a large-n artefact, not a huge effect —
I itself, 0.13, is the real effect size). **Formal result: reject H0 of spatial randomness at
α=0.05.** LISA (k=20 vs k=50 sensitivity check, not just one k): 81.2% exact label agreement,
56.6% significant at both k, 100% cluster-type agreement among those — core clusters are robust
to the choice of k, the boundary is what's k-sensitive.

### Results, Step 2 (CR vs. DNN vs. PINN comparison) — real numbers, genuinely informative

| Model | Semivariogram range | Moran's I |
|---|---|---|
| Chapman-Richards | 3,956m | 0.130 |
| DNN (no-env) | 2,585m | 0.072 |
| PINN (no-env, tuned) | 2,557m | 0.080 |

Same 11,743 plots for all three (confirmed identical row/plot counts before assuming — DNN/PINN's
extra predictor columns could plausibly have caused missing-value dropout CR wouldn't hit; they
didn't). **Both DNN and PINN show a shorter range and weaker Moran's I than CR — roughly half —
even with zero terrain/wind features given to either.** DNN and PINN land close to each other,
not dramatically different despite PINN's extra physics/trajectory-loss terms. Spatial structure
doesn't disappear in the more flexible models, it shrinks (still significant, p_norm≈0 for both).
**Reading**: some of what looks like "environmental" structure in CR's residuals is apparently
already capturable by model flexibility alone (`CanopyCover`/thinning/`yldc` as inputs) — the
smaller, remaining structure in DNN/PINN is the more honest target for terrain/wind attribution
to explain, not the larger CR-only signal. Worth measuring the eventual SHAP/NLME "variance
explained" claim against this smaller baseline, not CR's.

### Plan for next session

1. Build val-residual generation (CR first — trivial; DNN/PINN need checkpoint+scaler reload).
2. Extract terrain/wind covariates into a clean, reusable per-plot table (Tier 1 step 3 — still
   the biggest remaining blocker, not done yet despite the exploratory work in
   `av1_aux_data_resolution_check.ipynb`).
3. Run XGBoost + SHAP for real, on validation residuals + real covariates.
4. Build `pinn_env_terrain` using whatever SHAP confirms is worth conditioning on.

**Open question, not yet answered**: have we done *enough* testing to know which variables to
include? No — not yet. Steps 1-2 above only establish THAT real spatial structure exists (the
gate); no actual covariate has been tested against the residual with a proper multivariate method
yet. `av1_aux_data_resolution_check.ipynb`'s Spearman correlations are useful exploratory signal
(HadUK-Grid strongest at 0.291, TOPEX's correlation with residual vanishing once elevation is
controlled for, GWA keeping about half its signal) but are univariate, not the real screening
step — that's what Step 3 (SHAP) is for, still to come.

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

**Consolidated numeric record (retired `Top_Height99`+`yldc` pipeline, 13-20 July 2026)** —
`experiment_log.md`'s detailed table/Findings entries for this period were trimmed 29 July 2026
(not comparable to the rebuilt pipeline's numbers; full detail still in git history). Key
numbers preserved here:
- `plot_level` (13 Jul): RF best both cohorts (RMSE 4.65/3.86).
- `spatial_block` (14 Jul): RF loses its `plot_level` edge (RMSE +28.7%/+19.4%).
- `temporal_wide_gap` baselines (15 Jul): much larger degradation than spatial (up to +141.7%
  RMSE); CR most temporally robust, not RF.
- DNN/PINN tuned, `temporal_wide_gap` (17 Jul): DNN 4survey RMSE=5.8600/R²=0.4533 (overfitting
  climb softened, not fixed), 6survey RMSE=4.9388/R²=0.2749 (healthy). PINN (`W=1.0`) 4survey
  RMSE=6.0870/R²=0.4101, 6survey RMSE=4.8857/R²=0.2904.
- DNN/PINN tuned, `spatial_block` (17 Jul): DNN 4survey RMSE=5.0185/R²=0.6091, 6survey
  RMSE=3.6498/R²=0.7434. PINN (`W=1.0`) 4survey RMSE=5.4642/R²=0.5366, 6survey
  RMSE=3.7039/R²=0.7358 — DNN beat PINN on both, traced to the untested `W=1.0` default.
- Weight sweep, `spatial_block` (17 Jul): best per-cohort 4survey `W=0.0` (RMSE=5.0822,
  R²=0.5991), 6survey `W=0.05` (RMSE=3.6265, R²=0.7467, beats DNN). Chosen shared default
  `W=0.05` (4survey RMSE=5.1209, R²=0.5930).
- Weight sweep, `temporal_wide_gap` (18 Jul): 4survey optimum again `W=0.0` (RMSE=5.9091,
  R²=0.4441); 6survey optimum `W=0.2` (RMSE=4.8294, R²=0.3067), `W=0.05` itself a wash here
  (RMSE=5.9348/R²=0.4393 and RMSE=4.8915/R²=0.2888 respectively at the chosen shared default).
- Reseed check, 3 seeds (20 Jul): CONFIRMED not noise — 4survey's `W=0.0` preference (6/6
  seed×split checks) and DNN's 4survey win over PINN. RETRACTED as noise — 6survey's "optimal
  weight" and the "PINN wins 6survey" claim (PINN beat DNN in only 2/3 seeds, <0.2% RMSE gap).
- `temporal_narrow_gap` (20 Jul): 7/8 baseline model/cohort combos degrade less than
  `temporal_wide_gap`; DNN/PINN both improved 10-12% RMSE moving from wide to narrow gap on both
  cohorts (DNN: 4survey 5.8600→5.2814, 6survey 4.9388→4.3445; PINN: 4survey 5.9348→5.2716,
  6survey 4.8915→4.3795) — confirms gap length, not "temporal prediction is inherently hard,"
  drives `temporal_wide_gap`'s degradation.

**Deferred, 2026-07-21 (user's call — revisit later or fold into Env-PINN, not before)**:
- No formal statistical test on the reseed results yet — current analysis is informal seed-counting
  and eyeballing spread vs. mean gap. A bootstrap CI or Cohen's d on the DNN-vs-PINN gap would be
  more defensible (n=3 seeds limits precision either way).
- Not checked whether no-env DNN/PINN errors come from underfitting (not learning real patterns) or
  over-reliance on one feature (e.g. mostly `Age`, ignoring the rest) — a permutation-importance or
  partial-dependence check would answer this, and the same check applies again once terrain/wind
  features exist.

## Environmental attribution, Tier 2: XGBoost/SHAP, Elastic Net, grouped category analysis (28 July 2026)

**Tier 1 additions to `av1_aux_data_resolution_check.ipynb`** since the 21 July entry above: added
CEH TWI/subsurface_drainage/textural_composition and CHELSA gdd5/bio12 (found while answering the
notebook's own embedded "QUESTIONS I HAVE" cells — these map onto data already downloaded but
unused). Corrected an earlier mistake: HadUK-Grid and ERA5-Land were nearly excluded TOGETHER as
"single-year climate snapshots" — wrong, by category not by evidence. Their actual numbers differ
(HadUK: 149 distinct values, rho=0.29 vs residual; ERA5: 8 distinct values, rho=0.11) — HadUK kept,
ERA5 excluded, each on its own numbers. Also added `whcl` (raw GPKG windthrow hazard class, 0-6,
confirmed constant per plot across survey years) and six stand-structure/silvicultural variables
(`CanopyCover`, `Thin`, `time_since_thinning(+missing)`, `recent_thinning_5yr`, `yldc` — already
used by `rf_baseline.py`, aggregated to one value per plot by MEAN across survey years, the same
aggregation `mean_cr_residual` itself uses). `Age` was deliberately left OUT — see the circularity
finding below.

**New package `models/xgb_environmental/`** (plain Python, mirrors `rf_baseline.py`'s style):
`data.py`, `xgb_environmental.py` (`FEATURE_PROVENANCE`, `FEATURE_SETS`, fit/predict/SHAP),
`run_xgb_environmental.py` (CLI orchestrator), `grouped_analysis.py` (domain-category grouping +
grouped permutation importance + Moran's I before/after, added this session, see below). One
unified `all_environmental` feature set (35 variables) — no separate environmental-only vs
environmental+silviculture split kept in parallel, per an explicit "no bloat" steer.

**Bug found and fixed: SHAP computed on test rows.** `compute_shap_values()` was being called on
the FULL plot set (train+val+test), including the held-out test split — this let the "final"
test set be inspected ahead of its one honest evaluation, and (worse) the Tier-2 notebook's
ablation-refit comparisons were repeatedly re-using TEST R² to decide which features to keep,
turning it into a second validation set. Fixed: `run_xgb_environmental.py` now uses the full
train/val/test split `spatial_block_split()` already returns (val was previously silently
dropped — only train/test were read), SHAP is restricted to train+val only, and val is the only
split used for any feature-selection decision — test is read once, at the end, as the final
number.

**Age is circular with `mean_cr_residual`'s own construction — real finding, not assumed.**
Including `Age` alongside the stand-structure variables initially pushed XGBoost test R² from
0.567 to 0.914, with `Age` as the 2nd-highest SHAP feature. Checked why: binning plots by age and
averaging the residual per bin shows a real, non-monotonic bias in the single global
Chapman-Richards curve (+0.99 at 25-32yrs, -0.62 at 40-48yrs, +0.79 at 56-64yrs, -4.5 at
79-87yrs) — Age's raw correlation with the residual is ~0 (rho=0.02) because the pattern zigzags
rather than trending, but XGBoost's splits can re-learn this exact zigzag. Since `Age` is the
Chapman-Richards curve's own only input, handing it back as a feature lets the model patch the
curve's own fit bias rather than learn anything new about environment/silviculture. **`Age`
excluded from the feature set for this reason** (see the comment in `xgb_environmental.py`) —
the other stand-structure variables (`CanopyCover`, `Thin`, etc.) are genuinely independent of
how the residual was built and stay in.

**Elastic Net (`models/elasticnet_environmental/`), the SHAP-alternative built this session** —
chosen over GAM/Double ML/BART (all discussed and deferred as future work) because it directly
answers "does this cause more or less growth, by how much" via a standardized linear
coefficient, and its regularization handles correlated features differently to SHAP (spreads
credit across a correlated group instead of SHAP's credit-splitting, which a fitted
`ElasticNetCV` picks via CV on train only). The three CEH categorical raster layers are one-hot
encoded (feeding a class ID into a linear model would assume a fake ordering); rows with any
missing feature are dropped (XGBoost handles NaN natively, Elastic Net cannot, <1% of rows
affected).

**Grouped category analysis, new notebook: `notebooks/environmental_data/av1_grouped_category_importance.ipynb`**
— supersedes `env_variable_importance_RETIRED_2026-07-28.ipynb` (retired, note added at its own top, left
unmaintained not deleted). Domain categories (not the same as the earlier correlation-based
clusters): terrain, wind, soil/site, climate, spatial position/edge effects
(`dist_to_cpmt_boundary`/`dist_to_forest_perimeter` — edge-effect mechanism, kept separate from
neighbour features which are a weaker, non-exogenous proxy), stand structure, neighbour/spatial-lag.
Pipeline, in order: correlations (within + cross-category, flagging e.g. elevation's real
rho=+0.31 with GWA wind speed but rho=-0.17 with topex — the same "wind" label covers two
measures that disagree here) → Elastic Net → XGBoost → **grouped permutation importance**
(shuffles a whole category's columns together through the already-fit model, safer than SHAP
under correlation) → SHAP+ALE → **spatial cross-validation demonstrated directly**: the same
XGBoost model/data scored 0.567 test R² under the spatial-block split vs 0.903 under a plain
random plot-level split — +0.335 R² of pure inflation from skipping a spatial-aware split → **Moran's
I before/after** (reusing `models/spatial_attribution/spatial_autocorrelation.py`'s
`global_morans_i()`/`semivariogram_range()`): full-model residual Moran's I = 0.243 (p=0.005,
real spatial structure remains even with everything in); removing `terrain` increases it the most
(+0.297) — terrain is the category most responsible for explaining spatial pattern; removing
`neighbour_spatial_lag` DECREASES it (-0.162), a genuinely counterintuitive result not forced
into a tidy narrative. Closing section cross-tabulates Elastic Net/SHAP/permutation-importance
ranks per category so agreement/disagreement is visible directly (the same cross-check logic that
caught the `Age`/`elevation` circularity above).

**Python files created this session**: `models/xgb_environmental/{__init__.py, data.py,
xgb_environmental.py, run_xgb_environmental.py, grouped_analysis.py}`,
`models/elasticnet_environmental/{__init__.py, elasticnet_environmental.py,
run_elasticnet_environmental.py}`.

**Correction, 2026-07-31: `neighbour_spatial_lag` (`neighbour_mean_height`/
`neighbour_height_differential`) is a confirmed leak, not just "a weaker, non-exogenous proxy"
as described above — removed from `ALL_FEATURE_COLUMNS`/`CATEGORY_GROUPS` entirely, not kept as
a flagged-but-included category.** Both columns are built from every OTHER plot's own real 2023
height within a 75m radius, computed once on the full plot set BEFORE any train/val/test split
exists. Because `spatial_block_split()` holds out whole compartments, 95.9% of a TEST-set
plot's within-75m neighbours are ALSO test-set plots (82.3% of test plots have zero train-set
neighbours in range) — so this "feature" was built almost entirely from other test-set plots'
real ground truth, not learned from training data. This is why it dominated every ranking above
(`mean R2 drop=1.177`, ~10x every other category) and why removing it collapsed the model's
apparent skill (test R² 0.598→0.321 4survey, 0.327→−0.337 6survey, per the pre-existing
`all_environmental_no_neighbour` ablation) — not because it was a genuinely powerful
environmental driver. Every number in this section computed before 2026-07-31 needs re-reading
against the fixed numbers (`experiment_log.md`'s 2026-07-31 entry, and the re-run
`av1_grouped_category_importance.ipynb`), not cited as-is. Checked every other feature in
`ALL_FEATURE_COLUMNS` for the same construction pattern (an aggregate of other plots' own
height/growth, computed pre-split) — none share it.

**Still deferred**: causal SHAP, GAM, Double/Debiased ML, BART (documented future work, see the
new notebook's closing section); multi-year HadUK-Grid (currently 2021 only — needs 5 more manual
CEDA logins/downloads, a manual dependency, not scripted); `models/spatial_attribution/` rename
(naming collision with "attribution" flagged, not yet actioned).

## Four new spatial-position/edge-effect variables (28 July 2026)

`spatial_position_edge_effects` only had two variables (`dist_to_cpmt_boundary`,
`dist_to_forest_perimeter`) — asked what else could realistically go there, checked the raw
GPKG and clean master rather than guessing, and found four real, previously-unused additions:

- **`dist_to_scpt_boundary`** — distance to nearest sub-compartment boundary. `scpt` was already
  a column in `clean_master_4survey.parquet`, just never used for a boundary-distance feature.
  Confirmed `scpt` labels ("A", "B", "C"...) repeat across different compartments (all 26 values
  appear in more than one), so boundaries are dissolved by `cpmt`+`scpt` together, not `scpt`
  alone — otherwise unrelated sub-compartments would get wrongly merged.
- **`dist_to_block_boundary`** — same idea, one level coarser, dissolved by `blk` (18 blocks in
  the full raw GPKG).
- **`cpmt_compactness_ratio`** — NOT a per-plot distance like the others: a per-compartment shape
  property (perimeter ÷ area). A small or elongated compartment has proportionally more edge
  than a large, compact one — answers "how edge-dominated is the whole stand", a different
  question to "how far is this plot from an edge".
- **`dist_to_road`** — distance to nearest road/track, via OS Open Roads (confirmed live, no
  auth, same free-access pattern as OS Open Rivers already used for `dist_to_watercourse`; ~1GB
  GB-wide file, cached at `data/raw/environmental/oproad_gb.gpkg`, gitignored). Includes
  unclassified forest tracks, not just A/B roads — even an unclassified track creates a real
  canopy opening.

`models/common/export_compartment_boundaries.py` generalized to dissolve at all three scales
(compartment/sub-compartment/block) in one script; `models/common/geo.py` gained
`load_subcompartment_boundaries()`/`load_block_boundaries()` alongside the existing
`load_compartment_boundaries()`. All four added to `av1_aux_data_resolution_check.ipynb` (extraction
+ `screen_covariate()` + trust-score entries) and to `xgb_environmental.py`'s
`FEATURE_PROVENANCE`/`spatial_position_edge_effects` category — the unified feature set is now
39 variables (was 35).

**Real result, not just added for completeness**: `spatial_position_edge_effects`'s grouped
permutation importance went from negligible (-0.001 R² with just the original 2 variables) to a
genuine ΔR²=0.040 (5th of 7 categories) with all 6 — the new variables carry real signal, not
noise. 4survey `all_environmental` test R² rose from 0.567 to 0.612 (XGBoost) and Elastic Net's
test R² rose to 0.703 — a modest, plausible improvement, unlike the earlier `Age` case (a huge,
suspicious jump that turned out to be circular with the target).

## Systematic rebuild: retire Top_Height99 and yldc, restructure the pipeline (28-29 July 2026)

**Two decisions triggered a full, systematic rebuild rather than a quick patch.**

**1. `yldc` removed as a feature everywhere.** While reviewing `av1_grouped_category_importance.ipynb`,
asked whether `yldc` (Forestry Commission Yield Class) might be circular with the height target,
the way `Age` was found to be earlier this session. Researched how General Yield Class is
actually calculated (Edwards & Christie 1981, Forest Research Booklet 48): a deterministic
function of a stand's own measured top height and age via species-specific yield curves — the
user then confirmed `yldc` specifically comes from an external FC inventory *polygon layer*, not
computed from this survey's own rows (ruling out the most severe, direct form of circularity;
the dataset's own `GYCspec95`/`GYCspec99` columns ARE that direct per-survey recomputation, and
they are correctly never used as features). But the real, decision-relevant test — an actual
held-out ablation refit, not a correlation check — showed `yldc` hurts generalisation regardless
of mechanism, in every model checked:
- `xgb_environmental` (predicts the CR residual): val R² 0.649→0.729, test R² 0.612→0.617 without it.
- RF baseline (predicts height directly): test R² 0.446→0.498 without it (+0.052, ~12% relative).
- DNN (`dnn_noenv`, local smoke test): test R² 0.606→0.647 without it (+0.041, ~7% relative) —
  despite val LOSS looking marginally *better* with it during training, a classic sign it fits
  patterns near the validation compartments that don't generalise to test compartments.

Used as a real feature in `models/rf_baseline/rf_baseline.py`, `models/linear_baseline/
linear_baseline.py`, `models/common/torch_data.py` (drives `dnn_noenv`/`pinn_noenv`, including
both endpoints of PINN's trajectory-pair loss), and this session's `models/xgb_environmental/`/
`models/elasticnet_environmental/`. NOT used by Chapman-Richards or average-by-age (Age-only).
Its other use (`models/common/data.py::filter_data()`, a row-filter bound, not a feature) is
unrelated and stays as-is. Still present in the consolidated `model_table.parquet` for audit —
just not selected as a feature by any model now.

**2. Target changed from `Top_Height99` to raw `elev_percentile_95th`.** Verified with real data:
`Top_Height99 = elev_percentile_99th` exactly (already unadjusted — but retired entirely per
explicit instruction, "we aren't using that one"). `Top_Height95 = elev_percentile_95th × 1.1`
(confirmed exactly: e.g. 54.508148 × 1.1 = 59.958963) — the ×1.1 correction compensates for known
underestimation of top height at P95 from dense Sitka foliage. The new target is the smaller,
unadjusted `elev_percentile_95th` — not `Top_Height95`, not `Top_Height99`. This column exists in
the raw GPKG but was previously dropped during cleaning specifically to avoid leaking
`Top_Height95` as a predictor.

**Vol95/GYCspec95 kept as forestry-facing evaluation tools, decoupled from the model target.**
These are pre-computed FR-inventory fields (not derived by this project's own code) built from
the OLD `Top_Height95`, useful to foresters when reporting results (volume, yield class) even
though the modelling target has changed. Decision: keep `Vol95`/`GYCspec95` (and the
`Top_Height95` they're computed from) as audit-only columns, explicitly documented as still
referencing the retired height definition, not recomputed against the new target. Drop
`Vol99`/`GYCspec99` (the `Top_Height99` family) entirely.

**Rebuild done as a systematic, phased, checklisted process** (full plan retained in
`/Users/shreeyagorasia/.claude/plans/eventual-churning-pumpkin.md` at the time of writing), not a
quick patch, per explicit instruction — checked/verified after each phase before moving on:

- **Cleaning notebook → script.** `notebooks/data_exploration/lidar_years_all_data_cleaning.ipynb`'s
  actual cleaning/filtering/export logic (six-stage funnel: cohort years → complete survey
  coverage → Sitka spruce → valid planting year → plausible age → plausible height) reproduced in
  new `data_processing/clean_master_data.py`, now retaining `elev_percentile_95th` instead of
  dropping it. Verified: identical row/plot counts to the old pipeline (287,064/71,766 for
  4survey, 83,382/13,897 for 6survey) — this was a pure addition, not a filtering change. The
  height-plausibility filter (Stage 6, 0-60m bounds) now applies directly to the new, ~9% smaller
  raw column rather than being rescaled — checked empirically this changes zero rows either way
  (stage-6 count identical to stage-5 count, before and after), so the decision was inconsequential
  here, not a live risk.
- **Consolidated model tables.** `data_processing/export_model_tables.py` rewritten: checked
  directly that `dnn_noenv.parquet` and `pinn_noenv.parquet` were BYTE-FOR-BYTE IDENTICAL (same 14
  columns, `.equals()` True), and the other three per-model files were each a subset of the same
  core columns — genuine duplication, not just "many small files". Replaced with ONE
  `model_table.parquet` per cohort (the union of every column any model needs), each model's own
  `FEATURE_COLUMNS` selecting its own subset at load time. `models/common/data.py::load_model_table()`
  simplified to no longer take a `table_name` parameter (there's only one file now).
- **Chapman-Richards fit degeneracy found and fixed.** The rebuilt CR fit's `y_max` landed EXACTLY
  on the single tallest training tree (51.91116000, to 8 decimal places) — a classic sign
  `curve_fit` got pinned to a boundary rather than finding a real asymptote. Traced to
  `chapman_richards.py`'s own `lower_bounds = [max_observed_height, ...]` — the floor was exactly
  the observed max, not strictly above it. Confirmed empirically this ALSO happens under the OLD
  target (refit `elev_percentile_99th` with the identical code on the identical rows: y_max also
  landed exactly on ITS max, 53.490917, with a very similar shape parameter p≈0.859 vs the new
  target's p≈0.865) — a pre-existing fragility, not something the target change caused. Fixed:
  lower bound is now `max_observed_height * 1.001`. The fitted `y_max` still sits close to that
  floor even after the fix (51.963 vs the 51.911 max) — this is a separate, genuine data
  characteristic (weak asymptote identifiability given the observed age range), not something
  further boundary-tweaking would resolve, documented as an open caveat rather than chased further.
- **Baselines re-verified across all four split types** (`plot_level`, `spatial_block`,
  `temporal`/wide-gap, `temporal_narrow_gap`) — same qualitative pattern as the retired pipeline in
  every case (e.g. RF wins `plot_level`, loses to linear under `spatial_block`), confirming the
  rebuild changed the numbers, not the underlying story. Real numbers in `experiment_log.md`'s
  `baselines_rebuild_2026-07-28` row.
- **DNN/PINN smoke tests** (80 epochs, both cohorts, `spatial_block` and `temporal`) all ran clean
  after fixing one real bug caught along the way: a renamed transition-table column
  (`annual_height99_increment`→`annual_height_increment`, part of retiring the old target name)
  was missed in `print_pre_training_diagnostic()` on the first pass, causing a `KeyError` — fixed,
  re-verified. Result pattern matched history: DNN clearly ahead of untuned PINN (`physics_weight=1.0`)
  on 4survey, close on 6survey — same shape as before the rebuild.

**A real mistake, caught and contained.** The smoke tests above were run without a distinct
`--run-name`, so they silently overwrote the real, previously-reported full 500-epoch DNN/PINN
checkpoints/predictions/metrics at the default output paths (`outputs/spatial_block/dnn_noenv/
<cohort>/`, etc. — confirmed via file timestamps, all dated 2026-07-16 before, 2026-07-28/29
after). This is exactly the failure mode `run_dnn_noenv.py`'s own module docstring warns about,
and had already been correctly avoided once earlier this session (a `yldc`-ablation smoke test
used a distinct `--run-name`) — missed here. Confirmed recoverable: the cluster's own copies of
these exact files were untouched (dated 2026-07-16, verified directly by the user on the cluster)
since this session has no cluster access. Given the retired pipeline's numbers are already fully
documented in `experiment_log.md` regardless, and the user confirmed a "fresh log" is actually
preferable (the run-log's real purpose is tracking current hyperparameters/results, not an
immutable historical archive) — resolved by archiving the ENTIRE local `outputs/` directory
(~4.0GB, everything pre-dating this rebuild, including the accidentally-overwritten smoke tests)
wholesale to `legacy/2026-07-28/outputs/`, then regenerating baselines fresh locally. Going
forward: every exploratory/smoke run must pass a distinct `--run-name`, no exceptions.

**Local/cluster sync.** This repo uses git for code + rsync for large gitignored data/outputs
(`data/raw/`, `data/interim/`, `data/processed/`, `outputs/` are all fully gitignored). Also
discovered mid-session: the local working directory was on branch `spatial_attribution`, while
the cluster was on `baseline_models` — checked `git merge-base --is-ancestor baseline_models
spatial_attribution` (true, confirmed both locally and via `origin/baseline_models`) before
fast-forwarding `origin/baseline_models` to match `spatial_attribution`'s tip
(`git push origin spatial_attribution:baseline_models`), so the cluster's existing branch just
needed a normal `git pull`, no branch switch. The 5 old superseded per-model parquet files were
moved to `legacy/2026-07-28/data/processed/current_state/<cohort>/` (mirroring the local
cleanup) before transferring the 6 new consolidated/master/transition parquet files up. Baselines
were regenerated locally (not on the cluster's head node, which doesn't allow direct compute) and
the whole local `outputs/` transferred up wholesale via rsync, since the fits are deterministic
(fixed seeds) and cheap.

**Real cluster jobs submitted (2026-07-29), sequenced deliberately.** DNN/PINN at the plain,
untuned base case (`physics_weight`/`trajectory_weight` left at the model default of 1.0, single
seed 42), `--max-epochs 500`, both cohorts, `spatial_block`/`temporal`/`temporal_narrow_gap`.
**Deliberately NOT re-running the physics-weight sweep or the 40-job reseed check yet** — those
were expensive, deliberate investments that concluded `physics_weight=0.05` beats the untuned
default, built entirely on the retired pipeline. Re-running them now, before knowing whether the
target/`yldc` change moved the base numbers enough to matter, risks a lot of cluster time on a
question that might not even still be open in the same way. Plan: compare the base-case numbers
against the retired pipeline's own base case first, then decide explicitly whether the
sweep+reseed is still warranted.

**Still pending**: the real cluster job results (not yet back); Phase 6 (opportunistic cleanup of
now-dead code in touched files); the `models/spatial_attribution/` rename (naming collision with
"attribution", still deferred); the pure-SS dataset switch (still a separate, deferred decision —
see `legacy/pure_SS_dataset/pure_ss_vs_current_dataset_comparison.ipynb`'s own numbers: only
24.1%/40.8% of rows retained, "should be treated as a new dataset version").

**Cluster jobs came back bogus, root-caused and fixed (2026-07-29).** All 12 real DNN/PINN
cluster jobs "completed" in ~53 seconds each (exit code 0:0) — impossible for 500-epoch training.
The job logs showed why: `FileNotFoundError` on `data/processed/current_state/4survey/
model_table.parquet` — the cluster's `git pull` brought the new code (which already expects this
consolidated file, confirming the cluster branch was current), but the file itself is gitignored
and had never actually been rsynced up, despite an earlier note in this log describing that
transfer — that transfer either didn't happen or went to the wrong place. The SLURM wrapper
script doesn't propagate the Python failure as a job failure (no `set -e`), so `sacct` reported
`COMPLETED` on a job that never trained anything. Fixed by rsyncing `data/processed/
current_state/` and `outputs/chapman_richards/` (the re-fit CR params PINN's physics anchor
needs) up to the cluster directly; user re-submitting the 12 jobs from a clean state. **Lesson
for next time**: `sacct`'s `COMPLETED`/exit-code-0 is not sufficient evidence a cluster job did
real work when the wrapper script can't propagate a Python-level failure — check wall-clock time
against a sane expectation (500 epochs is not a 53-second job) and confirm the actual output
directory exists before trusting the SLURM record.

**Phase 5 complete (2026-07-29): environmental attribution chain re-run against the new
target/no-`yldc` pipeline.** Removed `yldc` from `xgb_environmental.py`'s `FEATURE_PROVENANCE`
and `grouped_analysis.py`'s `CATEGORY_GROUPS` (`stand_structure`) — confirmed via grep, nothing
else in either package or `elasticnet_environmental/` still referenced `yldc` or `Top_Height99`.

Before re-running, had to recover `av1_aux_data_resolution_check.ipynb`'s re-execution from a real
hang: the earlier plain `jupyter nbconvert --execute --inplace` background run had been running
13 hours with only 30 seconds of CPU time used — silently stuck, not slow (confirmed via `ps`
elapsed-vs-CPU-time and zero file writes the whole time). Root cause not conclusively identified
(the two cells that had been reset for re-execution were both plain, fast CPU-bound code — a
`cKDTree` neighbour query and a JSON-based CR-residual calc — neither should hang), but rather
than keep guessing, switched to a cell-by-cell driver (`nbclient.NotebookClient`, run manually
instead of through plain `nbconvert`) that saves the notebook to disk after every single cell and
bounds each cell to a 5-minute timeout — this makes a real hang visible within minutes instead of
silently for half a day. Re-ran cleanly (98/98 cells, ~14 minutes total, longest single cell
37s). This driver script is worth reusing for any future notebook re-execution in this repo,
plain `nbconvert --execute` no longer trusted at face value for anything long-running.

Also discovered mid-recovery: both `av1_aux_data_resolution_check.ipynb` and
`av1_grouped_category_importance.ipynb` were open in **live Jupyter browser sessions** while being
re-executed on disk — a real risk (an earlier session's note on the cleaning notebook already
flagged this exact race: a live tab's autosave can silently overwrite a fresh execution with
stale in-memory state). For `av1_aux_data_resolution_check.ipynb` the risk was already taken (executed
in place); flagged to the user to reload/close that tab rather than save over it. For
`av1_grouped_category_importance.ipynb`, used the safer pattern instead: copied to a temp file in the
same directory, executed the temp copy, verified it, then moved it over the original only after
confirming success — never executed the live file directly.

**Results** (`xgb_elasticnet_environmental_2026-07-29` row in `experiment_log.md` has the full
numbers): re-derived `plot_environmental_features.parquet` (residual now built on
`elev_percentile_95th`), then `run_xgb_environmental.py` / `run_elasticnet_environmental.py` /
`av1_grouped_category_importance.ipynb` all re-run cleanly. 4survey `all_environmental`: XGBoost val
R²=0.734/test R²=0.629, Elastic Net val R²=0.700/test R²=0.671 — broadly similar shape to the
retired pipeline's numbers (val/test both around 0.6-0.7), same qualitative story (neighbour
spatial-lag dominates grouped permutation importance, ~10x every other category; real spatial
autocorrelation remains in the residuals, Moran's I=0.197 vs the retired pipeline's 0.243, still
p=0.005). **New observation, not present before**: 6survey's val R² sits well below its own test
R² across every feature set for BOTH XGBoost and Elastic Net (e.g. XGBoost all_environmental: val
R²=0.107 vs test R²=0.398) — the opposite of the usual overfitting direction, and consistent
across two independent model types, so probably a real property of which compartments
`spatial_block_split` assigned to val vs test for the smaller cohort rather than noise in one
model. Not yet investigated further — flagged for whenever 6survey's environmental story gets
written up. Every markdown cell in `av1_grouped_category_importance.ipynb` was already written
number-agnostic ("how to read this chart", not "X equals Y") — checked and confirmed nothing
stale needed fixing, satisfying the "check markdown too, not just numbers" standing practice
without requiring any edits this time.

**Phase 6 complete (2026-07-29): cleanup pass.** Ran `pyflakes` across every file this rebuild
touched (`data_processing/`, all of `models/`) — one real finding: `elasticnet_environmental.py`
imported `FEATURE_PROVENANCE` from `xgb_environmental.py` but never used it (only referenced in a
comment) — removed. Whole tree clean after that. Checked several other candidates that turned out
to be legitimate, not dead: `models/common/data.py`'s `yldc`-based row filter (`filter_data()`'s
`yldc_min`/`yldc_max` bounds) is a genuinely different, deliberately-kept use of the raw column
(filtering rows by yield class, not using it as a model feature); `run_baselines.py`/
`evaluate_baselines.py`'s `table_name` parameter still does real work (distinguishing
`load_cohort_data()`'s trimmed view from `load_model_table()`'s full view, both now backed by the
same consolidated `model_table.parquet`) — already correctly commented, not vestigial. Re-ran
`elasticnet_environmental` after the import removal to confirm nothing broke — same numbers as
before.

**Found and fixed, opportunistically**: `baseline_results.ipynb` sections 8-9 (the actual
dissertation-argument section, not just a chart caption) were still stale despite Phase 4 being
marked complete — section 9 said `target Top_Height99` and listed XGBoost+SHAP/
`temporal_narrow_gap` as "not started yet" (both now done); section 8 cited specific old RMSE
percentages with no acknowledgment they predate the rebuild, and its `yldc`-as-GYC-proxy argument
sat oddly next to `yldc` having just been removed for hurting generalisation. Asked the user how
to handle it rather than silently rewriting the dissertation's own argument text — chose "caveat
now, rewrite later": section 9's facts were corrected directly (safe, they're just wrong
otherwise); section 8 got a dated pending-re-verification note (the specific percentages need the
real cluster DNN/PINN numbers once they're back) without touching the argument itself, since the
qualitative reasoning is expected to still hold but isn't yet confirmed. Both edits done via
surgical raw-text replacement (the file is too large for the Read/Edit/NotebookEdit token limit)
following the same technique the cleaning-notebook lesson established earlier — verified valid
JSON, execution counts intact, diff scoped to exactly the two touched cells each time.

**Phase 7 (2026-07-29): archived the superseded cleaning notebooks.** Moved
`lidar_years_all_data_cleaning.ipynb` and `lidar_years_all_data_understanding.ipynb` (git mv, not
a delete) to `legacy/2026-07-28/` now that `data_processing/clean_master_data.py` fully replaces
their export logic — neither was open in a live Jupyter session, confirmed via the API before
moving. Updated the three direct references (`clean_master_data.py`'s own header comment,
`lidar_years_cleaning_findings_summary.md`, and `README.md`'s "Active workflow" section, which
described the retired notebook-driven pipeline as current) so nothing points at a moved file.

**Found, NOT fixed — flagged for a separate decision**: `README.md`'s "Repository structure"
section is stale in ways that predate this whole rebuild — it describes `data_exploration_gpkg/`
and `results_notebooks/` directories that no longer exist (both were already reorganized to
`notebooks/data_exploration/` and `notebooks/model_results/` before this session's work started).
Out of scope for this rebuild's Phase 6 (limited to files the rebuild itself touched) and Phase 7
(limited to the specific docs the plan named) — left as-is rather than doing an unplanned,
unscoped README rewrite; worth a dedicated pass whenever the user wants it.

**`xgb_environmental.py`'s TODOs filled in (2026-07-29).** The user added `#TODO` markers to
several `FEATURE_PROVENANCE` entries asking for the actual formula/source rather than the
one-line summary, plus a syntax typo (stray `=` after the `soilgrids_ph` line, breaking the whole
module on import — fixed, clearly accidental, not an intentional edit). Traced every "own
calculation" feature back to where it's actually computed in
`av1_aux_data_resolution_check.ipynb` and wrote the real formula into each entry (not guessed):
`slope_degrees`/aspect via `np.gradient` on the 50m DTM grid; `northness`/`eastness` = cos/sin of
that aspect; `profile_curvature`/`plan_curvature` (Zevenbergen & Thorne 1987, sign-validated
against a synthetic bowl/dome); `tpi` = elevation minus a 100m-window mean
(`scipy.ndimage.uniform_filter`); `elevation_roughness` = windowed std via the
Var=E[X^2]-E[X]^2 identity; `solar_radiation_index` = the standard slope/aspect solar-noon,
summer-solstice formula (no horizon shading — a known simplification, not fixed); `frost_hollow_flag`
= TPI below its own 15th percentile AND concave `plan_curvature` (an uncalibrated heuristic
threshold, not validated against real frost data); `topex`/`windward_topex` (Wilson 1984
horizon-angle sum, 1000m radius, 8 directions vs. a single 225°/SW bearing for the windward
version); `dist_to_cpmt_boundary`/`dist_to_forest_perimeter` (compartment polygon geometry, the
latter via +60m/-60m morphological closing, dropping <1ha closing artefacts);
`dist_to_watercourse` (distance to nearest OS Open Rivers line, a real vector survey, not a
flow-accumulation derivation); `neighbour_mean_height`/`neighbour_height_differential` (`cKDTree`,
75m radius, 2023 heights only). `haduk_tas_2021_mean`'s TODO was a real, still-open limitation
(every survey year reuses the single 2021 raster) — written up as a known limitation rather than
silently marked resolved, since it hasn't actually been fixed (would need one HadUK-Grid raster
per survey year, joined by year).

**Full aspect_degrees-exclusion reasoning** (shortened to a pointer in the file itself, per the
user's own TODO to move it here): aspect_degrees is a raw compass bearing (0-360) — 359 and 1 are
almost the same direction but numerically far apart, a bad input for any model to split on
directly. northness/eastness (cos/sin of the same aspect) are its fixed replacement and used
instead everywhere. This is the only column excluded from `ALL_FEATURE_COLUMNS` for a
data-validity reason — everything else, including `inverse_slope_proxy` (renamed 2026-07-30 from
`soil_depth_proxy`, which named it for what it's used as a stand-in for rather than what it
actually is — an exact transform of `slope_degrees`), is deliberately left in raw and not
pre-judged as redundant; that decision is
left to the real SHAP/permutation-importance evidence the Tier-2 notebook produces, not assumed
in advance.

**Found, flagged for a decision, NOT changed**: the user's edit also removed
`dist_to_scpt_boundary`, `dist_to_block_boundary`, `cpmt_compactness_ratio`, and `dist_to_road`
from `FEATURE_PROVENANCE` entirely (not just from `TERRAIN_AND_WIND_COLUMNS`), and the whole
`stand_structure` category (`CanopyCover`, `Thin`, `time_since_thinning`,
`time_since_thinning_missing`, `recent_thinning_5yr`) is gone too — none of these had a missing
formula (all were previously documented, real GPKG-geometry or raw-survey-field calculations), so
this looks like a deliberate feature-set trim rather than something the TODOs were flagging.
`grouped_analysis.py`'s `CATEGORY_GROUPS` still references all 9 of these columns (`stand_structure`
category plus 4 `spatial_position_edge_effects` entries) — a real mismatch that will `KeyError` the
next time grouped permutation importance or Moran's-I-by-category actually runs (the
`check_category_groups_complete()` guard only checks one direction — a feature-set column missing
its category — not the reverse, so it stayed silent).

**Resolved**: user confirmed the trim was not intentional — restored all 9 columns to
`FEATURE_PROVENANCE` (their original descriptions, plus real formulas for the 4 geometry columns
traced the same way as the rest of this entry: `dist_to_scpt_boundary`/`dist_to_block_boundary`
via GPKG sub-compartment/block polygon boundaries, `cpmt_compactness_ratio` = compartment
perimeter/area, `dist_to_road` = distance to nearest OS Open Roads line, same method as
`dist_to_watercourse`) and back into `TERRAIN_AND_WIND_COLUMNS`. `ALL_FEATURE_COLUMNS` is back to
38 (from the trimmed 29), now matching `CATEGORY_GROUPS`'s union exactly in both directions
(checked programmatically, not just by eye). Since Phase 5's `xgb_environmental`/
`elasticnet_environmental` run happened before this trim occurred, the restored 38-feature set
matches what those numbers were actually computed on — re-ran `run_xgb_environmental.py` to
confirm: 4survey `all_environmental` val R²=0.734/test R²=0.629, exactly matching the row already
in `experiment_log.md`, so no correction needed there.

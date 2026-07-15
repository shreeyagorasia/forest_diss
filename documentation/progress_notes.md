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
└── pinn_noenv/          # pinn_noenv.py (network+physics losses+fit/predict) + run_pinn_noenv.py (FIT)
                         # + evaluate_pinn_noenv.py (EVALUATE, local)

data_processing/
└── export_model_tables.py   # standalone, no notebook dependency (see below)

outputs/
├── <model_name>/<cohort>/   # gitignored, regenerate by running the model
└── run_logs/                # gitignored, one JSON per run attempt — see "Run logging" below

results_notebooks/
└── baseline_results.ipynb       # reads outputs/ only, never refits — sklearn/CR baselines only,
                                  # covers all three split types (plot_level, spatial_block, temporal)

data_exploration_gpkg/notebooks/
├── lidar_years_all_data_cleaning.ipynb        # the cleaning funnel, still notebook-only
└── spatial_temporal_split_visualisation.ipynb # maps/visualises the three split types

jobs/                  # SLURM submission scripts (user-maintained)
├── dnn/run_dnn_noenv.sh
├── pinn/run_pinn_noenv.sh
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
DNN/PINN train under `temporal_split` specifically (train on early years, early-stop on 2021,
held-out test on 2023) — see "DNN and PINN (no-environment)" below. `spatial_block_split` is not
yet wired into DNN/PINN. Visualised in `spatial_temporal_split_visualisation.ipynb`.

***

## DNN and PINN (no-environment)

Both trained under `temporal_split` (train on early years → early-stop on 2021 → held-out test on
2023), age + thinning-status features only, no terrain/wind covariates yet (matches the baselines'
current feature set). Shared architecture: `NoEnvNetwork`, 3 hidden layers, 128 neurons, LeakyReLU
(`models/common/torch_model.py`).

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

1. **Push cleaned data to the cluster** (`data/processed/` is gitignored, so it never travels via
   git): `rsync` `data/processed/master/*.parquet` to the cluster, then on the cluster run
   `python -m data_processing.export_model_tables` to regenerate `data/processed/current_state/`
   there (proven byte-for-byte deterministic — safe to regenerate on a different machine).
2. **Submit real training jobs** on the cluster: `jobs/dnn/run_dnn_noenv.sh` and
   `jobs/pinn/run_pinn_noenv.sh` (these call `run_dnn_noenv.py`/`run_pinn_noenv.py` with the real
   `--max-epochs 500`, not a quick sanity check). Cluster logs land in `logs/<model>/<job>_<id>.out`
   /`.err`.
3. **Pull results back**: `rsync` `outputs/dnn_noenv/`, `outputs/pinn_noenv/`, and
   `outputs/run_logs/` down from the cluster to the laptop.
4. **Evaluate locally** (cheap, CPU): `python -m models.dnn_noenv.evaluate_dnn_noenv` and
   `python -m models.pinn_noenv.evaluate_pinn_noenv` (omit `--cohort` to run both 4survey and
   6survey). This writes the real `metrics.json`/`predictions.csv` and its own run-log entry.

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
7. `results_notebooks/baseline_results.ipynb` — how metrics get read back and compared across splits.

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

***

## What's not started yet

Real (500-epoch) DNN/PINN training runs on the cluster (code is built and verified with short
sanity-check runs, but no full-length results exist yet as of 15 July 2026). Also not started:
terrain/wind feature extraction, XGBoost + SHAP, `spatial_block_split` wired into DNN/PINN, and a
DNN/PINN results notebook comparable to `baseline_results.ipynb` (waiting on real training results
to exist first).

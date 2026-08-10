# Clean retrain, 2026-08-10 -- priority tiers

DRAFT plan, not yet run. You asked for "every single script to re-train everything" as a final
check before sleep -- something you MAY do, not a decision made yet. This is that: every command
needed, split into one file per tier so you can run exactly the ones you want and skip the rest.

See `_lib.sh` for the full naming-convention writeup (old inconsistent patterns vs. this batch's
`clean_<model>_<featuretag>_<split>[_fold<N>]_seed<N>`) and the shared helper functions the
`_fit.sh`/`_evaluate.sh` scripts call.

## Local vs. cluster -- how it actually works

Nothing in these scripts auto-detects or switches machines. It's purely: which machine you're
logged into when you run the file, plus whether a line calls `sbatch` (only exists on the
cluster) or calls `.venv/bin/python`/`rsync` directly (runs on whatever machine you type it on).
So every tier is **files meant for exactly one machine each** -- mixing them doesn't work on
either machine:

- `tierN_fit.sh` -- **the cluster** (GPU fits, `sbatch` calls). Exception: `tier1_fit.sh` is
  baselines, which are CPU-only and fast -- that one runs **locally** instead.
- `tier1_push.sh` -- **your Mac**, after `tier1_fit.sh` finishes. Uploads the resulting CR-anchor
  file to the cluster (every PINN fit job needs it, wherever it runs).
- `tierN_sync.sh` -- **your Mac**, after `squeue` on the cluster shows that tier's fit jobs
  `COMPLETED`. Downloads only the specific run folders that tier just fit.
- `tierN_evaluate.sh` -- **your Mac**, after the sync finishes. Evaluation never runs on the
  cluster for this project -- one cheap CPU forward pass over an already-fit checkpoint.

Each file's required machine is labelled in its own header comment.

`tier1_push.sh` and every `tierN_sync.sh` are **fully explicit, standalone rsync commands** --
the filter rules (which folders get copied) are written out literally as a heredoc inside the
file, not computed at runtime by sourcing `_lib.sh`. That means you can open any of them, read
exactly what will and won't be touched, and copy-paste the whole thing straight into your own
terminal instead of running the file at all, if you'd rather not run a script of mine against
your cluster account.

## Sync, not wholesale rsync

Your existing habit was `rsync -avz ... hastings:~/forest_diss/outputs/ outputs/` -- the WHOLE
tree, every time. That's the likely mechanism behind the local/cluster mix-up this session's audit
found: with no merge logic, whichever side you last rsync'd from silently wins for every folder
that changed on both sides, with no record of which. Every sync/push script here instead lists,
by name, only the exact run folders that specific tier touched -- everything else in `outputs/`
is left alone, even if it differs on both sides. Same connection details as your own command
(`s2887183@student.ssh.inf.ed.ac.uk` jump host, `s2887183@hastings.inf.ed.ac.uk` target,
`~/forest_diss/outputs/` remote path) and the same `growth_curve_attribution/gnnwr/models/`
exclusion kept as a standing safety default on every download.

## Why tiers, not one big job

Literally everything ("every split, 5-fold vs one-fold, every feature set, every seed") is
roughly **2,200 jobs total, ~1,100 of them real GPU fits**. At your 20-concurrent-job association
limit, even an optimistic 10-15 min/fit average is 11-14 hours of GPU queue time assuming zero
contention from other students on the shared Teaching partition -- realistically 1-3 days, not
overnight. I can't give a precise ETA either way (this project's own logs record identical
code/seed swinging 0.10 in R2 across different physical nodes, so per-job timing here is
genuinely variable) -- the numbers below are honest planning input, not a promise.

## Priority list

Each tier's "Jobs" column is fit + evaluate combined (baselines' fit+push count as tier 1's
"jobs" too, even though neither touches the GPU).

| Tier | What | Jobs (fit+eval) | Cum. jobs | Cum. GPU fits | Why |
|---|---|---|---|---|---|
| 0 | E6 plot_level stage4 gap-fill, seed 42 only | 12 | 12 | 6 | Already-known-broken, cheapest fix |
| 1 | Baselines (local), every split incl. all 5 kfold folds | 16 | 28 | 6 | Near-free, no GPU at all; also the PINN CR-anchor prerequisite for every tier below |
| 2 | env_terrain @ set2 (default), 5-seed, `spatial_block` | 60 | 88 | 36 | Closes: env-conditioned DNN/PINN comparison currently has ZERO seed-robustness |
| 3 | no-env, 5-seed, `spatial_block_kfold` (all 5 folds) | 200 | 288 | 136 | Closes: the POOLED/CI'd headline no-env result is still 1 seed/fold |
| 4 | env_terrain @ set2, 5-seed, `spatial_block_kfold` | 300 | 588 | 286 | Same gap as tier 3, env_terrain side |
| 5 | env_terrain @ set2, 5-seed, `temporal` | 60 | 648 | 316 | Secondary RQ (temporal generalisation) |
| 6 | no-env + env_terrain @ set2, 5-seed, `plot_level` | 100 | 748 | 366 | Lowest value -- ledger calls plot_level "the easy case, not a generalisation test" |
| 7 | Full feature-tier reseed (set3a/set3b/set4 x 5-seed x every split, on top of tiers 2/4/5/6's set2-only coverage) | 1,440 | 2,188 | 1,086 | Nice-to-have only -- E6 already answers "does more env help" at single seed; this mostly reseeds an answered question. Deliberately overlaps tier 0 by one cell (set4/plot_level/seed42) -- harmless, not worth special-casing out |

**Recommended cutoff: tiers 0-3** (288 jobs, 136 GPU fits, roughly 1.5-3 hours of GPU queue time
with a quiet cluster -- plausibly done overnight). Tiers 4-6 add real value with more time budget.
Tier 7 is the one I'd skip unless deliberately committing multiple days.

## Run-name patterns per tier -- for updating the results ledger later

Every folder this batch creates under `outputs/` starts with `clean_` (baselines are the one
exception -- they keep their plain default names, `chapman_richards`/`linear_baseline`/
`rf_baseline`/`average_by_age`, since there's only ever one baseline configuration per split/fold,
nothing to disambiguate with a prefix). When it's time to fold results into the ledger, this is
what to look for on disk (`{42-46}` means one folder per seed, `{0-4}` means one per kfold fold):

| Tier | Run-name pattern(s) | Where (relative to `outputs/`) |
|---|---|---|
| 0 | `clean_dnn_env_terrain_set4_plot_level_seed42`, `clean_pinn_env_terrain_set4_plot_level_seed42`, `clean_pinn_env_terrain_k_set4_plot_level_seed42` | `<run_name>/<cohort>/` (no split prefix -- plot_level) |
| 1 | `chapman_richards`, `linear_baseline`, `rf_baseline`, `average_by_age` (+ `_fold{0-4}` suffix for kfold) | `<run_name>/<cohort>/` for plot_level; `<split>/<run_name>/<cohort>/` for spatial_block/temporal; `spatial_block_kfold/<run_name>_fold{0-4}/<cohort>/` for kfold |
| 2 | `clean_{dnn_env_terrain,pinn_env_terrain,pinn_env_terrain_k}_set2_spatial_block_seed{42-46}` | `spatial_block/<run_name>/<cohort>/` |
| 3 | `clean_{dnn_noenv,pinn_noenv}_spatial_block_kfold_fold{0-4}_seed{42-46}` | `spatial_block_kfold/<run_name>/<cohort>/` |
| 4 | `clean_{dnn_env_terrain,pinn_env_terrain,pinn_env_terrain_k}_set2_spatial_block_kfold_fold{0-4}_seed{42-46}` | `spatial_block_kfold/<run_name>/<cohort>/` |
| 5 | `clean_{dnn_env_terrain,pinn_env_terrain,pinn_env_terrain_k}_set2_temporal_seed{42-46}` | `temporal/<run_name>/<cohort>/` |
| 6 | `clean_{dnn_noenv,pinn_noenv}_plot_level_seed{42-46}`, `clean_{dnn_env_terrain,pinn_env_terrain,pinn_env_terrain_k}_set2_plot_level_seed{42-46}` | `<run_name>/<cohort>/` (no split prefix -- plot_level) |
| 7 | `clean_{dnn_env_terrain,pinn_env_terrain,pinn_env_terrain_k}_{set3a,set3b,set4}_<split>[_fold{0-4}]_seed{42-46}` | same split-prefix rule as above, per `<split>` |

`<cohort>` is `4survey` or `6survey` -- always a subfolder under the run name, never part of the
run name itself (one fit per cohort, but they share the same run-name folder). Inside each run
folder: `metrics.json` (the number you want), `predictions.csv` (per-row residuals, needed for
Moran's I/semivariogram), `checkpoints/` and `preprocessing/` (not needed for the ledger).

## How to run a tier

From the project root, on the machine each file's header names:

```
# on the cluster:
bash jobs/retrain/tier0_fit.sh
# ... wait for squeue to show COMPLETED ...

# on your Mac:
bash jobs/retrain/tier0_sync.sh
bash jobs/retrain/tier0_evaluate.sh
```

Before any tier that fits a PINN model, run tier 1 first: `bash jobs/retrain/tier1_fit.sh` (your
Mac, fits baselines locally), then `bash jobs/retrain/tier1_push.sh` (your Mac, uploads the
resulting CR anchor to the cluster) -- every PINN fit job reads that anchor from wherever it runs.

## Feature-tier tags used above

`set2` = `terrain_wind_solid` (current default, 5 vars) &middot; `set3a` = `stage1_terrain` (13
vars) &middot; `set3b` = `stage2_terrain_wind` (21 vars) &middot; `set4` =
`stage4_all_environmental` (33 vars, max set). Matches the Set numbers in the results ledger's own
feature-sets tracking table. Set 3c (`stage2_terrain_wind_plus_temperature`) is coded but was
deliberately dropped from the dissertation plan for time -- excluded everywhere here on purpose.

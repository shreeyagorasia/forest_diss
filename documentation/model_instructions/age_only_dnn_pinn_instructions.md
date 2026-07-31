# Instructions: No-environment DNN and PINN (Version 1)

> **Stale on specifics, 2026-07-30 — verify against the code before trusting a number here.**
> This document's *design rationale* (why derivative-consistency physics loss, why L1
> regularisation, why the DNN is a fair architectural control, why scale Age separately) is still
> the real reasoning behind `models/dnn_noenv/`/`models/pinn_noenv/` and hasn't been rewritten
> here, since it's still accurate. But the concrete values throughout are outdated, in three ways:
> - **Target column**: this document says `Top_Height99` throughout. The real target since the
>   28-29 July 2026 rebuild is `elev_percentile_95th` (raw, unadjusted) --
>   `models/common/torch_data.py::TARGET_COLUMN` is the current source of truth.
> - **Feature list**: this document includes `yldc` in every feature list shown. `yldc` was
>   removed as a feature everywhere on 2026-07-28 (real ablation showed it hurts every model's
>   generalisation) -- `models/common/torch_data.py`'s `NUMERIC_SCALED_COLUMNS`/
>   `BINARY_PASSTHROUGH_COLUMNS` (plus Age and thinning_status, see that file's own comments) are
>   the current source of truth, not the lists below.
> - **Batch sizes**: this document says "DNN = 128, PINN = 32" (from Lynch, 2025). Neither matches
>   the current code -- `dnn_noenv.py`'s `BATCH_SIZE` is 512, `pinn_noenv.py`'s is 128, and
>   whether either is actually correct for the current pipeline is an open question being
>   resolved by a batch-size sweep (see `documentation/experiment_log.md`'s 2026-07-29/30
>   entries) -- this document's specific numbers should not be treated as the current answer.
>
> If you're implementing or debugging DNN/PINN, read the *reasoning* here, but get every column
> name, feature list, and hyperparameter value from the actual code, not from this document.

**Status: DRAFT — not yet approved, do not execute.** This builds the fifth and sixth baselines in
`models/`: a plain DNN and a CR-PINN, both using the already-exported **no-environment feature
set** (`Age, CanopyCover, Thin, time_since_thinning, time_since_thinning_missing,
recent_thinning_5yr, thinning_status, yldc`) — matching `data/processed/current_state/<cohort>/
dnn_noenv.parquet` and `pinn_noenv.parquet`, which already exist on disk. No terrain/wind features
yet — those come later as Env-PINN v2/v3 (dissertation plan section 4.6), once XGBoost + SHAP have
identified which terrain/wind variables matter.

The PINN's physics term is still purely a function of `Age` (Chapman-Richards is an equation of
age, not of canopy cover or thinning), using this cohort's already-fitted, frozen `y_max/k/p`. The
DNN is built alongside it purely as a fair architectural control — same depth, width, optimiser,
and regularisation as the PINN, same input features — so any difference between the two is
attributable to the physics loss term, not to model capacity or input access.

**Status update: all open questions from the first draft are resolved**, using Lynch (2025)'s
thesis directly (checked against the actual document rather than inferred) — see section 1.

**Update — temporal split and trajectory loss added.** This document now also incorporates a
second round of changes: the split is now temporal (train on 2008+2012, test on 2023, matching the
existing `spatial_block_split` extension pattern), and the PINN gains a second physics loss term
that checks whether the network's *predicted* height change between two consecutive training
surveys matches the CR curve's growth rate — on top of, not instead of, the original per-row
CR-derivative loss. See sections 3 (trajectory data), 7 (PINN, updated), and 9 (diagnostics).

---

## 0. What already exists (reuse, don't rebuild)

- **Data tables, already exported** —
  `data/processed/current_state/<cohort>/dnn_noenv.parquet` and `pinn_noenv.parquet`. Both cohorts
  (4survey, 6survey). Columns: `identification, LiDAR_year, blk, cpmt` (metadata, never features),
  `Age, CanopyCover, Thin, time_since_thinning, time_since_thinning_missing, recent_thinning_5yr,
  thinning_status, yldc` (features), `Top_Height99` (target), `Top_Height95` (fallback/audit only,
  never a feature). No new export script needed — read these directly.
- **CR params, already fitted** — `outputs/chapman_richards/<cohort>/params.json`:
  - 4survey: `y_max=53.4909, k=0.010990, p=0.858848`
  - 6survey: `y_max=46.5132, k=0.023737, p=1.162634`
  - The PINN's physics term reads these values and treats them as **frozen constants** — it must
    never refit `y_max`/`k`/`p`.
- **Split function** — `models/common/splits.py::temporal_split()`. Already implemented and
  tested (`models/common/test_splits.py`), with a year assignment already decided:
  `4survey: train=[2008,2012], val=[2021], test=[2023]`;
  `6survey: train=[2002,2006,2008,2012], val=[2021], test=[2023]`. See
  `baseline_models_temporal_split_instructions.md` — this exact split is also being wired into the
  four existing baselines, so all six models share one split methodology. Unlike
  `plot_level_split`, the same plot legitimately appears in both train and test rows here — that's
  expected, not a leak, since this split tests time generalization, not plot generalization.
- **Filtering** — `models/common/data.py::filter_data()`. Per-plot maturity gate (Age ≥ 30 at the
  plot's 2023 survey row) plus yield-class 2–50 filter. Apply this before splitting, same as every
  other baseline.
- **Trajectory / transition data, already built** —
  `data_processing/export_model_tables.py::build_transition_table()` already produces
  `data/processed/transitions/transition_growth_{cohort}.parquet`: one row per plot per
  *consecutive* survey-year pair, with `previous_age`, `Age`, `previous_top_height99`,
  `Top_Height99`, `height99_increment`, `annual_height99_increment`
  (`= height99_increment / survey_interval_years`), and `survey_interval_years`. Checked directly:
  `Age - previous_age` equals `survey_interval_years` exactly, for every single row in both
  cohorts — so `annual_height99_increment` **is already** the observed growth rate
  (`delta_height / delta_age`) the trajectory physics loss needs. No new growth-rate computation
  required, just filtering this existing table down to pairs relevant to training (section 3).
  Note: this table currently spans *every* consecutive pair across all survey years, not
  restricted to any split — that filtering happens when it's loaded for training, same as every
  other table in this codebase.
- **Metrics** — `models/common/metrics.py::compute_metrics(observed, predicted, age=...)`. Returns
  the exact `mae/mse/rmse/r2/mre/accuracy_pct/bias` + `age_bands` dict every other baseline's
  `metrics.json` already uses. Reuse this directly — do not reimplement metric maths.
- **Saving helper** — `models/common/saving.py::save_run(output_dir, predictions_df, metrics,
  model_name, cohort, seed)`. Already writes `predictions.csv`, `metrics.json`, and
  `run_metadata.json` (with git commit hash) in one call. Reuse this; only the
  checkpoint/scaler/training-history files below are new.
- **Categorical encoding pattern to follow** — `models/linear_baseline/linear_baseline.py::
  encode_features()`. One-hot encodes `thinning_status` via `pd.get_dummies`, and — critically —
  saves the resulting column names so val/test data is `reindex`ed onto the exact same columns
  the model was trained on (a category seen only in training gets a column of 0s at eval time; a
  category never seen in training is dropped). Reuse this exact pattern for both the DNN and the
  PINN, not a fresh implementation.
- **Missing-value pattern to follow** — `models/rf_baseline/rf_baseline.py::prepare_features()`.
  `time_since_thinning` is `NaN` for never-thinned plots; fill with `0` and rely on
  `time_since_thinning_missing` (already `True`/`False`) to tell the model the `0` is a
  placeholder, not a real elapsed time. Same convention here.

## 1. Resolved decisions (confirmed this session)

- **Target column: `Top_Height99`.** Matches every other baseline (CR, average-by-age, linear,
  RF) — keeps the eventual six-model comparison table apples-to-apples. `Top_Height95` is never a
  feature or a fitting target, only carried through for audit.
- **Framework: PyTorch.** Not currently installed in `.venv` — installing it is the first
  concrete step. Chosen because its autograd makes the physics loss's `∂Height/∂Age` term
  straightforward (see section 7), and it's the framework the PINN literature cited in the
  dissertation plan (Raissi et al. 2019 and successors) mostly uses.
- **Feature set: no-environment (`dnn_noenv`/`pinn_noenv`), not age-only.** Confirmed this
  session, correcting an earlier draft of this document. Reuses the already-exported tables named
  above — model output folders are named `dnn_noenv` and `pinn_noenv` to match.
- **Batch sizes: DNN = 128, PINN = 32.** Lynch (2025) used batch size 32 for the PINN specifically
  because smaller batches gave the physics loss more gradient update steps per epoch, which helped
  convergence. The DNN comparison used a larger batch; the thesis text doesn't give one single
  explicit DNN number, so 128 (the other value named in the ablation) is used here as the DNN's
  fixed batch size.
- **Physics loss formulation: derivative-consistency (Raissi-style).** Confirmed against Lynch's
  actual implementation — Age is passed into the network as its own separate tensor precisely so
  that autograd differentiates only through Age while `CanopyCover, Thin, time_since_thinning,
  thinning_status, yldc` are held fixed as ordinary context. The physics loss compares the
  network's `∂Height/∂Age` against the CR curve's own analytical derivative (from the frozen
  `y_max/k/p`) at the same ages — never the raw predicted heights, and never `y_max` itself. See
  section 7 for the exact code pattern and section 2 for what this constraint means physically.
- **Physics weight `λ_ph = 1.0`, L1 coefficient `λ_1 = 1e-5`.** Both taken directly from Lynch
  (2025) — `λ_ph = 1.0` is stated as the peak from their Figure 4.8 physics-weight sweep, `λ_1 =
  1e-5` from their regularisation section. Used here as fixed values; the later ablation stage
  (dissertation plan section 4.7) is what actually sweeps `λ_ph`, not this build.
- **Scale every numeric input feature, not just Age.** Lynch's thesis states inputs are "scaled
  using a Standard Scaler to speed up convergence" — applied to all numeric inputs, not Age alone.
  One difference from a generic scaling setup: **Age gets its own separate scaler instance**, kept
  apart from the other numeric features, specifically so `sigma_age` (Age's training-split standard
  deviation) can be pulled out cleanly for the physics loss's chain-rule correction (section 3). If
  Age were folded into one joint scaler over all features, its individual standard deviation
  wouldn't be directly recoverable.

## 2. What the CR-derivative physics loss actually constrains

*(This section is about the original, per-row physics loss term — `λ_ph`. Section 3 below covers
the new, second trajectory loss term, `λ_traj`, which is a different constraint on different
data.)*

A natural but incorrect reading of "derivative-consistency" is that it constrains the *shape* of
the predicted growth curve — in particular its flat top near `y_max` and its flat tail for young
trees. **It does not.** At every training step, for every plot, the physics loss checks exactly
one thing: *at this plot's current age, is the network's growth rate (how fast predicted height
rises with age) close to the CR formula's growth rate at that same age?* It never looks at the
predicted height itself — only the slope.

The CR growth-rate curve has three regions across age: low near the start (age ~10–20, trees still
establishing), a peak in the middle (age ~30–50), and flattening as trees approach `y_max` (age
70+). Since this project filters to ages 30+ (see `filter_data()`) and Aberfoyle's Sitka spruce is
mostly middle-aged, the constraint mostly bites in the fast-growth middle stretch, not at the
sigmoid's top or tail. Lynch's own thesis notes the growth-rate curve "looks more linear than
non-linear" across that middle range — and that's the likely reason their physics-weight sweep
made so little difference to results: a near-linear rate constraint is easy for a network to
satisfy regardless of how strongly it's enforced. Worth keeping in mind when interpreting this
project's own results later — a small effect from `λ_ph` would replicate, not contradict, Lynch's
finding.

---

## 3. Trajectory data: growth rate and pairs (for the new trajectory loss)

This section covers what CHANGE 1/3 of the follow-up brief asked for — matching plots across
years and computing an observed growth rate — most of which turns out to already exist.

**Plot matching is not a new problem.** `identification` is already the stable per-plot ID used
throughout this codebase (every split function, every existing table). No coordinate-rounding or
new matching logic is needed. Checked directly against the actual data: **every single plot in
both cohorts appears in every survey year of its cohort** — 71,766/71,766 plots for 4survey,
13,897/13,897 for 6survey — because the cleaning pipeline already balances each cohort to a fixed
number of surveys per plot before export. There is no partial-coverage question to investigate;
full trajectories already exist for every plot that survives the maturity filter.

**The observed growth rate is `annual_height99_increment` from the existing transition table**
(section 0) — checked directly: `Age - previous_age` equals `survey_interval_years` for every row
in both cohorts, with zero mismatch, so `annual_height99_increment` (`= height99_increment /
survey_interval_years`) already **is** `delta_height / delta_age`. No new column needs to be
computed from scratch.

**Building the trajectory pairs table for training:**

1. Load `transition_growth_{cohort}.parquet`.
2. Apply `filter_data()`'s maturity gate to whichever *plots* survive it (a pair only exists for a
   plot that isn't dropped entirely by the Age≥30-at-2023 gate).
3. Keep only pairs where **both** `previous_lidar_year` and `LiDAR_year` fall in `train_years`
   (never `val_years` or `test_years`) — see the leakage rule in section 1. Concretely, for
   4survey this keeps only the single 2008→2012 pair per plot; for 6survey it keeps the
   2002→2006, 2006→2008, and 2008→2012 pairs.
4. The resulting columns needed per pair: `identification`, `previous_age`, `Age` (used to compute
   `age_mid = (previous_age + Age) / 2`), `previous_top_height99`, `Top_Height99`,
   `annual_height99_increment` (the observed growth rate), plus whichever no-environment features
   the trajectory loss's forward passes need (see section 7) — these come from re-joining each
   pair's two endpoint rows back onto `dnn_noenv.parquet`/`pinn_noenv.parquet` by `identification`
   and `LiDAR_year`, since the transition table itself doesn't carry the full no-environment
   feature set for both endpoints.
5. This pairs table is **never used as a model input feature** (`annual_height99_increment` /
   `OBSERVED_GROWTH_RATE` must not appear in `x_other` in section 7's code) — it exists purely to
   supply the trajectory loss's target values.

**Given every surviving plot has full year coverage**, expect close to 100% of training plots to
have a usable pair (4survey: exactly one pair each; 6survey: up to three). This makes the "warn if
fewer than 30% of training plots have an observed growth rate" diagnostic (section 8) very likely
to pass by a wide margin — worth keeping anyway as an actual check, since it would only fail if the
maturity filter or pairing logic behaves unexpectedly, which is exactly the kind of silent bug this
diagnostic exists to catch.

---

## 4. Shared setup (both models)

- **Inputs**: `Age, CanopyCover, Thin, time_since_thinning, time_since_thinning_missing,
  recent_thinning_5yr, thinning_status, yldc` — the full no-environment feature set.
  `thinning_status` is categorical and must be one-hot encoded (see section 0's encoding pattern).
- **Target**: `Top_Height99`.
- **Split**: `temporal_split()` — see section 1 for the exact year assignment and section 3 for
  how the PINN's trajectory pairs are built on top of it.
- **Filtering**: `filter_data()` — per-plot Age≥30-at-2023 maturity gate + yield class 2–50 (same
  as all baselines).
- Unlike CR/average-by-age (which have nothing to tune), **val is genuinely used here**: early
  stopping on validation loss for both the DNN and the PINN.
- **Missing values**: fill `time_since_thinning` NaNs with `0`, matching `rf_baseline.py` — the
  missingness flag column tells the model the `0` isn't a real elapsed time.
- **Scaling — three separate `StandardScaler` instances, not one**, matching Lynch (2025):
  - `scaler_age`: fit on `Age` only (training split only).
  - `scaler_other_features`: fit on the remaining numeric features (`CanopyCover,
    time_since_thinning, yldc`) as one group (training split only). `time_since_thinning_missing`
    and the one-hot `thinning_status` columns are already 0/1 and are not scaled.
  - `scaler_height`: fit on `Top_Height99` only (training split only).
  - Age is kept in its own scaler specifically so its training-split standard deviation
    (`scaler_age.scale_[0]`) can be read directly for the chain-rule correction below — if Age
    were folded into a joint scaler with the other features, that number wouldn't be cleanly
    recoverable.
  - Apply all three fitted scalers to val/test — never refit on val/test data.
- **Recap — split and trajectory pairs** (full reasoning in sections 1 and 3): both models train
  on the same `temporal_split()` row-level data; the PINN additionally uses a separate, precomputed
  trajectory-pairs table restricted to training-years-only pairs (never touching the validation
  year), fed through its own `DataLoader` alongside the main one.
- **Physics loss chain-rule correction** (for the section 2 CR-derivative loss): since both Age and Top_Height99 are standardised, the
  raw derivative the network computes is in scaled units. Convert it back to real units (metres
  per year) before comparing to the CR curve's derivative:
  ```
  d(unscaled_height)/d(unscaled_age) = d(scaled_height)/d(scaled_age) * (sigma_height / sigma_age)
  ```
  where `sigma_height = scaler_height.scale_[0]` and `sigma_age = scaler_age.scale_[0]`, both from
  the training-split scalers above.

## 5. Architecture (identical for DNN and PINN — isolates the physics term as the only difference)

- 3 hidden layers, 128 neurons each, leaky ReLU activation.
- Adam optimiser, learning rate `0.0001`, with `ReduceLROnPlateau` (factor `0.8`) watching
  validation loss.
- L1 regularisation on the network weights, coefficient `λ_1 = 1e-5` (Lynch, 2025).
- Batch size: DNN = 128, PINN = 32 (see section 1).

## 6. Model 1 — plain DNN (`dnn_noenv`)

- Loss: plain MSE against `Top_Height99` (scaled).
- Trains on ordinary per-row data only — no trajectory pairs, no second loss term. The DNN has no
  physics loss of any kind, so it needs nothing from section 3.
- Early stopping on validation loss.
- Save: best checkpoint, final checkpoint, scaler(s), per-epoch training history
  (`train_loss, val_loss, learning_rate`).

## 7. Model 2 — PINN (`pinn_noenv`) — CR-PINN Version 1

- **Network input structure**: Age must be passed into the network as its own tensor, separate
  from the other features, with `requires_grad=True` set on the Age tensor only — this is what
  lets `torch.autograd.grad` differentiate the network's output with respect to Age alone, holding
  `CanopyCover, Thin, time_since_thinning, thinning_status, yldc` fixed as ordinary (non-
  differentiated) context for that row. Code pattern (illustrative, keep this as a small,
  clearly-commented helper function per the code-style rules in section 11):

  ```python
  # Age is kept separate so autograd can differentiate through it alone.
  # The other features are ordinary inputs -- no gradient tracking needed for them.
  age_tensor = scaled_age.clone().requires_grad_(True)
  other_features_tensor = torch.cat([canopy_cover, thin, time_since_thinning,
                                      time_since_thinning_missing, recent_thinning_5yr,
                                      thinning_status_onehot, yldc], dim=1)

  predicted_height = model(other_features_tensor, age_tensor)

  # Differentiate the network's output with respect to Age only.
  height_wrt_age_scaled = torch.autograd.grad(
      outputs=predicted_height,
      inputs=age_tensor,
      grad_outputs=torch.ones_like(predicted_height),
      create_graph=True,
  )[0]

  # Undo scaling so the derivative is in real units (metres per year) before
  # comparing it to the CR curve's own derivative.
  height_wrt_age = height_wrt_age_scaled * (scaler_height.scale_[0] / scaler_age.scale_[0])
  ```

- **CR-derivative physics loss** (`λ_ph`, original term): compare `height_wrt_age` above against
  the CR curve's own analytical derivative (`d(CR height)/d(Age)`, computed from this cohort's
  frozen `y_max/k/p` at the same, unscaled ages) — never the raw predicted heights, and never
  `y_max` itself. See section 2 for what this constraint physically means (growth *rate*, not
  curve shape). This term uses **ordinary per-row data** — one row, one age, no pairing needed.

- **Trajectory physics loss** (`λ_traj`, new, second and separate term): uses the trajectory
  pairs table from section 3, **not** the ordinary per-row data. For each pair (a plot's earlier
  and later training-year observations):

  1. Run the network's forward pass **twice** — once on the pair's earlier-year feature row, once
     on its later-year feature row — producing `predicted_height_earlier` and
     `predicted_height_later`. Both passes use the *network's own predictions*, not the observed
     heights.
  2. Compute the network's implied growth rate between the two predictions:
     `predicted_growth_rate = (predicted_height_later - predicted_height_earlier) / delta_age`,
     where `delta_age = age_later - age_earlier` for that plot (unscaled years).
  3. Compute the CR curve's analytical derivative at `age_mid = (age_earlier + age_later) / 2`,
     using this cohort's frozen `y_max/k/p` — the same CR-derivative function used for the
     per-row physics loss above, just evaluated once per pair instead of once per row.
  4. `trajectory_loss = MSE(predicted_growth_rate, cr_derivative(age_mid))` — averaged over all
     pairs in the batch.

  **This is a genuinely different check from the CR-derivative loss above**: that term asks "does
  the network's *instantaneous* slope at this one row's age match CR's slope there?" (computed via
  autograd, no second row needed). The trajectory loss instead asks "does the network's own
  *predicted change* between two specific real observations of the same plant match what CR would
  predict for that specific age gap?" — a finite-difference check across a real, observed pair,
  not an instantaneous derivative at one point. `OBSERVED_GROWTH_RATE` (`annual_height99_increment`
  from section 3) is used only to build/filter which pairs qualify for this term (both years must
  be training years) — it is **not** itself compared against anything and is **never** a model
  input feature, only a data-availability signal.

- **Total loss**:
  ```
  L_total = data_loss + (λ_ph * physics_loss) + (λ_traj * trajectory_loss) + (λ_1 * l1_loss)
          = data_loss + (1.0 * physics_loss)  + (1.0 * trajectory_loss)     + (1e-5 * l1_loss)
  ```
  `λ_ph = 1.0` and `λ_1 = 1e-5` are Lynch's confirmed values (section 1, unchanged). `λ_traj = 1.0`
  is a fixed starting value for this build, matching `λ_ph`'s scale as a reasonable default — no
  sweep yet, that's a later ablation (dissertation plan section 4.7) alongside `λ_ph`'s own sweep.
- Early stopping on validation loss (the combined `L_total`, evaluated on val rows — trajectory
  pairs never touch the validation year, per section 1, so the trajectory loss's contribution to
  the *validation* metric is always exactly zero; only `data_loss` and `physics_loss` are actually
  evaluated on val). Lynch (2025) reportedly didn't have a genuine held-out validation set for
  this; this build does — flag clearly in the results if early stopping behaves oddly (e.g. never
  triggers, or triggers immediately), since that would be a new finding worth noting, not an
  expected outcome.
- Save: everything the DNN saves, plus `data_loss`, `physics_loss`, and `trajectory_loss` logged
  **separately** per epoch (not just the combined total) — this is what lets the eventual
  write-up show whether either physics term is doing anything, and whether the CR-derivative
  term's small effect (section 2) replicates here too.

## 8. Diagnostic to print before training starts

Before the training loop runs (for both models, both cohorts), print — this is a pure sanity
check, no training happens yet:

- Number of unique plots (`identification`) in the training split.
- Number of unique plots in the test split.
- Number of training plots with at least one usable trajectory pair (i.e. an
  `annual_height99_increment` value from section 3's pairs table).
- Mean and standard deviation of `annual_height99_increment` across those training pairs.
- Row count breakdown by year within the training split (how many rows from 2008, from 2012, and
  — for 6survey — from 2002 and 2006 too).
- A warning if fewer than 30% of training plots have a usable trajectory pair — per section 3,
  this is expected to pass easily given known full year coverage, but keep the check as a real
  guard against a silent bug in the filtering/pairing logic, not just a formality.

## 9. Why this approach, not pooling years cross-sectionally

Worth stating explicitly, since it's easy to lose sight of once the mechanics are in place: the
alternative to all of this — pooling all survey years into one undifferentiated set of rows and
splitting randomly (`plot_level_split`, what every baseline so far has done) — never lets the
model see an individual plant's *trajectory*. It only ever sees a cross-section: at some given age,
here is a height. Two different rows at age 40 could be two entirely different plots that happen
to share an age, or the same plot seen at two different calendar times — the model has no way to
tell, and no loss term ever asks it to.

`temporal_split` plus the trajectory pairs changes that. The same plot's 2008 and 2012 observations
are now linked, explicitly, as *one plant's own growth* — and the new trajectory loss asks a
question the CR-derivative loss alone cannot: not just "is your instantaneous slope plausible at
this age," but "does the height change you actually predicted for this specific plant, over this
specific real gap, match what CR would say." That's a strictly stronger, more specific check,
because it's anchored to one plant's own two real measurements rather than compared only against a
population-level curve.

## 10. Outputs

Per model, per cohort, under `outputs/<model_name>/<cohort>/` (`dnn_noenv` or `pinn_noenv`):

```
metrics.json              # via models/common/metrics.py — same format as every other baseline
predictions.csv            # identification, blk, cpmt, LiDAR_year, Age, observed_top_height,
                            # predicted_top_height, residual, split
training_history.csv        # epoch, train_loss, val_loss,
                            # (PINN only: data_loss, physics_loss, trajectory_loss), learning_rate
run_metadata.json           # via models/common/saving.py::save_run — includes git commit hash,
                            # and should also record physics_weight (1.0), trajectory_weight (1.0),
                            # L1 coefficient (1e-5), batch size, the temporal_split year assignment
                            # used, and which scalers were used, so the fixed values used in this
                            # build are easy to find once the ablation stage sweeps them
checkpoints/best_model.pt, final_model.pt
preprocessing/             # scaler_age.joblib, scaler_other_features.joblib, scaler_height.joblib,
                            # plus the saved encoded_column_names list from the thinning_status
                            # one-hot encoding (needed to reproduce predictions)
```

`predictions.csv` includes `cpmt` and `LiDAR_year` (not listed in the original brief) to match
what `evaluate_baselines.py` already writes for the four existing baselines — needed for the
existing growth-curve/bias-by-year and spatial-generalization notebook sections to work on these
two new models without modification.

## 11. Code style — non-negotiable for this codebase

This project's modelling code is deliberately written for a beginner to read and extend, not
idiomatic/compressed Python (established convention across `chapman_richards.py`,
`linear_baseline.py`, `rf_baseline.py`, `average_by_age.py`). Apply the same standard here,
including inside the PyTorch model/training-loop code, which is more tempting to write tersely
than the sklearn-based baselines were:

- Prefer explicit `for` loops and `if`/`else` over nested `torch.where`, comprehensions with
  multiple conditions, or chained one-liners — even in the training loop.
- Descriptive variable names spelled out in full (`predicted_heights`, `validation_loss_history`),
  not single letters or abbreviations (no bare `y`, `yhat`, `dH`).
- A short comment explaining *what a block does* is welcome and expected here, more so than the
  general project convention — this is explicitly written for someone still learning PyTorch, not
  just forestry modelling.
- Split the training loop into small, named, testable functions (`train_one_epoch`,
  `evaluate_on_validation_set`, `compute_physics_loss`) rather than one long `main()` — every step
  should be individually readable without holding the whole loop in your head at once.
- No premature abstraction: don't build a generic "trainer class" shared by both models unless it
  stays simple — two similar, separately-readable training scripts are better than one clever
  shared one if the shared one gets hard to follow.

## 12. Explicit "don't"s

- No terrain/wind features — no-environment feature set only, for both models.
- No physics-weight, trajectory-weight, or batch-size sweep — fixed values only, ablations come
  later.
- Never refit `y_max`/`k`/`p` inside the PINN — read them from CR's saved `params.json`, frozen.
- Never touch the test split (2023) until final evaluation (same rule as every other baseline).
- Never build a trajectory pair that touches the validation year (2021) — see section 1's leakage
  rule.
- Never use `annual_height99_increment` / `OBSERVED_GROWTH_RATE` as a model input feature — pairs
  data is for the trajectory loss target only, never a predictor.

## 13. Finish criteria

- One comparison table: DNN and PINN, both cohorts, same metrics/age-band format as CR/average-by-age
  (so all six models — CR, average-by-age, linear, RF, DNN, PINN — line up in one table, all six
  now sharing the same `temporal_split` methodology per
  `baseline_models_temporal_split_instructions.md`).
- Train vs. validation loss curves for both models, both cohorts — confirm early stopping triggers
  sensibly (neither "still improving when stopped" nor badly overfit).
- For the PINN: report final `data_loss` vs. `physics_loss` vs. `trajectory_loss` per cohort, so
  it's visible whether either physics term is actually doing anything or has been drowned out /
  is dominating.
- The section 8 diagnostic output, run against the real data, shown before any training happens.

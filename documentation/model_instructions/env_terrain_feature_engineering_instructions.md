# Instructions: Feature-engineered `dnn_env_terrain`/`pinn_env_terrain` (new folders)

**Status: DRAFT — not yet approved, do not execute.** Written to be reviewed and confirmed before
any code is written. New folders (`models/dnn_env_terrain_engineered/`,
`models/pinn_env_terrain_engineered/`), never an edit to the existing `dnn_env_terrain`/
`pinn_env_terrain` folders, which stay as the reported joint-training baseline this is measured
against.

## 0. Why this exists (one paragraph, so this doc stands alone)

2026-08-02's scope-matched XGBoost check found a real, controlled result: given the IDENTICAL
rows/features/target/split `dnn_env_terrain` sees, XGBoost gets a real positive uplift from
terrain (+0.0111 test R2) while the MLP gets a real negative one (-0.0083). Dropout, learning
rate, and network-size sweeps were all null the same session — ruling out regularization,
optimizer step size, and raw capacity as the explanation. What's left is an inductive-bias gap:
trees split on thresholds/interactions natively; a dense linear+LeakyReLU network has to discover
that same structure by composing hidden units via gradient descent, which is known to be harder
when the signal is small relative to a dominant feature (Age already explains ~63% of variance
here). This experiment directly tests that explanation by handing the network pre-built
threshold/interaction features instead of asking it to discover them from raw scaled scalars.

Full reasoning and numbers: `documentation/experiment_log.md`'s 2026-08-02 entries (dropout/LR
sweep, architecture-size sweep, feature-set-parity check, scope-matched XGBoost check).

## 1. What already exists (reuse, don't rebuild)

- **`NoEnvNetwork`/`YMaxSubNetwork`** (`models/common/torch_model.py`) — completely unchanged.
  This experiment is a pure input-representation change; the network classes, training loop,
  loss function, and hyperparameters stay identical to `dnn_env_terrain`/`pinn_env_terrain`, so
  any result difference is attributable to the input features, not a second confounded change.
- **`load_split_table_with_terrain(cohort, split_type, feature_columns)`**
  (`models/common/torch_data.py`) — reuse as-is. It already takes a raw column list (not a name
  looked up by string), already handles the `whcl` merge-collision fix and the `ceh_twi`
  missingness drop (both fixed 2026-08-02), and already respects the `spatial_block` buffer
  exclusion. The engineered columns just need to exist in the dataframe it returns before
  `feature_columns` is passed in — see section 2.
- **`ENV_TERRAIN_FEATURE_SETS`/`DEFAULT_ENV_TERRAIN_FEATURE_SET`** (`models/common/torch_data.py`)
  — extend this dict with a new named entry (e.g. `"terrain_wind_engineered"`), same pattern as
  `terrain_wind_solid`/`terrain_wind_extended` already use. `--feature-set` on both `run_*.py`
  scripts already reads its choices from this dict's keys, so a new entry is immediately usable
  with zero CLI changes.
- **`run_dnn_env_terrain.py`/`run_pinn_env_terrain.py`/`evaluate_*.py`** — copy these three files
  into the new folders and adjust only the import paths and `MODEL_NAME` constant, same "copy the
  pattern, don't redesign it" convention the original `dnn_env_terrain`/`pinn_env_terrain` build
  followed from `dnn_noenv`/`pinn_noenv`.
- **`--hidden-layer-sizes`/`--dropout-rate`/`--learning-rate`** CLI flags — already wired into all
  four existing models (2026-08-02). Keep every one at its default for this experiment (same
  reasoning as the network classes point above) — don't reopen the capacity/regularization
  question this session already nulled.

## 2. What's genuinely new here

### 2a. Which base columns to engineer from

Use the per-variable refit ablation (`grouped_category_importance.ipynb` Section 7.2,
leak-free per the 2026-07-31 fix) to pick columns, not intuition:

| Column | refit r2_drop | Currently in `terrain_wind_solid`? |
|---|---|---|
| `ceh_twi` | 0.181 (largest) | yes |
| `eastness` | 0.093 | yes |
| `elevation` | 0.091 | yes |
| `northness` | 0.090 | yes |
| `plan_curvature` | 0.065 | no (`terrain_wind_extended` only) |
| `topex` | 0.061 | yes |
| `whcl` | 0.023 (small but positive) | no (`terrain_wind_extended` only) |

Start with `ceh_twi` and `elevation` (the two largest) for binning, and one interaction term —
don't engineer all 7 at once. The point of this experiment is to attribute any R2 change to a
specific piece, not to produce one more black-box feature set.

### 2b. Binning

Add `ceh_twi_bin`/`elevation_bin` as one-hot bucket columns (e.g. 5 quantile bins each via
`pd.qcut`, `q=5`). **Bin edges must be computed on the TRAIN split only**, then applied to
val/test via `pd.cut` with those same edges — identical discipline to `fit_scalers(train_df)`
already using train-only statistics everywhere else in this pipeline. A bucket with zero training
rows is possible for val/test under `spatial_block_split`; handle it the same way
`encode_thinning_status(df, encoded_column_names=...)` already handles an unseen category
(reindex onto the training columns, fill with 0), not a new pattern.

### 2c. Interaction terms

One theoretically-motivated pair to start: `elevation * topex` (exposure conditional on
altitude — a real forestry mechanism, not an arbitrary cross). Compute as a plain elementwise
product of the two RAW (not yet scaled) columns, then let `fit_terrain_scaler()` standard-scale
the resulting column same as every other terrain feature — don't hand-scale it separately.

**Open design question, confirm before implementing, don't decide silently:** `YMaxSubNetwork`
currently only ever sees terrain columns, never `Age` (`pinn_env_terrain.py`'s own docstring: "y_max
sub-network... only ever called... with the y_max sub-network's own... terrain features"). An
`Age * elevation`-style interaction is therefore straightforward for the flat `dnn_env_terrain`
network (which already concatenates everything into one input) but is NOT available to
`pinn_env_terrain`'s y_max sub-network without changing its signature to accept Age — which starts
to blur into the UDE-style option this experiment is explicitly meant to stay separate from. Two
honest choices: (i) keep every engineered feature terrain-only (terrain x terrain products, no
Age), so both models stay directly comparable and the y_max sub-network's interface is untouched;
or (ii) allow `dnn_env_terrain_engineered` to test Age-interactions since it has no architectural
constraint against it, while `pinn_env_terrain_engineered` only gets the terrain-only subset —
accepting the two models then aren't tested on quite the same feature set. **Recommend (i)** for a
clean comparison, but this is a real fork worth confirming, not something to pick unilaterally.

### 2d. New helper function

Add `build_engineered_terrain_columns(df, base_columns)` to `models/common/torch_data.py`
(alongside `ENV_TERRAIN_FEATURE_SETS`) that takes the row-level dataframe
`load_split_table_with_terrain()` already returns and adds the bin/interaction columns onto it in
place, returning the new column names to feed as `feature_columns`. Keep it a plain function
operating on a dataframe — same "small named functions, no custom classes" style the rest of this
file already uses.

## 3. Sweep plan — incremental, not all-at-once

Run on `dnn_env_terrain_engineered` first (cheap, no physics loss, isolates whether it's purely an
input-representation fix), `spatial_block`/4survey, comparing test R2 against the two anchors
already established: `dnn_noenv`=0.6330 (no terrain at all) and `dnn_env_terrain`=0.6247 (raw
terrain, current regression).

| Variant | What's added |
|---|---|
| control | `terrain_wind_solid` unchanged (0.6247, already known) |
| +bin_ceh_twi | raw 5 columns + `ceh_twi_bin` one-hot only |
| +bin_elevation | raw 5 columns + `elevation_bin` one-hot only |
| +interaction | raw 5 columns + `elevation * topex` only |
| +all three | raw 5 columns + both bins + the interaction |

Only carry a variant forward to `pinn_env_terrain_engineered` (both `pw=0` and `pw=1`, matching
the existing sweep pattern) if it beats `dnn_env_terrain`'s 0.6247 by more than the ~0.002 noise
band already established (Stage 1/Stage 3 batch-size and physics-weight sweeps).

## 4. Explicit "don't"s

- Don't modify `NoEnvNetwork`, `YMaxSubNetwork`, or any training-loop code — this is an
  input-feature experiment only.
- Don't change `LEARNING_RATE`/`WEIGHT_DECAY`/`L1_COEFFICIENT`/`BATCH_SIZE`/dropout from their
  current defaults — reopening those was already nulled this session; changing them here would
  confound the result.
- Don't add climate or soil columns — stays within the terrain/wind scope
  `pinn_env_terrain.py`'s own docstring commits to ("terrain/wind conditions ONLY y_max").
- Don't engineer all 7 base columns' worth of bins/interactions in one shot — incremental,
  per section 3, so a result is attributable.
- Don't touch `models/dnn_env_terrain/` or `models/pinn_env_terrain/` — new folders only.

## 5. Finish criteria

- `outputs/spatial_block/dnn_env_terrain_engineered/4survey/metrics.json` for each sweep variant
  in section 3, compared against 0.6330/0.6247.
- If any variant beats `dnn_env_terrain` by more than noise: matching
  `pinn_env_terrain_engineered` runs (`pw=0`/`pw=1`, terrain-only interactions per section 2c's
  recommended choice) on the same split/cohort.
- One dated entry in `documentation/experiment_log.md`'s Findings log, same four-part shape as
  every other entry (What I found / What's working / What's not working / What this means for
  what's next).

# Instructions: Decoupled deviation model, `models/env_deviation/` (new folder)

**Status: DRAFT — not yet approved, do not execute.** Written to be reviewed and confirmed before
any code is written. New folder, never an edit to `dnn_env_terrain`/`pinn_env_terrain` (which stay
as the reported joint-training baseline this is measured against) or to `xgb_environmental`/
`grouped_category_importance.ipynb` (which stay as the existing attribution-only pipeline this
extends, not replaces).

## 0. Why this exists, and how it differs from what's already built (read this first)

The core idea: instead of forcing one network to jointly (a) fit height consistent with the CR
curve and (b) respond to terrain, in one shared loss where the two objectives compete for
gradient (the exact trade-off the 2026-08-02 physics-weight sweep measured: `pw<=0.1` -> zero
gradient to `y_max`, `pw=1.0` -> real responsiveness but test R2 drops 0.633->0.620) -- decouple
them. Fit the base growth model however it's already fit, then treat "does terrain predict the
LEFTOVER deviation" as its own separate supervised-learning problem.

**This is not a new idea in this repo -- `xgb_environmental.py`/`grouped_category_importance.ipynb`
already predict `mean_cr_residual` from terrain, decoupled from the PINN's joint training, and
already show real signal (terrain+wind alone: R2=0.162-0.188).** What's actually new here is
closing three specific, concrete gaps in that existing pipeline, not a new modeling idea:

| | Existing (`mean_cr_residual`) | This model |
|---|---|---|
| Target granularity | one value per PLOT, averaged across all survey years (collapses age) | one value per plot-SURVEY-YEAR (row-level, matches `dnn_env_terrain`'s own granularity) |
| CR anchor | `outputs/chapman_richards/<cohort>/params.json` -- the POOLED anchor, confirmed 2026-08-01 to leak (its random 60% training plots inevitably overlap `spatial_block`'s test plots) | the split-matched anchor (`outputs/<split_type>/chapman_richards/<cohort>/params.json`) `pinn_env_terrain` already correctly uses |
| Feature scope | full 37-variable set, or 16-column `terrain_and_wind_only` | `ENV_TERRAIN_FEATURE_SETS["terrain_wind_solid"]` (5 cols) -- same scope `dnn_env_terrain`/`pinn_env_terrain` actually use, for direct comparability |
| Residual model's own inputs | full feature set (age not included, but stand-structure/climate/soil are) | terrain-only -- genuinely decoupled, not "what's left after removing one circular column" |
| Output used for | attribution ranking only (SHAP, permutation importance) -- never composed back into a height prediction | composed back onto the base prediction and scored with `compute_metrics()` on the SAME test split, giving a number directly comparable to `dnn_env_terrain`=0.6247 / `pinn_env_terrain(pw=1)`=0.6195 / `dnn_noenv`=0.6330 |

That comparable number is the one thing that's genuinely missing right now -- nobody has checked
whether decoupling actually recovers what joint training loses, only that a decoupled residual
model finds *some* signal in isolation.

## 1. What already exists (reuse, don't rebuild)

- **`load_split_table_with_terrain(cohort, split_type, feature_columns)`**
  (`models/common/torch_data.py`) -- reuse as-is for row-level data with terrain merged in and
  the `split` column (train/val/test/buffer) already assigned.
- **`chapman_richards(age, y_max, k, p)`** (`models/chapman_richards/chapman_richards.py`) --
  reuse to compute the base CR-predicted height per row.
- **`load_cr_params(cohort, split_type)`** -- currently duplicated verbatim in
  `run_pinn_noenv.py` and `run_pinn_env_terrain.py` (both read
  `outputs/<split_type>/chapman_richards/<cohort>/params.json`). This model is a third caller --
  worth moving it to a shared location (e.g. `models/common/saving.py`) as a small bundled
  cleanup rather than a third copy-paste, flagged per section 4.
- **`fit_with_columns(train_df, feature_columns, val_df=None, target_col=..., seed=42,
  **xgb_params)`/`predict_with_columns(df, model, feature_columns)`**
  (`models/xgb_environmental/xgb_environmental.py`) -- reuse directly for the residual-predicting
  model. `target_col` is already a parameter, not hardcoded to `mean_cr_residual`.
- **`HYPERPARAMETER_GRID`/`tune_hyperparameters()`** (`models/xgb_environmental/xgb_environmental.py`)
  -- reuse this project's own existing max_depth/reg_lambda grid rather than assuming the fixed
  `max_depth=6, n_estimators=500, lr=0.1` config the 2026-08-02 scope-matched check used
  un-tuned.
- **`dnn_noenv.load_best_model()`/`predict()`** (`models/dnn_noenv/dnn_noenv.py`) -- reuse to get
  predictions for the second base-model variant (section 2b).
- **`compute_metrics()`** (`models/common/metrics.py`) -- reuse for the final composed-prediction
  evaluation, so the output is directly comparable to every other model's `metrics.json`.
- **`n_jobs=1`** on `XGBRegressor` -- required on this Mac (torch+xgboost in the same process
  segfaults otherwise, confirmed 2026-08-02 -- `KMP_DUPLICATE_LIB_OK=TRUE` alone isn't enough).

## 2. What's genuinely new here

### 2a. Row-level, split-matched residual target

```python
cr_params = load_cr_params(cohort, split_type)  # split-matched, not pooled
predicted_cr_height = chapman_richards(split_df["Age"], cr_params["y_max"], cr_params["k"], cr_params["p"])
split_df["cr_residual"] = split_df[TARGET_COLUMN] - predicted_cr_height
```

One row per plot-survey-year, not averaged per plot -- age-varying deviation is preserved, unlike
`mean_cr_residual`.

### 2b. A second base-model variant: `dnn_noenv`-residual

`cr_residual` asks "what does terrain explain that the PHYSICS anchor misses." A second variant,
`dnnnoenv_residual = split_df[TARGET_COLUMN] - dnn_noenv_prediction_unscaled`, asks "what does
terrain explain that the BEST AVAILABLE STATISTICAL model misses" -- a different, both
scientifically interesting question, not redundant with the CR version. Requires loading
`dnn_noenv`'s saved `scaler_height`/`scaler_age`/`scaler_other_features`/`encoded_column_names`
from its `preprocessing/` directory (same joblib files `evaluate_dnn_env_terrain.py` already
loads) to unscale its prediction back to raw metres before subtracting.

Build both variants -- don't pick one without checking, same reasoning as the physics-weight
sweep's "don't report a single seed/config as settled" precedent.

### 2c. Terrain-only inputs -- the actual decoupling

Feature columns for the residual model = `ENV_TERRAIN_FEATURE_SETS["terrain_wind_solid"]` (or
`"terrain_wind_extended"`, swept per section 3) and **nothing else** -- no Age, no
`CanopyCover`/`Thin`/thinning-status. This is what makes it decoupled rather than a repeat of
2026-08-02's scope-matched check, which concatenated age+no-env+terrain into one joint XGBoost
feature space. Here the base model has already used age/stand-structure to produce its
prediction; the residual model's only job is what terrain explains on top of that.

### 2d. Composition and evaluation

```python
predicted_height = base_prediction + predicted_residual  # base_prediction: CR or dnn_noenv, matching section 2a/2b
metrics = compute_metrics(test_df[TARGET_COLUMN].values, predicted_height, age=test_df["Age"].values)
```

Save as `outputs/<split_type>/env_deviation_<base>/<cohort>/metrics.json`, same shape every other
model's metrics.json already has, so it drops into any existing results notebook/comparison table
with no special-casing.

## 3. Sweep plan

| Axis | Values | Why |
|---|---|---|
| Base model | `cr` / `dnn_noenv` | different question each answers, section 2b |
| Terrain feature set | `terrain_wind_solid` / `terrain_wind_extended` | already defined, already tested null on the JOINT models 2026-08-02 -- worth re-checking whether the same null holds decoupled |
| XGBoost hyperparameters | `HYPERPARAMETER_GRID`, picked by val R2 on the residual target | reuse existing tuning code (section 1), don't assume the untuned config from the earlier scope-matched check is right here too |

4 base/feature-set combinations x the existing hyperparameter grid, `spatial_block`/4survey
first (primary split). `n_jobs=1` throughout (section 1).

## 4. Small fix worth bundling in

`load_cr_params(cohort, split_type)` becomes a THIRD verbatim copy once this model exists
(currently in `run_pinn_noenv.py` and `run_pinn_env_terrain.py`). Move it to
`models/common/saving.py` (alongside `model_output_dir()`, which every one of these three callers
already imports from there) and have all three import it, rather than adding a third copy that
can silently drift from the other two. Small, mechanical, no behaviour change.

## 5. Explicit "don't"s

- Don't reuse or modify `mean_cr_residual`/`aux_data_resolution_check.ipynb`'s existing
  computation in place -- it's plot-level/pooled and belongs to the attribution notebook's own
  purpose. Its still-pooled (leaky) CR anchor is a separate, real bug worth its own
  `experiment_log.md` line, independent of this model.
- Don't include Age or any no-env feature in the residual model's own inputs -- that breaks the
  decoupling this model exists to test.
- Don't touch `dnn_env_terrain`/`pinn_env_terrain`/`xgb_environmental` -- new folder only.
- Don't skip the `dnn_noenv`-residual variant just because the CR-residual variant looks good (or
  bad) first -- both answer genuinely different questions, per section 2b.

## 6. Finish criteria

- `outputs/spatial_block/env_deviation_cr/4survey/metrics.json` and
  `outputs/spatial_block/env_deviation_dnnnoenv/4survey/metrics.json` (each for both feature-set
  variants -- 4 files total), with test R2 directly comparable to `dnn_noenv`=0.6330,
  `dnn_env_terrain`=0.6247, `pinn_env_terrain(pw=1)`=0.6195.
- `load_cr_params()` de-duplicated into `models/common/saving.py` (section 4).
- One dated entry in `documentation/experiment_log.md`'s Findings log, same four-part shape as
  every other entry.

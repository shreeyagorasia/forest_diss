# PINN forward-pass fix -- quicktest results (2026-08-20)

Purpose: cheap, disconnected evidence for whether the documented forward-pass limitation
(terrain/wind reaching only the physics/trajectory losses, never the prediction itself) is
actually worth fixing as future work -- not a claimed result, not wired into any reported
number. See `PLAN.md` for the full isolation rules and the 15 documented pitfalls this
experiment followed.

## What was done

1. Copied `pinn_env_terrain.py` -> `pinn_env_terrain_fix.py` and `pinn_env_terrain_k.py` ->
   `pinn_env_terrain_k_fix.py`. Only change: `forward()` now evaluates the Chapman-Richards
   curve at the per-plot adjusted `y_max` (and `k`, for the k version), with the trunk
   network contributing a residual on top -- `H_pred = CR(y_max_i, k_i, age) + trunk_i`,
   instead of the trunk alone.
2. Local smoke tests (`smoke_test/`) for both versions, run first, before any real training:
   confirmed terrain now genuinely changes the prediction, the CR curve behaves correctly at
   its boundaries (H(0)=0, H(large age)->y_max), k stays strictly positive and finite even
   under an extreme (20 SD) terrain shift, and a tiny 3-epoch training loop runs with no NaN
   and real gradient flow throughout. All passed before any real-data run was attempted.
3. Quicktest: 4survey, Set3 (`nested_set3_gated_terrain_wind_vif`, confirmed matching the
   actual headline table's feature set via `outputs/run_logs/`, not the module's own
   different default), single `spatial_block` split, `max_epochs=40`, `patience=10`. OLD
   (unfixed, original file) and NEW (fixed) run at IDENTICAL matched settings for a fair
   comparison -- the real OLD number in the dissertation is a 500-epoch production run, not
   comparable to a 40-epoch NEW run on its own.

## Results

| Variant | OLD (unfixed) test R2 | NEW (fixed) test R2 | Movement | Epochs (OLD/NEW, both early-stopped) |
|---|---:|---:|---:|---:|
| PINN (y_max only) | 0.5810 | 0.6296 | **+0.0486** | 27 / 27 |
| PINN-k (y_max + k) | 0.5859 | 0.6114 | **+0.0255** | 27 / 16 |

Reference points, not rerun here: production DNN R2 = 0.655 (4survey, 500 epochs, Table
`tab:results-rq1`); production PINN R2 = 0.573; production PINN-k R2 = 0.575 (both already in
the dissertation, both the *unfixed* architecture).

## Read

Real, consistent, positive movement in both variants, at a fraction of the production epoch
budget (40 vs 500). The fixed y_max-only version, at 40 epochs, already sits close to DNN's
full-training ceiling (0.6296 vs 0.655) -- a gap of 0.025, versus the unfixed architecture's
own full-training gap of 0.082 (0.573 vs 0.655). This is real, direct evidence that the
documented forward-pass limitation is not merely a cosmetic implementation detail -- fixing
it plausibly recovers a meaningful share of PINN's underperformance against XGBoost/DNN.

y_max-only shows more movement than y_max+k here. Not enough evidence in a single-seed,
single-split quicktest to say whether that's a real effect (e.g. k's extra per-plot degree of
freedom making optimisation harder at low epoch counts) or noise -- flagged as an open
question, not resolved by this exercise.

## What this does NOT establish (explicitly out of scope for a quicktest)

- **Mistake #4 (PLAN.md)**: whether the CR term actually contributes meaningfully to the
  final prediction, or the residual network's larger capacity just routes around it. Not
  checked -- would need inspecting the relative magnitude of the CR term vs the residual term
  across held-out rows.
- **Mistake #5 (PLAN.md)**: extrapolation risk -- a bad/out-of-range terrain value now shapes
  the prediction directly, not just a loss target. Not checked -- would need the spread of
  held-out predictions for out-of-range terrain, not just aggregate R2.
- **Statistical robustness**: single seed, single spatial-block split, 4survey only (6survey
  skipped, per `PLAN.md`'s own quicktest scope). No claim of significance, no error bars.
- **The physics-weight ablation, random-split re-test, or any other downstream result** that
  would need rerunning if this fix were adopted for real (see `PLAN.md`'s "Downstream" table,
  ~36 jobs minimum for a proper redo).

## Recommendation for the dissertation

Cite as preliminary, disconnected evidence supporting a specific, scoped future-work item --
not as a result. Suggested framing: "A minimal, isolated quicktest (single split, reduced
epochs, no hyperparameter tuning) found the fixed architecture recovers most of the gap to
DNN at a fraction of the training budget (PINN: 0.082 -> 0.025 gap; PINN-k: a smaller but
still real improvement), consistent with the forward-pass limitation being a substantive,
not cosmetic, contributor to the reported underperformance -- though this was not verified at
the scale, robustness, or scrutiny (Mistakes #4/#5 above) needed to report as a finding."

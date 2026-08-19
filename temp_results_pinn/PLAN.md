# PINN embedding fix — plan

## Problem, one line
`forward()` in `pinn_env_terrain.py` / `pinn_env_terrain_k.py` never uses
`y_max_subnetwork`/`k_subnetwork` output. Terrain shapes the loss target only,
never the prediction. True at every `physics_weight`, including 1.

## Fix, one line
Prediction becomes CR curve (using per-plot `y_max^(i)`/`k^(i)`) plus a trunk
residual, not the trunk alone:

```
H_pred_i = y_max_i * (1 - exp(-k_i * age_i))**p + trunk_residual_i
```

Trunk keeps its flexibility (fixes a possibly-wrong pooled curve locally).
`y_max^(i)`/`k^(i)` now genuinely reach the prediction. Physics/trajectory
loss unchanged.

**Not** doing: concatenating terrain into the trunk's raw input. That makes
PINN identical to DNN and throws away the whole point of the sub-networks.

---

## Isolation rules (do not break these)

- No file under `models/` edited. Ever.
- No file under `outputs/` written. Ever.
- Everything new lives under `temp_results_pinn/`.
- Every run uses `--run-name` so output paths never collide with real ones.
- Old and new checkpoints never share a directory (see Mistake #1 below).

## New files

```
temp_results_pinn/
  PLAN.md                                  <- this file
  pinn_env_terrain_fix/
    pinn_env_terrain_fix.py                <- copy + forward() fix
    run_pinn_env_terrain_fix.py            <- copy, import updated
    evaluate_pinn_env_terrain_fix.py       <- copy, import updated
  pinn_env_terrain_k_fix/
    pinn_env_terrain_k_fix.py
    run_pinn_env_terrain_k_fix.py
    evaluate_pinn_env_terrain_k_fix.py
  jobs/
    run_pinn_env_terrain_fix.sh            <- sbatch, new log dir
    run_pinn_env_terrain_k_fix.sh
  smoke_test/
    smoke_test_forward.py                  <- local, no cluster, no training
  outputs/
    quicktest/                             <- fast local/cluster diagnostic
    full_rerun/                            <- only if quicktest works
logs/temp_results_pinn/                    <- new log dir (outside the folder,
                                               matches existing logs/ convention)
```

---

## Step order

1. Copy `pinn_env_terrain.py` -> `pinn_env_terrain_fix.py`. Edit only
   `forward()` (and `predict()`, which just calls `forward()`).
2. Same for `pinn_env_terrain_k.py`.
3. Fix unit scaling: trunk output is *scaled* height, CR curve is *real*
   metres. Unscale trunk output before adding, or scale the CR term. Print
   both terms separately for 5 rows and sanity-check by eye before trusting
   anything downstream.
4. Write `smoke_test_forward.py` (see below). Run it. Do not skip.
5. Copy `run_*.py` / `evaluate_*.py`, update imports only, add `--run-name`
   default pointing into `temp_results_pinn/outputs/`.
6. Write sbatch scripts, new log paths.
7. Local smoke test (CPU, tiny data, few steps) before touching the cluster.
8. Cluster quicktest (real data, low epochs, one split).
9. Only then decide on the full rerun.

---

## Smoke test (local, no cluster, minutes not hours)

Goal: catch code errors and the "does terrain even reach the prediction"
question before spending any compute.

```python
# temp_results_pinn/smoke_test/smoke_test_forward.py
# Two rows, identical age + other_features, DIFFERENT terrain.
# If predictions match, the fix didn't work. Stop here if so.
import torch
model = build_model(...)  # fixed model, random-init is fine
age = torch.tensor([[30.0], [30.0]])
other = torch.zeros((2, N_OTHER_FEATURES))
terrain_a = torch.zeros((2, N_TERRAIN_FEATURES))
terrain_b = terrain_a.clone()
terrain_b[1] += 2.0  # 2 SD shift, one row only
pred_a = model(other, age, terrain_a)
pred_b = model(other, age, terrain_b)
assert not torch.allclose(pred_a[0], pred_b[1]), "terrain still not reaching prediction"
print("PASS: predictions differ by", (pred_b[1] - pred_a[0]).item())
```

Second smoke test: tiny synthetic dataset (50 rows), 3 epochs, CPU, full
`fit()` loop end to end. Checks nothing crashes (shape errors, NaN loss,
optimizer step failing) before a real GPU job burns cluster time on a bug.

```python
# check after 3 epochs:
assert not torch.isnan(loss)
assert grad_norm > 0  # confirms gradient actually reaches y_max/k subnetworks
```

---

## Cluster quicktest

- 4survey only.
- One spatial_block split, not 5-fold.
- `max_epochs=40`, `patience=10`.
- Run 3 jobs at *matched* settings: old-PINN (unfixed), new-PINN (fixed), DNN.
- Compare $R^2$: does new-PINN move toward DNN, versus old-PINN?
- Also check: CR-term variance vs residual-term variance across held-out
  rows (Mistake #4 below) — not just the aggregate $R^2$.
- 6survey: skip for time, but say so explicitly in any write-up. Do not
  assume 4survey's result carries over (cohorts have already disagreed once
  this project, on DNN tuning).

---

## Mistakes to account for

1. **Old checkpoint loads into new model, no error.** Trunk/sub-network
   shapes are unchanged — only `forward()`'s combination logic changed. An
   old `.pt` file will load into the new class silently. Never point a new
   run at an old checkpoint dir. Fresh output dirs only.
2. **Batch size / hyperparameter drift.** Do not trust the module's default
   constant. After any fix run, read the run's own `run_metadata.json` and
   confirm `batch_size`, `learning_rate`, etc. match what was intended —
   same mistake already made once this project with the DNN.
3. **Wrong baseline for the quicktest comparison.** Old-PINN's real number
   is a 500-epoch production run. Comparing a 40-epoch new-PINN against it
   is not a fair test. Run a matched 40-epoch old-PINN too.
4. **Residual absorbs everything.** Wiring $y_{max}^{(i)}/k^{(i)}$ into the
   prediction doesn't guarantee they matter — the residual network has more
   capacity and training could route around the CR term entirely. Check its
   actual contribution to the final prediction, don't assume "it's wired so
   it's fine."
5. **Sharper extrapolation risk.** A bad/out-of-range terrain value now
   directly shapes the prediction, not just a soft loss target. Check the
   spread of held-out predictions, not just the mean $R^2$ — look for new
   outliers in held-out compartments with terrain outside the training
   range.
6. **Unit mismatch.** Scaled trunk output + real-unit CR term, added without
   converting, gives a number that looks plausible and is wrong. Verify by
   printing both terms separately (see step 3 above).
7. **Data pipeline drift between old and new scripts.** The fix scripts are
   copies — if `torch_data.py`/feature-set definitions change upstream
   later, the copies could silently diverge from what Table 1 actually used.
   Before the real rerun, diff `terrain_train`/`other_train` construction
   in the new run script against the original line by line, not just trust
   the copy was faithful.
8. **Fairness vs DNN.** Same trunk architecture, same `n_other_features`
   (still excludes terrain — that part of the design was correct and
   citable), same optimizer settings, same batch size, same epochs/patience
   as the DNN comparison. The only intended difference is the physics loss
   and the now-functional $y_{max}^{(i)}/k^{(i)}$ path. Any other difference
   (dropout, hidden size, seed) breaks the "fair comparison" claim — check
   `run_metadata.json` on both sides before writing any result down.
9. **NaN from the CR term, not the residual.** $(1-e^{-k^{(i)}a})^p$ with
   fractional $p$ can blow up near zero, and it's now inside the main
   prediction, not just a loss term. One bad row poisons the whole batch via
   `torch.mean`. Check for NaN after every epoch in the smoke test.
10. **Autograd graph missing half the derivative.** The physics loss needs
    $\partial \hat H/\partial a$ via autograd. If the CR term's age input
    isn't the *same* `age_batch.clone().requires_grad_(True)` tensor the
    residual uses, `torch.autograd.grad` silently returns only the
    residual's derivative — no error, just a wrong number.
11. **Reusing `.detach()` from the wrong place.** `compute_physics_loss`
    already unscales age via `.detach()` for building a target — correct
    there. Copy that into the new prediction path and gradients stop
    flowing through age for the actual output. Easy copy-paste bug.
12. **A new CR-value helper diverging from the existing CR-derivative one.**
    Need $H(a)$ now, not just $H'(a)$. Mirror `chapman_richards_derivative()`
    line-by-line for broadcasting/shape conventions, don't write it fresh —
    a shape or sign mismatch here gives wrong numbers with no error.
13. **Residual head starts loud, not quiet.** Sub-networks start near zero
    on purpose, so training "starts from the validated pooled curve." A
    fresh MLP residual head does not start near zero by default — at step 0
    the sum could be dominated by random residual noise. Zero-init the
    residual head's final layer too.
14. **Shape broadcasting between `y_max_per_row` and `age`.** `[batch, 1]`
    against `[batch]` silently broadcasts to `[batch, batch]`, not
    `[batch]` — `.mean()` over that is a plausible-looking, completely
    wrong number. Print `.shape` on every new tensor during the smoke test.
15. **Double-transforming in the eval script.** Inverse-scaling the residual
    and CR term separately, then summing, instead of scaling the combined
    prediction once — silently wrong $R^2$, still looks like a real number.

---

## Downstream — what changes if the fix works

| Item | Jobs | Notes |
|---|---|---|
| Table 1 headline, Set3 only | 20 | 5-fold x 2 cohorts x 2 models |
| Physics-weight ablation | 12 | 3 weights x 2 cohorts x 2 models |
| Random-split re-test | 4 | inflation finding may not hold anymore |
| **Minimum total** | **36** | |

Dropped for time: temporal check (~4 jobs), architecture sweep (~16),
appendix Sets 2/4 (~40).

**Sections needing rewrite if fix validated:**
- Table 1 (PINN/PINN$_k$ rows)
- Physics-weight ablation subsection
- Effect-of-evaluation-split subsection (inflation finding may flip)
- $y_{max}$/$k$ identifiability paragraph (training dynamics change)
- Every methodology/related-work edit made this session — these currently
  describe the *broken* architecture accurately; they'd need rewriting to
  describe the fixed one.

**Confirmed unaffected, no action needed:**
- RQ2b (`mean_cr_residual`, classical pooled CR fit — no neural network)
- RQ3 (`fit_y_max_per_plot`, classical per-plot least-squares — no neural
  network)
- RQ2a: reuses RQ1's predictions.csv, but current results chapter doesn't
  show a PINN row there yet — optional upside only, not a required fix.

---

## Fallback

If quicktest shows no real movement toward DNN, or breaks in a way that
isn't quickly fixable: delete `temp_results_pinn/`, keep the original
architecture and the honest limitation write-up already done in the main
chapters. Nothing outside this folder was touched, so falling back costs
nothing.

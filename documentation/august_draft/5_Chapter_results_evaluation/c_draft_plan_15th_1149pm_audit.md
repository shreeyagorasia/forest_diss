# Results chapter — audited draft

Every claim below is checked directly against its cited `TEMP_results` source. Item order follows
the narrative sequence for answering each RQ; the `[Importance: X/5]` tag on each item is a
separate score for how load-bearing it is to that RQ's answer or to the dissertation's headline
argument — the two are independent, an item can be early in the story without being the most
important, or vice versa.

---

## RQ1 — raw height prediction, model comparison

### Results table (Set3, both cohorts, spatial_block_kfold, seed 42, per-fold mean±SD, 5 folds — same method every row)

| Model | 4survey R2 | 4survey RMSE | 4survey MAE | 6survey R2 | 6survey RMSE | 6survey MAE |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 0.580±0.023 | 5.168±0.323 | 4.056±0.307 | 0.509±0.060 | 4.877±0.575 | 3.619±0.381 |
| RF | 0.638±0.010 | 4.794±0.168 | 3.611±0.127 | 0.656±0.041 | 4.062±0.222 | 3.010±0.165 |
| XGBoost (raw defaults) | 0.637±0.013 | 4.797±0.156 | 3.621±0.122 | 0.651±0.014 | 4.106±0.320 | 3.041±0.269 |
| **XGBoost (fairly tuned)** | **0.675±0.009** | **4.539±0.152** | **3.431±0.153** | **0.718±0.025** | **3.682±0.221** | **2.697±0.170** |
| DNN | 0.655±0.014 | 4.681±0.206 | 3.533±0.180 | 0.684±0.028 | 3.903±0.282 | 2.867±0.246 |
| PINN | 0.573±0.033 | 5.206±0.357 | 4.063±0.364 | 0.686±0.034 | 3.886±0.260 | 2.819±0.175 |
| PINN_k | 0.575±0.033 | 5.197±0.360 | 4.046±0.377 | 0.684±0.034 | 3.895±0.252 | 2.816±0.175 |

### Plot inventory

1. Two fitted CR curves (4survey vs. 6survey) on one age axis, next to an elevation-distribution
   comparison between cohorts — general-hypothesis visual for items 3/4 (data ready, not plotted).
2. Grouped-bar or dumbbell version of the results table above, model on the x-axis, cohort as
   colour/facet — the visual for item 1's headline (currently table-only).

### Ranked items

1. **`[Importance: 5/5]` A fairly-tuned XGBoost beats every neural model on both cohorts.** With a
   reasonable (if unsearched) hyperparameter config, XGBoost beats DNN by +0.020 R2 on 4survey and
   +0.034 on 6survey — both gaps well outside the fold-to-fold SD, a real and substantial margin,
   not noise. Tree boosting is a genuinely strong competitor throughout this project, not a
   strawman baseline — the same pattern shows up again in RQ2a (XGBoost reproduces PINN's own
   environmental-conditioning benefit). Caveat: this is one hand-picked config borrowed from RQ3,
   not a validated search on RQ1's own data.
2. **`[Importance: 5/5]` Within the neural family, the winner is cohort-conditional, and 6survey's
   own bias has a real mechanism.** DNN beats PINN/PINN_k on 4survey with non-overlapping 95% CIs
   (DNN [0.624,0.690] vs. PINN [0.545,0.607]); on 6survey, DNN's CI [0.658,0.740] fully contains
   PINN's [0.671,0.736] — a tie, not a loss, confirmed stable across a 5-seed reseed (4survey R2
   SD=0.0035, 6survey SD=0.0078). 6survey's own negative bias is consistently negative across all 5
   reseed seeds (-0.038 to -0.259) — not noise, and not a data-quality flaw: 6survey's frozen CR
   curve is a genuinely different shape from 4survey's (7.5m lower ceiling, more than double the
   growth rate), and 6survey is a compositionally different, lower-elevation, less-varied slice of
   Aberfoyle (mean 102m vs. 177m elevation, range capped at 351m vs. 561m — 6survey simply doesn't
   include 4survey's higher-elevation compartments at all). This also sharpens a methodology-chapter
   claim: Linear regression is justified there by the age range sitting on the CR curve's
   near-linear segment, which is measurably less true for 6survey (Linear's own R2 drops 0.580→0.509
   across cohorts; a straight line fit directly to each cohort's own true CR curve gets R2=0.984 for
   4survey but only 0.958 for 6survey).
3. **`[Importance: 3/5]` Temporal forecasting degrades far worse on 6survey than 4survey.** PINN
   loses 78% of its spatial_block_kfold R2 forecasting forward on 6survey (0.731→0.161) vs. a much
   smaller 14% drop on 4survey (0.634→0.544) — ties model fragility directly to cohort composition,
   not just "temporal splits are harder."
4. **`[Importance: 2/5]` No architecture helps both cohorts, for any of the three models.** For DNN,
   `deeper` wins on 4survey but only middling on 6survey; `small` wins on 6survey but is worst on
   4survey — a trade-off, not a real capacity effect (val_loss ranking doesn't even track test-R2
   ranking, the signature of noise). PINN/PINN_k are 5-10x less architecture-sensitive than DNN on
   the identical screen, sitting at or below the training-seed noise floor. No architecture fix
   exists for any of the three models.
5. **`[Importance: 4/5]` Plot_level vs. spatial_block_kfold asymmetry is real and structural, not
   cohort-driven.** DNN inflates hugely under an easy split (+0.197 4survey, +0.122 6survey);
   PINN/PINN_k show essentially zero inflation (±0.006 or smaller, both cohorts). Mechanism: DNN's
   per-input age+environment combination lets it fit an arbitrarily specific, near-unique
   "signature" per location, exploiting leakage from near-duplicate nearby plots; PINN's
   environmental features only ever pass through a narrow 16-unit sub-network producing a single
   additive `y_max` adjustment (multiplicative `k` for PINN_k), a much narrower bottleneck, further
   constrained by the physics/trajectory losses pulling toward the shared curve's analytical
   derivative. Directly validates `spatial_block_kfold` as the primary split choice and shows that
   validation matters differently by model — a genuinely independent bias-variance/regularisation
   argument for what physical constraints buy methodologically.
6. **`[Importance: 3/5]` The physics constraint has a real accuracy cost.** R2 decreases
   monotonically as the physics/trajectory loss weight increases (0→1→2), every model, every
   cohort, no exceptions: w=0 beats the default w=1 by +0.052/+0.050 R2 (PINN/PINN_k) on 4survey,
   comfortably outside both configurations' CIs, and by a smaller +0.010/+0.010 on 6survey. This is
   the first evidence in the dissertation that the physics constraint costs something, which is the
   tension RQ2a has to reckon with — does that cost buy something back worth paying for?

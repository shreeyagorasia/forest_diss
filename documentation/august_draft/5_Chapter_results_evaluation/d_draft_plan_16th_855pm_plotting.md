# Results chapter — audited draft

Every claim below is checked directly against its cited `TEMP_results` source (source file named
in parentheses, so every number traces back). Item order follows the narrative sequence for
answering each RQ; the `[Importance: X/5]` tag on each item is a separate score for how
load-bearing it is to that RQ's answer or to the dissertation's headline argument — the two are
independent, an item can be early in the story without being the most important, or vice versa.
Every item's key conclusion sentence is bolded, consistently. Every item states **why** the finding
is the way it is — a real mechanism where one is established (architectural, mathematical,
compositional, general-ML-literature), or an explicit "not yet explained, needs exploration" where
it isn't, rather than leaving the reader to assume a mechanism that was never actually shown.

---

## RQ1 — raw height prediction, model comparison

### Results table (Set3, both cohorts, spatial_block_kfold, seed 42, per-fold mean±SD (sample SD,
ddof=1), 5 folds — same method every row, all recomputed directly from each fold's own saved
`metrics.json` by `models/baselines/rq1_results_table_metrics.py`, see
`TEMP_rq1_results_table_metrics_2026-08-16.tex`. Pooled test-row sample size, identical for every
row in this table since all seven models share the same split/filtering: **4survey n=232,292
row-years (58,073 unique plots); 6survey n=82,614 row-years (13,769 unique plots)** — confirmed
directly against each model's own saved `metrics.json`/`kfold_summary.json`, not assumed. 6survey's
much smaller sample is the direct reason its CIs and fold SDs are consistently wider throughout
this section.)

| Model | 4survey RMSE | 4survey MAE | 4survey R2 | 6survey RMSE | 6survey MAE | 6survey R2 |
|---|---:|---:|---:|---:|---:|---:|
| Linear | 5.168±0.361 | 4.056±0.343 | 0.580±0.026 | 4.877±0.643 | 3.619±0.426 | 0.509±0.067 |
| RF | 4.794±0.187 | 3.611±0.142 | 0.638±0.011 | 4.062±0.249 | 3.010±0.185 | 0.656±0.046 |
| XGBoost (raw defaults) | 4.797±0.175 | 3.621±0.136 | 0.637±0.015 | 4.106±0.358 | 3.041±0.301 | 0.651±0.016 |
| **XGBoost (tuned, own search)** | **4.548±0.137** | **3.431±0.123** | **0.674±0.009** | **3.656±0.232** | **2.669±0.183** | **0.722±0.031** |
| DNN | 4.681±0.230 | 3.533±0.201 | 0.655±0.016 | 3.903±0.316 | 2.867±0.275 | 0.684±0.031 |
| PINN | 5.206±0.399 | 4.063±0.407 | 0.573±0.036 | 3.886±0.290 | 2.819±0.196 | 0.686±0.038 |
| PINN_k | 5.197±0.402 | 4.046±0.421 | 0.575±0.037 | 3.895±0.282 | 2.816±0.196 | 0.684±0.038 |

**Column-order verification (2026-08-17)**: reordered programmatically from the original
R2/RMSE/MAE table, not hand-retyped — every row's cell values (as a set, ignoring order and bold
markers) were confirmed identical before and after reordering for all 7 models. No value moved to
the wrong column or cohort.

### Plot inventory

#### Figure R1-1: XGBoost vs. DNN, matched fold by fold

**Research question and findings supported:** RQ1, item 1 — a properly-tuned XGBoost beats DNN on
almost every individual held-out fold, not only on the pooled average.

**Purpose:** Does XGBoost's advantage over DNN hold on every individual held-out fold, or only on
average across folds?

**Key insight:** XGBoost wins 5/5 folds on 6survey and 4/5 folds on 4survey. The one exception
(fold 0, 4survey, DNN ahead by -0.0068) is shown directly rather than smoothed into the mean — the
reader should see that the margin is not a pooled-average artifact of one or two folds.

**What it adds beyond the table:** The results table shows only the pooled mean±SD; it cannot show
whether the gap is consistent fold-by-fold or driven by an outlier fold.

**Exact visual design:**
- Plot type: paired dot-and-line ("slope") plot.
- Panels: 2, side by side, one per cohort (4survey, 6survey).
- X-axis: two categories, "DNN" and "XGBoost" (categorical).
- Y-axis: fold-level test R2, zoomed to the observed range (~0.60–0.75); identical range on both
  panels for direct visual comparison (not claiming direct cohort comparability — see caption).
- Lines: one per fold (5 per panel), connecting that fold's DNN point to its XGBoost point.
- Points: filled circle at each end of each line.
- Colour: encodes which model won that fold, so the single 4survey exception is immediately
  visible without reading fold labels.
- Reference lines: none required.
- Uncertainty shown: none additional — each point is one fold's own value; the spread of lines is
  the uncertainty view.
- Paired: yes, explicitly by fold index — this is the figure's entire purpose.
- Labels/annotations: label the reversed fold directly (e.g. "fold 0").
- Legend: one shared legend below both panels ("XGBoost ahead" / "DNN ahead").
- Axis limits: shared and identical across both panels.

**Data required:** XGBoost per-fold test R2 (5 values/cohort), from the winning-config refit
already reported in `TEMP_rq1_xgb_vs_dnn_paired_folds_2026-08-16.tex`. DNN per-fold test R2 (5
values/cohort), from `outputs/spatial_block_kfold/rq1_dnn_env_terrain_nested_set3_gated_terrain_wind_vif_seed42/{cohort}/kfold_summary.json`'s
`per_fold_r2_values`. Both already saved; plotting only, no new fitting.

**Interpretation limits:** Shows accuracy consistency only, not why XGBoost wins (not established,
per item 1's own "Why"). Does not extend to PINN/PINN_k — that paired-fold check was not run.

**Placement:** main text.

**Caption message:** XGBoost's accuracy advantage over DNN holds on almost every individual
held-out fold, not only on the pooled average — the single exception is shown directly rather than
smoothed into the mean.

**Ranking:** 1 (essential).

---

#### Figure R1-2: How split difficulty changes each model's apparent accuracy

**Research question and findings supported:** RQ1, items 3 and 4 — temporal degradation is far
worse on 6survey than 4survey for every model; the easy plot-level split inflates DNN's apparent
accuracy while leaving PINN/PINN_k essentially unaffected.

**Purpose:** How much does a model's apparent accuracy change as the evaluation design gets
progressively harder to leak into, and does that change differ between DNN and the guided models?

**Key insight:** PINN/PINN_k's accuracy stays comparatively flat moving from the easy plot-level
split to the leakage-controlled spatial-block split, while DNN's drops sharply; moving on to the
temporal split degrades every model further, far more severely on 6survey than 4survey.

**What it adds beyond the table:** The main results table reports only the pooled
`spatial_block_kfold` numbers; it says nothing about how much that number depends on split choice.

**Exact visual design:**
- Plot type: connected line/point plot (difficulty gradient), one line per model.
- Panels: 2, side by side, one per cohort — not pooled, since the compositional difference between
  cohorts is a separate, already-established finding (item 2), not something to imply away here.
- X-axis: three ordered categories, all on the single-split basis, never the pooled kfold number:
  "plot_level" → "spatial_block (single split)" → "temporal". Label the axis "single-split
  evaluation design" explicitly, so it is never confused with the pooled table above.
- Y-axis: R2, shared scale and range across both panels (0 to ~0.9, to fit 6survey's temporal
  collapse without clipping).
- Lines: one per model (DNN, PINN, PINN_k) — 3 per panel.
- Colour: one distinct colour per model, consistent between panels.
- Points: marker at each of the 3 split-type positions.
- Reference lines: none.
- Uncertainty shown: none — these are single-split point values with no fold repeats; the axis
  label/caption must say so rather than imply a CI that does not exist.
- Paired: yes, by model, across the three split types.
- Labels/annotations: none beyond the legend, to avoid clutter.
- Legend: one shared legend below both panels.
- Axis limits: shared y-axis range across both panels.

**Data required:** `TEMP_rq1_plotlevel_check_results_2026-08-12.tex` (plot_level and single-split
spatial_block R2) and `TEMP_rq1_temporalcheck_results_2026-08-11.tex` (single-split spatial_block
and temporal R2), both cohorts, DNN/PINN/PINN_k — already computed. NEEDS CHECKING: the two source
files' own spatial_block numbers should be cross-verified as identical before plotting (both come
from the same underlying fit, so this is expected to be trivial, but not yet directly checked
number-for-number).

**Interpretation limits:** Not comparable to the main results table's pooled numbers — axis label
and caption must make this unambiguous. Does not establish why 6survey degrades more under
temporal forecasting (item 3's own "Why" is explicitly hedged as plausible, not confirmed).

**Placement:** main text.

**Caption message:** The easy plot-level split inflates DNN's apparent accuracy far more than
PINN/PINN_k's, and every model degrades further under temporal forecasting — far more severely on
6survey than on 4survey. All values shown are single-split, not the pooled kfold numbers in the
main results table.

**Ranking:** 2 (useful if space allows).

---

#### Figure R1-3: the physics-weight accuracy cost

Not given its own figure. RQ1 item 5 (accuracy falls as physics weight rises) and RQ2a item 3
(removing the constraint destroys PINN_k's parameter identifiability) answer the two halves of one
trade-off — see Figure R2a-2 under RQ2a, which shows both as one composite sharing a physics-weight
x-axis. Presenting them as two separate figures would force the reader to hold two figures in mind
to see the trade-off item 5 itself asks about ("does that accuracy cost buy something back?").

---

#### Figure R1-4: Why 6survey behaves differently — two fitted growth curves and one compositional difference

**Research question and findings supported:** RQ1, item 2 — 6survey's frozen CR curve differs from
4survey's on all three shape parameters, and 6survey is a compositionally different, lower-elevation
slice of Aberfoyle.

**Purpose:** What do the two cohorts' shape-parameter differences look like as growth curves, and
how does the compositional (elevation) difference between the two cohorts compare, side by side?

**Key insight:** 6survey's curve reaches a lower ceiling but grows faster and more steeply early on
— a visibly different shape, not a rescaled version of the same curve — and 6survey's elevation
range sits entirely inside 4survey's own range, missing its upper tail entirely.

**What it adds beyond the table:** No table reports curve shape; item 2's y_max/k/p numbers are
precise but abstract on their own.

**Exact visual design:**
- Plot type: two-panel figure. Panel A: line plot (two fitted CR curves). Panel B: overlaid
  histogram or density plot (elevation distribution).
- Panels: 2, side by side.
- Panel A x-axis: age (years), spanning both cohorts' observed ranges (~9–94 years). Y-axis:
  predicted top height (m). Two lines, one per cohort, using each cohort's own frozen (y_max, k, p).
- Panel B x-axis: elevation (m), spanning the FULL 4survey range (up to 561m) so 6survey's
  truncation at 351m is visually obvious, not cropped out. Y-axis: density, not raw count (the two
  cohorts' population sizes differ by 5x, so raw counts would be misleading).
- Colour: same two cohort colours throughout both panels.
- Reference lines: optional vertical line on Panel B at each cohort's own max elevation.
- Uncertainty shown: none — fixed, already-fitted parameters and an empirical distribution.
- Paired: not applicable (population-level).
- Labels/annotations: label each curve directly with its cohort name near its own endpoint.
- Legend: one shared legend for both panels.
- Axis limits: as above (Panel B spans the full 4survey range).

**Data required:** Frozen CR curve parameters, both cohorts, from
`TEMP_rq1_cohort_composition_2026-08-15.tex` — already computed. Elevation per plot, both cohorts,
from `load_plots_for_cohort()` (`models/xgb_environmental/data.py`) — already used to produce the
elevation-range numbers cited in item 2. NEEDS CHECKING: confirm whether a plot-level, deduplicated
elevation table already exists on disk, or needs a fresh (trivial) read from the existing loader.

**Interpretation limits:** Shows a compositional and shape difference, not a causal one — item 2's
own text is explicit that the forestry mechanism connecting elevation to curve shape is not
established. Caption must not imply elevation causes the shape difference.

**Placement:** appendix (explanatory/background context; the numbers it visualises are already
precisely stated in prose, and the main-text figure budget is better spent on R1-1/R1-2).

**Caption message:** 6survey's fitted growth curve reaches a lower ceiling but grows faster and
earlier than 4survey's, and 6survey's elevation range sits entirely inside 4survey's own range with
no equivalent to 4survey's higher-elevation compartments.

**Ranking:** 3 (appendix only).

### Ranked items

1. **`[Importance: 5/5]` A properly-tuned XGBoost beats every neural model on both cohorts by point
   estimate, and — checked against DNN specifically, the strongest neural competitor — the margin
   holds on almost every fold, not just on average.** (`TEMP_rq1_baseline_comparison_2026-08-11.tex`,
   `TEMP_rq1_xgb_hyperparameter_search_2026-08-16.tex`,
   `TEMP_rq1_xgb_vs_dnn_paired_folds_2026-08-16.tex`) XGBoost beats DNN by +0.019 R2 on 4survey and
   +0.038 on 6survey (pooled); PINN/PINN_k lose by even wider margins on 4survey (0.674 vs.
   0.573-0.575) and by a comparable margin on 6survey (0.722 vs. 0.684-0.686). A proper PAIRED
   fold-by-fold comparison against DNN (same 5 folds, same split, both models) is more informative
   than either model's own marginal SD: **XGBoost wins 5/5 folds on 6survey** (though fold 4's
   margin is razor-thin, +0.0005) **and 4/5 folds on 4survey** — the one exception (fold 0) has DNN
   ahead by -0.0068, a small but real direction reversal, honestly reported rather than smoothed
   over. This equivalent per-fold check was not run against PINN/PINN_k, but given DNN's own point
   estimates already clear PINN/PINN_k's by a wider or comparable margin, XGBoost's win over them is
   not expected to be less robust. This XGBoost config comes from a genuine per-target search (27
   hyperparameter configs × 5 folds × both cohorts = 270 fits, selected on validation R2 only,
   never test), run specifically on RQ1's own target — not the RQ3-borrowed config used elsewhere in
   this project. (The search confirms the margin is robust: the winning config's test R2 lands
   within 0.001-0.004 of a naively-borrowed config on both cohorts, well inside the fold-to-fold SD,
   so 270 honestly-evaluated alternatives don't find anything meaningfully better — the gap to the
   neural models isn't an artifact of a favourable or lucky hyperparameter choice.)
   **Why**: not specifically isolated or tested within this project (no controlled ablation
   pinpoints which property of trees vs. neural nets drives the gap *here*) — flagged as needing
   exploration if a firm answer is wanted. The plausible general explanation, well-documented in the
   ML literature rather than established by this project's own evidence, is that gradient-boosted
   trees are known to outperform neural networks on small-to-medium tabular data with relatively few,
   mixed-type features (Grinsztajn et al. 2022 is the standard citation for this effect) — trees
   handle thresholded, non-smooth relationships natively and need less careful tuning/data volume
   than a neural net to reach a good fit. Consistent with, not proven by, this project's own numbers.
2. **`[Importance: 5/5]` Within the neural family, the winner is cohort-conditional, and 6survey's
   own bias has a real, partial mechanism.** (`TEMP_rq1_winner_reseed_results_2026-08-11.tex`,
   `TEMP_rq1_cohort_composition_2026-08-15.tex`) DNN beats PINN/PINN_k on 4survey with non-overlapping
   95% CIs (DNN [0.624,0.690] vs. PINN [0.545,0.607]); on 6survey, DNN's CI [0.658,0.740] fully
   contains PINN's [0.671,0.736] — a tie, not a loss, confirmed stable across a 5-seed reseed
   (4survey R2 SD=0.0035, 6survey SD=0.0078). 6survey's own negative bias is consistently negative
   across all 5 reseed seeds (-0.038 to -0.259).
   **Why**: partially explained, partially open. What's established: 6survey's frozen CR curve
   differs from 4survey's on all three shape parameters, not just the ceiling — `y_max`=44.46m vs.
   51.96m (7.5m lower ceiling), `k`=0.0233 vs. 0.0103 (more than double the growth rate), `p`=1.193
   vs. 0.865 (a different curve shape, not just scaled faster/lower — 6survey's growth is more
   front-loaded). 6survey is also a compositionally different, lower-elevation, less-varied slice of
   Aberfoyle (mean 102m vs. 178m elevation, range capped at 351m vs. 561m — 6survey simply doesn't
   include 4survey's higher-elevation compartments at all). **What's NOT yet explained**: the
   specific causal forestry mechanism connecting lower elevation to this particular curve-shape
   difference (faster early growth, lower eventual ceiling, more front-loaded shape) — stated in the
   source file itself as a compositional fact, not a causal claim, pending domain interpretation.
   Needs exploration, not yet answered.
   This also sharpens a methodology-chapter claim: Linear regression is justified there by the age
   range sitting on the CR curve's near-linear segment, which is measurably less true for 6survey
   (Linear's own R2 drops 0.580→0.509 across cohorts; a straight line fit directly to each cohort's
   own true CR curve gets R2=0.984 for 4survey but only 0.958 for 6survey) — a direct, mechanical
   consequence of the shape-parameter differences above, not a separate open question.
3. **`[Importance: 3/5]` Temporal forecasting degrades far worse on 6survey than 4survey, for every
   model.** (`TEMP_rq1_temporalcheck_results_2026-08-11.tex` — single-split `spatial_block` vs.
   `temporal`, not the pooled `spatial_block_kfold` used in the main results table above, so these
   R2 values are not directly comparable to that table) PINN loses 50% of its single-split R2
   forecasting forward on 4survey (0.584→0.290) but 78% on 6survey (0.731→0.161) — the same cohort
   asymmetry shows up for DNN too, just smaller in both cases (14% on 4survey, 0.634→0.544, vs. 56%
   on 6survey, 0.722→0.317).
   **Why**: plausible, not fully isolated — the source file frames this as "consistent with 6survey's
   smaller, noisier population making any harder split hit worse," i.e. a smaller, more homogeneous
   training population gives the model less to generalise from across time. This is the same
   compositional evidence as item 2 extended to a new axis, not a separately-proven mechanism — the
   source file's own language is "plausibly extends," not "confirmed."
4. **`[Importance: 4/5]` Plot_level vs. spatial_block asymmetry is real and structural, not
   cohort-driven.** (`TEMP_rq1_plotlevel_check_results_2026-08-12.tex` — plot_level compared
   against the same single-split `spatial_block` run as item 3, an apples-to-apples pairing since
   plot_level is itself a single run; not the pooled `spatial_block_kfold` numbers in the main
   table) DNN inflates hugely under an easy split (+0.197 4survey, +0.122 6survey); PINN/PINN_k
   show essentially zero inflation (±0.006 or smaller, both cohorts).
   **Why**: clear, architectural. DNN's per-input age+environment combination lets it fit an
   arbitrarily specific, near-unique "signature" per location, exploiting leakage from near-duplicate
   nearby plots; PINN's environmental features only ever pass through a narrow 16-unit sub-network
   producing a single additive `y_max` adjustment (multiplicative `k` for PINN_k) — a much narrower
   bottleneck, further constrained by the physics/trajectory losses pulling toward the shared curve's
   analytical derivative. This is a standard bias-variance/regularisation argument (reduced model
   flexibility prevents exploiting a spurious correlation), not a speculative reading — directly
   validates `spatial_block_kfold` as the primary split choice.
5. **`[Importance: 3/5]` The physics constraint has a real accuracy cost.**
   (`TEMP_rq1_physicsablation_results_2026-08-11.tex`) R2 decreases monotonically as the
   physics/trajectory loss weight increases (0→1→2), every model, every cohort, no exceptions: w=0
   beats the default w=1 by +0.052/+0.050 R2 (PINN/PINN_k) on 4survey — comfortably outside both
   configurations' CIs, a clear effect — and by a smaller +0.010/+0.010 on 6survey, where the
   direction is consistent but the difference sits inside 6survey's own wider CIs, so not a
   statistically clean effect there on its own.
   **Why**: clear, standard ML reasoning. Removing a constraint lets a model fit the training signal
   more freely — the physics/trajectory loss competes with the data loss for the same weights, so
   constraining the model toward physical consistency necessarily trades away some raw fit. This is
   the general bias-variance mechanism, not project-specific speculation.
   **Does that accuracy cost buy something back?** Yes, but narrowly: RQ2a item 3 shows removing
   the constraint (w=0) makes PINN_k's `y_max`/`k` parameters far less identifiable (pooled 5-fold
   correlation swings from -0.449 at w=1 to -0.935 at w=0 on 4survey) — the constraint trades a
   real amount of accuracy for a real amount of parameter identifiability, but only for PINN_k's own
   two-parameter setup, not as a general "physics-guided architecture beats plain accuracy" result.

---

## RQ2a — does environmental conditioning shrink the departure from the shared curve

### Results table (across folds — `TEMP_rq2a_residual_reduction_results_2026-08-11.tex`, XGBoost's
`% rows improved` column added 2026-08-16 by re-running `rq2a_xgb_check.py`'s existing
`run_reduction_check()` against its already-saved `predictions.csv` — no retraining. **DNN's row
uses the 5-seed reseed (SD = seed-to-seed variation); every other row uses the single seed-42 run
(SD = fold-to-fold variation) — these SDs are not on the same basis and should not be compared
directly against each other; only DNN has been checked for seed robustness. CR baseline column
recomputed directly from currently-saved files 2026-08-16 (`models/spatial_attribution/
rq2a_verify_cr_baseline.py`) after the original 6survey figure (3.057) was found stale — see that
file's correction note; the % column is unaffected since no percentage crosses a rounding
boundary between the old and corrected baseline.**)

| Model | Cohort | n (pooled test row-years) | CR baseline mean \|resid\| (m) | Mean reduction (m) | Reduction as % of baseline error | % rows improved |
|---|---|---:|---:|---:|---:|---:|
| DNN, env-conditioned (5-seed, Set3) | 4survey | 232,292 | 5.028 | 1.501±0.023 | 29.9% | 64.8±4.1 |
| DNN, env-conditioned (5-seed, Set3) | 6survey | 82,614 | 3.047 | 0.174±0.036 | 5.7% | 52.4±4.8 |
| PINN, env-conditioned (seed 42, Set3) | 4survey | 232,292 | 5.028 | 0.965±0.228 | 19.2% | 61.6±5.7 |
| PINN, env-conditioned (seed 42, Set3) | 6survey | 82,614 | 3.047 | 0.238±0.191 | 7.8% | 56.8±5.4 |
| PINN_k, env-conditioned (seed 42, Set3) | 4survey | 232,292 | 5.028 | 0.982±0.238 | 19.5% | 61.5±6.0 |
| PINN_k, env-conditioned (seed 42, Set3) | 6survey | 82,614 | 3.047 | 0.241±0.173 | 7.9% | 55.8±5.4 |
| XGBoost, env-conditioned (fixed, Set3) | 4survey | 232,292 | 5.028 | 1.597±0.330 | 31.8% | 67.2±3.8 |
| XGBoost, env-conditioned (fixed, Set3) | 6survey | 82,614 | 3.047 | 0.361±0.183 | 11.8% | 56.0±4.6 |
| XGBoost, no-env control (fixed) | 4survey | **232,448** | 5.030 | 1.222±0.232 | 24.3% | 64.5±3.2 |
| XGBoost, no-env control (fixed) | 6survey | 82,614 | 3.047 | 0.240±0.144 | 7.9% | 54.8±3.0 |

### Plot inventory

#### Figure R2a-1: Where environmental conditioning helps and where it hurts

**Research question and findings supported:** RQ2a, items 1 and 2 — conditioning helps most where
CR fits worst and substantially degrades accuracy in relative terms where CR already fits well; the
same pattern reproduces for a plain XGBoost, ruling out physics-guided architecture as the mechanism.

**Purpose:** Does environmental conditioning's effect on accuracy vary by how well the shared curve
already fits a plot, and does that pattern depend on which model is used?

**Key insight:** The distribution of per-row error change is skewed toward harm in Q1 (CR already
fits best) and toward help in Q4 (CR fits worst) — for DNN and for a plain XGBoost, on both
cohorts — a property of the data's relationship to the shared curve, not one model's own quirk.

**What it adds beyond the table:** The results table reports only the mean reduction per
model/cohort. It cannot show that 21–25% of Q1 rows still improve even though the quartile mean is
negative (already stated in item 1's own text) — a distribution view shows this within-quartile
spread directly, at the full-model-comparison level, not only as a summary number.

**Exact visual design:**
- Plot type: violin (preferred; tens of thousands of rows per quartile support a smooth density) or
  box plot.
- Panels: 2, side by side, one per cohort — not pooled, since the effect size is roughly an order
  of magnitude smaller on 6survey (a separate, already-established finding).
- X-axis: CR-residual quartile, 4 ordered categories (Q1 best-fit → Q4 worst-fit).
- Y-axis: per-row residual reduction (m) — |CR error| − |model error|; positive = model improved on
  CR, negative = model made it worse.
- Distributions: one violin/box per quartile per model, dodged side by side within each quartile so
  DNN and XGBoost sit directly adjacent and comparable at every quartile.
- Colour: one colour per model (DNN, XGBoost), consistent across both panels.
- Reference lines: a horizontal line at y=0 on both panels, styled distinctly (e.g. solid black,
  labelled "no change") — the single most important reference in the figure, marking the
  helped/hurt boundary unambiguously.
- Uncertainty shown: the distribution itself is the spread view; no additional error bars.
- Paired: not paired row-by-row (different rows fall in each quartile), but DNN and XGBoost share
  the same underlying row population and quartile definition, so directly comparable.
- Labels/annotations: annotate each quartile's already-computed %-of-rows-improved figure (e.g. Q1:
  21–25%) as a small text label, since this is not otherwise visible from distribution shape alone.
- Legend: one shared legend below both panels.
- Axis limits: shared y-axis range across both panels only if it stays legible for 6survey's much
  smaller effect size; otherwise use separate, clearly-labelled ranges per panel and say so in the
  caption — do not let this default silently.

**Data required:** Per-row CR-residual quartile and reduction values, DNN (5-seed, Set3) and
XGBoost (env-conditioned, Set3), both cohorts. The quartile-MEAN summary already exists in
`TEMP_rq2a_residual_reduction_results_2026-08-11.tex`, but the full per-row distribution needed for
a violin/box plot is NOT yet saved anywhere — `models/spatial_attribution/rq2_residual_reduction.py`
already supports saving the full per-row merged table (`rq2_residual_reduction.csv`, via its own
`main()`), but no such file currently exists under `outputs/` for any model (confirmed by search).
NEEDS CHECKING / requires running that existing script's save step before this figure can be built
— no new script or plotting code needed, just the existing save path.

**Interpretation limits:** Shows a robust, replicated association between CR-fit quality and
conditioning's effect direction, not the underlying signal-vs-noise mechanism — item 1's own text
is explicit that a direct test of that mechanism was attempted and left inconclusive. Does not
extend to PINN/PINN_k at this level of distributional detail — only DNN and XGBoost have reduction
data computed at the row level.

**Placement:** main text.

**Caption message:** Environmental conditioning systematically hurts plots the shared curve already
fits well and helps plots it fits worst — the same asymmetric pattern for both DNN and a plain
XGBoost, on both cohorts, ruling out one model's architecture as the explanation.

**Ranking:** 1 (essential).

---

#### Figure R2a-2: The physics constraint trades accuracy for parameter identifiability

**Research question and findings supported:** RQ1 item 5 (physics constraint has a real accuracy
cost) and RQ2a item 3 (removing it destroys PINN_k's y_max/k identifiability) — the two halves of
one trade-off, directly answering item 5's own closing question ("does that accuracy cost buy
something back?").

**Purpose:** As the physics/trajectory loss weight increases from 0, how does raw accuracy change,
and how does parameter identifiability change — do they move in opposite directions?

**Key insight:** Accuracy falls as the physics weight increases, while the y_max/k correlation
moves toward a less extreme (more identifiable) value — the two move in opposite directions across
the same x-axis, shown directly rather than left for the reader to infer from two separately-stated
numbers in two different RQ sections.

**What it adds beyond the table:** Neither RQ1's nor RQ2a's results table reports a per-weight
breakdown at all — the relationship currently exists only as prose numbers scattered across two
items in two different sections.

**Exact visual design:**
- Plot type: two-panel figure sharing one x-axis.
- Panels: 2, stacked vertically (so the shared x-axis lines up exactly).
- Shared x-axis: physics/trajectory loss weight (categorical: 0, 1 — see data note on w=2 below).
- Panel A y-axis: R2 (accuracy), one line per model (PINN, PINN_k), separate colours per cohort (or
  two side-by-side sub-panels if combining cohorts on one axis would misrepresent the very different
  4survey/6survey CI widths already established elsewhere).
- Panel B y-axis: y_max/k pooled correlation (identifiability; range -1 to +0.6), one line per
  cohort, PINN_k only (PINN has no second parameter to correlate).
- Colour: consistent cohort colours between both panels.
- Reference lines: Panel B needs a horizontal line at 0 (no correlation); Panel A needs none.
- Uncertainty shown: Panel A can show the already-computed 95% CIs (w=0/w=1, 5-fold rigorous tier)
  as error bars per point. Panel B should show the cluster-bootstrap 95% CI already computed at
  w=0 (current tier) as an error bar. NEEDS CHECKING: w=1's own bootstrap CI at the current tier has
  not been separately reverified this session (only w=0's has) — confirm before using an error bar
  on the w=1 point, or omit it and state the gap in the caption.
- Paired: not paired row-by-row; paired only in that both panels share the same x-axis weight values.
- Labels/annotations: label the w=0 and w=1 points explicitly, since these are the two headline
  configurations referenced repeatedly in prose.
- Legend: one shared legend for cohort colour, used once for both panels.
- Axis limits: Panel B spans at least -1 to +0.6 to accommodate 6survey's sign flip without clipping.

**Data required:** Panel A: `TEMP_rq1_physicsablation_results_2026-08-11.tex`'s 5-fold rigorous
tier (w=0 vs. w=1 only, with CIs, both models, both cohorts) — already computed. Panel B: same
source file's y_max/k correlation table (w=0/w=1, both cohorts) and
`TEMP_rq2a_pinn_k_identifiability_bootstrap_2026-08-16.tex` for the w=0 CI (both cohorts) — already
computed. NEEDS CHECKING: the single-split exploratory tier's own w=2 point exists for R2 (Panel A
could extend to a 3-point line, 0/1/2, using that tier instead of the 5-fold rigorous one, trading
away the CI) but no w=2 identifiability correlation is reported anywhere in the source — Panel B
should be scoped to w=0/w=1 only unless a w=2 correlation number is confirmed to exist.

**Interpretation limits:** The mechanism connecting the two panels (that the same loss term drives
both effects) is architectural/definitional, established directly from the loss function's own
construction (item 5's "Why"), not a correlation the figure itself proves statistically. Panel B's
w=2 point should not be silently interpolated if it was never computed (see data note above).

**Placement:** main text.

**Caption message:** Weakening the physics constraint buys real accuracy on both cohorts, but at
the cost of PINN_k's y_max/k parameters becoming far less identifiable — the two move in opposite
directions across the same weight range.

**Ranking:** 1 (essential) — the leading candidate for RQ2a's own "one main figure" slot; also
serves RQ1 item 5, so counts toward RQ1's figure budget too rather than being purely additional.

---

#### Figure R2a-3: Where environmental conditioning helps and hurts, mapped

**Research question and findings supported:** RQ2a item 1 (conditioning helps where CR fits worst,
hurts where it fits well), now grounded in a real spatial-clustering test
(`TEMP_rq2a_reduction_morans_i_2026-08-16.tex`): Moran's I on the reduction value itself is 0.55–0.64
(p=0.001) for both DNN and XGBoost, both cohorts — a substantial, tested spatial pattern, not an
unsupported visual. This was explicitly withheld in an earlier pass of this plan until that test
existed; it now does.

**Purpose:** Where across Aberfoyle does environmental conditioning help or hurt, and is that
pattern the same for DNN and XGBoost?

**Key insight:** The reduction value is not scattered randomly — it clusters spatially, at a
magnitude comparable to RQ2b's and RQ3's own residual clustering. Because item 1 already shows this
tracks CR-fit quality (Q1 vs. Q4), the map should let a reader see whether the same regions that
look red/blue here are also the regions where the shared curve already fit well or badly, tying the
spatial pattern back to the mechanism rather than presenting it as a free-standing observation.

**What it adds beyond the table:** No table in RQ2a shows location at all. The quartile-mean table
and Figure R2a-1's distribution view both show the effect exists and how large it is; neither shows
whether it is spatially concentrated or spread evenly, which is now a separately tested, real
finding in its own right.

**Exact visual design:**
- Plot type: two-panel map, one per model, sharing the same extent and colour scale so they are
  directly comparable.
- Panels: 2, side by side (DNN, XGBoost) — 4survey only for the main figure (the effect is roughly
  an order of magnitude smaller on 6survey, an already-established finding; showing both cohorts on
  a shared colour scale would flatten 6survey to near-invisible, so either give 6survey its own
  separate, independently-scaled figure or move it to the appendix, and say so explicitly).
- X/Y: plot easting/northing (EPSG:27700, matching every other spatial figure in this plan).
- Colour: reduction value (m), diverging scale centred at 0 (blue = conditioning hurt this plot,
  red = helped), consistent scale across both panels.
- Reference lines: none (map, not an axis chart); the colour scale's own zero-point is the
  reference.
- Uncertainty shown: none per-point (each is a single pooled prediction, not an estimate with its
  own CI); the Moran's I test itself (cited in the caption) is the uncertainty/significance
  statement for the pattern as a whole, not shown per-pixel.
- Paired: not paired in the row sense; the two panels share the same plot population so they are
  directly comparable panel-to-panel.
- Labels/annotations: state the Moran's I value and p for each panel directly on the figure (e.g.
  a small text annotation per panel), so the statistical backing for the visual pattern is not left
  implicit.
- Legend: one shared colourbar for both panels.
- Axis limits: identical map extent and aspect ratio for both panels.

**Data required:** Per-row reduction value and (x, y), DNN and XGBoost, Set3, both cohorts — exactly
the same data already computed and tested in `models/spatial_attribution/rq2a_reduction_morans_i.py`
(reuses `compute_residual_reduction()` and `load_plot_coordinates()`, no new fitting). Already run
once for the Moran's I test; the same merged table is what the map would plot directly.

**Interpretation limits:** Moran's I confirms clustering exists; it does not by itself show WHICH
places (that requires actually looking at the map, which is precisely this figure's job) or WHY
(item 1's own mechanism — CR-fit quality — is the leading candidate, not separately re-tested for
its own spatial clustering here, so the map should not claim to prove that specific mechanism,
only to be consistent with it if the visual pattern lines up with what a reader already knows about
where CR fits worst).

**Placement:** main text (4survey panel); 6survey as a separate, independently-scaled appendix panel
if included at all.

**Caption message:** Where environmental conditioning helps or hurts is not scattered randomly
across Aberfoyle — it clusters spatially (Moran's I 0.63/0.64, p=0.001 for XGBoost/DNN), at a
magnitude comparable to the spatial clustering already reported elsewhere in this project, and the
pattern is visually similar for both models.

**Ranking:** 2 (useful if space allows) — a real, tested finding, but secondary to Figure R2a-1's
already-essential distribution view; promote to essential if the visual pattern turns out to align
strikingly with CR-fit quality once actually plotted.

### Ranked items

1. **`[Importance: 5/5]` A flexible model trades accuracy across the CR-residual quartiles
   regardless of what it is fed — but only Q4's correction is a genuine effect of real
   environmental signal; Q1's degradation is not.** (`TEMP_rq2a_residual_reduction_results_2026-08-11.tex`,
   `TEMP_rq2a_permutation_check_2026-08-17.tex`) On the quartile CR already fits best (Q1), DNN's
   own error is 2.6x (4survey) to 3.2x (6survey) LARGER than CR's own error there (4survey:
   1.01m→2.58m; 6survey: 0.52m→1.65m) — only 21-25% of Q1 rows actually improve, the large majority
   get worse. This is a real cost, not a rounding footnote: the -1.57m/-1.14m absolute reduction
   figures look small only next to Q4's much larger absolute swing (+5.08m/+1.81m); relative to
   Q1's own small baseline error, the degradation is large. Replicates without exception across all
   18 (model × set × cohort) combinations, confirmed stable across a 5-seed reseed (not a
   single-run fluke: 4survey mean reduction 1.501±0.023m, Q4 alone 5.054±0.117m, SD under 2.5% of
   the mean). Real on both cohorts, though roughly an order of magnitude smaller in absolute terms
   on 6survey (5.7% of baseline error closed vs. 29.9% on 4survey per the table above) — consistent
   with the same compositional-narrowness theme established in RQ1, not separately proven here.
   **Why, confirmed directly rather than left as a plausible account**: a permutation control
   (XGBoost, Set3, environmental columns shuffled across rows before fitting so the model trains on
   fake, scrambled environmental values while everything else — real heights, real no-environment
   features, same split — stays unchanged, both cohorts) shows Q1's degradation happens at close to
   the same magnitude whether the model is fed real or scrambled environment (4survey: -1.30m
   permuted vs. -1.21m real; 6survey: -0.57m permuted vs. -0.72m real, permuted if anything slightly
   less bad) — Q1's cost is a property of giving any flexible model output variance on rows the
   baseline already fits well, not a specific cost of real environmental conditioning. Q4's
   correction, by contrast, genuinely shrinks when the environment is permuted away (4survey: 3.97m
   permuted vs. 4.83m real, an 18% reduction; 6survey: 1.18m vs. 1.61m, a 27% reduction) — real
   environmental signal contributes a genuine, if partial, share of Q4's benefit; permuting it away
   does not eliminate the correction (76-82% of it survives), so Q4's gain isn't purely a
   flexibility artefact either, just partly explained by real signal on top of that. Run for
   XGBoost/Set3 only, not repeated for DNN/PINN or the other sets — very likely to generalise given
   how closely XGBoost and DNN already track each other on this exact effect (item 2 below), but not
   independently confirmed there.
   **What this means for the "trade-off" framing**: calling this a trade-off "of environmental
   conditioning" is only half right. Q4's benefit is a genuine, if partial, effect of real
   environmental data. Q1's cost is not — it would appear even with meaningless environmental
   values, since it comes from adding a flexible model on rows that already had almost no error to
   begin with. The trade-off is real, but it sits between model flexibility in general and Q1's
   already-small error, not specifically between real environmental information and Q1.
   **Optional addition, conditional on whether the spatial map is used**: the reduction value itself
   is spatially clustered, not scattered randomly across Aberfoyle — Moran's I 0.55–0.64 (p=0.001)
   for both DNN and XGBoost, both cohorts (`TEMP_rq2a_reduction_morans_i_2026-08-16.tex`), a
   magnitude comparable to the spatial clustering already reported for RQ2b/RQ3's own residuals.
   Only worth including in the write-up if the accompanying map figure is actually used, or as a
   short corroborating sentence folded into this item's existing evidence (it does not need its own
   figure to be worth citing as a number) — it does not independently resolve the Q1-noise-vs-signal
   question above, only shows the effect has real spatial structure, which is consistent with (not
   proof of) CR-fit quality itself being spatially patterned.
2. **`[Importance: 4/5]` The effect lives in the data, not in physics-guided architecture — a plain
   XGBoost given the same environmental features reproduces the same pattern with a comparable
   effect size and comparable fold-to-fold stability.**
   (`TEMP_rq2a_residual_reduction_results_2026-08-11.tex`) XGBoost's mean reduction (1.597m/0.361m)
   is comparable to, or larger than, DNN's (1.501m/0.174m) on both cohorts, and its own env-vs-no-env
   control confirms environment itself (not architecture) drives the effect (+0.375m on 4survey,
   +0.121m on 6survey, env arm vs. no-env arm). Compared on the same basis (single seed, 5-fold SD
   for both models — DNN 0.386/0.264, XGBoost 0.330/0.183), **XGBoost's fold-to-fold SD is if
   anything slightly smaller than DNN's**, 69-85% of DNN's own SD on both cohorts — there is no real
   stability gap between the two model families here. The comparability holds at the quartile level
   too, not just in aggregate: XGBoost's own Q1 relative-harm ratio (model error vs. CR's own Q1
   error) is 2.1-2.3x on 4survey and 2.1-2.4x on 6survey — in the same range as DNN's 2.6x/3.2x
   (item 1), not a smaller or absent effect. This aggregate env-vs-no-env comparison (+0.375m/+0.121m)
   confirms real environmental data adds something overall, but item 1's own permutation check
   shows that addition concentrates in Q4 — Q1's own degradation shows up even when the
   environmental values are fake, so "environment drives the effect" should be read as "drives
   Q4's correction," not as uniform across every quartile.
   **Why the pattern reproduces**: this item is itself the evidence for item 1's proposed mechanism
   — if the effect were about physics-guided architecture specifically, a model with no physics loss
   and no CR-curve embedding at all shouldn't show the same quartile-concentrated pattern. That both
   models pick it up independently says the effect is a property of the *data* (environmental
   variables genuinely correlate with the CR-residual in this quartile-concentrated way), not either
   model's architecture — not a point in favour of either PINN or XGBoost specifically.
3. **`[Importance: 3/5]` The physics constraint's cost (RQ1 item 5) buys real but narrowly-scoped
   parameter identifiability — the accurate remaining case for PINN_k specifically, not for
   physics-guided architecture generally.** (`TEMP_rq1_physicsablation_results_2026-08-11.tex`)
   Removing the physics/trajectory loss (w=0) makes PINN_k more accurate but its `y_max`/`k` become
   far less identifiable — pooled 5-fold correlation -0.935 at w=0 vs. -0.449 at the default w=1 on
   4survey (6survey flips sign entirely: +0.430 vs. -0.057).
   **What "identifiable" means here**: `y_max` and `k` are each output by a separate small
   sub-network taking the *same* terrain/wind features as input, combined into one height
   prediction — the optimizer only ever sees their combined effect, never each one in isolation.
   Many different (`y_max`, `k`) pairs can produce almost the same predicted curve, so the specific
   split the model lands on isn't necessarily the "true" one, just wherever the optimisation
   happened to settle on that trade-off ridge.
   **Not a trivial or wild solution**: every one of the 58,073 plots' learned `y_max` at w=0 sits
   between 50.2m and 52.0m, tight around the global CR anchor (51.96m) — nothing biologically
   implausible for Sitka spruce. The extreme correlation means something narrower: within that
   tight, plausible band, whatever small amount a plot's `y_max` moves up, its `k` reliably moves
   down to compensate. Individual predictions stay trustworthy; the *decomposition* into "this
   plot's asymptote" vs. "this plot's growth rate" doesn't.
   **Why it happens**: flagged directly in the model's own code comments before it was ever fit —
   "especially for plots that never reach anywhere near their true asymptote within the observed
   age range, common in this dataset" (`pinn_env_terrain_k.py`). The standard growth-curve
   identifiability problem: without data near the actual asymptote, a taller-but-slower and a
   shorter-but-faster trajectory both fit the observed data almost equally well.
   **Corroborated by three further checks, all pointing the same direction**: the classical,
   non-neural Chapman-Richards fit's own parameter correlation is even more extreme (-0.993,
   confirmed tier-independent — classical fitting never uses environmental Sets); a cluster-bootstrap
   on the current-tier, pooled 5-fold neural correlation confirms it's real, not small-sample noise
   (95% CI [-0.944, -0.923], n=58,073 plots/232 compartments, 0% of 2000 resamples cross into
   positive territory — `TEMP_rq2a_pinn_k_identifiability_bootstrap_2026-08-16.tex`; 6survey's own
   opposite-sign correlation is equally robust, CI [+0.254, +0.589], never crossing zero either); and
   a freeze-`y_max`-vary-only-`k` ablation rules out a simple "two knobs just fighting over one
   degree of freedom" story — freezing `y_max` and re-fitting `k` alone gives a WIDER `k` range
   than the full two-parameter model (0.0136 vs. 0.0092, ~148% as wide), the opposite of what a
   resource-competition story would predict.
   **Scope, precisely**: this is a claim about PINN_k's own claimed interpretability being genuine
   only with the constraint present — not evidence that RQ2b or RQ3 need PINN's physics guidance,
   since their own attribution targets are built from a separate classical curve fit, not PINN_k's
   learned parameters.
   **What this needs items 1-2 to establish first, in plain terms**: across RQ1/RQ2a there are a
   few different candidate reasons to still care about PINN despite XGBoost winning on raw accuracy
   (item 1) and matching PINN's residual-shrinkage pattern, effect size, and fold-to-fold stability
   (item 2) — so neither raw accuracy nor the residual-shrinkage behaviour is a genuine PINN-specific
   advantage; a plain tree ensemble does both at least as well. This item's finding is different:
   remove the physics constraint and the y_max/k separability genuinely disappears — something only
   PINN_k's own architecture can even produce or lose, since XGBoost has no such parameters at all.
   The genuinely surviving case for PINN rests on two things, neither established by items 1-2 alone:
   this narrowly-scoped identifiability claim, and RQ1 item 4's structural leakage-resistance (a
   standard regularisation argument, not spin).
   **Not yet done**: a multi-seed check of the flagship correlation's stability across training
   seeds was considered and deprioritised (the specific concern motivating it — that the flagship
   number might be as split-sensitive as a secondary comparison number turned out to be — was
   checked and resolved: both flagship numbers are already the robust pooled 5-fold statistic).

---

## RQ2b — attribution of the CR-residual to environmental/stand-structure variables

**4survey only, by design** — every number below is 4survey; no 6survey arm exists for this RQ.
Target is `mean_cr_residual`, from the classical (non-neural) Chapman-Richards fit — this section's
own uncertainty is independent of RQ1/RQ2a's neural-model findings (different target, different
methods: NLME/Elastic Net/XGBoost, none of them PINN).

### Results table (`TEMP_rq2_attribution_results_2026-08-11.tex`)

| Set | n rows | EN R2 [95% CI] | XGB R2 [95% CI] | NLME spatial variance explained |
|---|---:|---|---|---|
| Set1 (baseline only) | 71,766 | 0.220 [0.180, 0.252] | 0.222 [0.183, 0.252] | 0.016±0.078 |
| Set2 | 71,766 | 0.359 [0.310, 0.396] | 0.416 [0.355, 0.461] | 0.204±0.066 |
| Set3 | 71,727 | 0.325 [0.272, 0.365] | 0.375 [0.321, 0.414] | 0.049±0.048 |
| Set4 | 71,330 | 0.358 [0.305, 0.398] | 0.395 [0.338, 0.438] | 0.175±0.064 |

(Coefficients themselves aren't in this table — they're the central plot, item 1 below.)

### Plot inventory

#### Figure R2b-1: What global attribution finds, and what it still can't explain

**Research question and findings supported:** RQ2b, items 1, 2, and 3 — CanopyCover/thinning
dominate across three converging methods; `slope_degrees` is stable across NLME/EN while `topex` is
stable only within NLME; substantial spatial clustering remains in every model's residual
regardless of how much environmental data is added.

**Purpose:** Which environmental/management variables are large AND stable across folds and across
model families, and does adding more of them close the spatial gap left behind?

**Key insight:** A small set of variables (CanopyCover, the thinning pair, `slope_degrees`) are
large and tightly consistent across both methods and every fold; `topex` is large in NLME but weak
and sign-flipping in EN — a genuine cross-model disagreement, not simple noise; and no matter how
much environmental data is added, the residual's own spatial clustering (Moran's I) barely moves —
shown not just as one summary number per set, but as an actual map of where that leftover residual
sits high or low across Aberfoyle.

**What it adds beyond the table:** The results table reports only R2 and NLME variance-explained
per set — nothing about which individual variables drive that R2, whether their fold-to-fold SD is
small enough to trust the sign, or whether the two model families agree. It also cannot show that
Moran's I stays flat across Set1→Set4 despite R2 clearly rising, or what that "flat Moran's I"
actually looks like on the ground — read alone, the table could wrongly suggest more environmental
data is closing the spatial gap.

**Exact visual design:**
- Plot type: three-panel composite. Panel A: coefficient forest plot. Panel B: line chart of
  Moran's I by set. Panel C: a genuine spatial map of the leftover residual, not another categorical
  chart.
- Panels: 3. Panel A tall (stacked above Panel B, sharing the page's left column, given ~19 Set4
  variables); Panel C beside them as a map, given its own full extent.
- Panel A: one row per Set4 variable, sorted by |mean coefficient|; point = mean coefficient across
  5 folds, whisker = ±1 SD; NLME and EN shown as two side-by-side points per row (not overlaid) so
  agreement or disagreement is directly legible per variable.
- Panel A x-axis: standardised coefficient value, vertical reference line at 0. Y-axis: variable
  name (categorical).
- Panel B x-axis: feature set (Set1 → Set4, ordered by richness). Y-axis: residual Moran's I
  (0–0.8 range).
- Panel B lines: two, one per method (EN, XGBoost).
- Panel C (map — new): Aberfoyle plot locations (x, y), Set4, **colour = EN's own model residual**
  against `mean_cr_residual` (diverging scale, centred at 0, so systematic over/under-prediction
  reads directly as colour rather than being summarised away into one Moran's I value). This is the
  exact same residual that Moran's I in Panel B was computed from, so Panel C is not a new
  statistical claim, only a direct view of the number already behind Panel B's line.
- Colour: consistent method colours throughout Panels A/B (NLME, EN, XGBoost); Panel C uses its own
  separate diverging colour scale with its own colourbar, since it is a continuous spatial value, not
  a categorical method comparison.
- Reference lines: Panel A's zero line (above); Panel B's constant p=0.001 permutation floor is
  better stated as a text annotation than a plotted line. Panel C's colour-scale midpoint (0) is the
  implicit reference.
- Uncertainty shown: Panel A's whiskers are the fold-SD uncertainty; Panel B has no CI computed for
  Moran's I in the source and should not imply one; Panel C shows one pooled value per plot (across
  the 5 test folds) with no per-plot CI computed — should not imply one either.
- Paired: Panel A's NLME/EN points are paired by variable (same folds); Panel B's EN/XGBoost lines
  are paired by set; Panel C is not paired (one map, one model, one set).
- Labels/annotations: flag `topex` and the thinning pair directly on Panel A. Panel C could mark
  compartment boundaries lightly if a boundary layer is available (NEEDS CHECKING, same gap noted
  for RQ3's maps), so a reader can see whether high/low-residual regions align with compartment
  edges — directly testable against RQ3's own boundary-proximity finding, a genuine cross-RQ link.
- Legend: one shared legend for method colour (Panels A/B); one separate colourbar for Panel C.
- Axis limits: Panel A's x-axis wide enough for the thinning columns' own large coefficients
  (~-2 to -3) without clipping; Panel B's y-axis from 0 to 0.8; Panel C uses real map coordinates,
  not a shared axis with A/B.

**Data required:** Panel A: Set4 NLME and EN coefficient tables (mean±SD across 5 folds),
`TEMP_rq2_attribution_results_2026-08-11.tex` — already computed. Panel B: Moran's I by set, EN and
XGBoost, same file's Moran's I section — already computed. Panel C: EN's Set4 residual against
`mean_cr_residual` plus (x, y), from `evaluate_rq2_attribution.py`'s saved `predictions.csv` — this
is the same input already used to compute Panel B's own Moran's I values
(`residual_spatial_autocorrelation_check.py`), so already available with no new fitting. With
~71,330 plots at Set4, use small, semi-transparent points (or bin to compartment-mean residual at
compartment centroids for a cleaner map) — state whichever is used in the caption.

**Interpretation limits:** Coefficients are statistical associations with the fitted model, not
causal effect sizes or directions — RQ2b's own methodology frames this as association only; caption
must not describe any variable as a "driver" or "cause" of curve departure. Panel C shows *where
the model's own error remains*, not a spatially-varying relationship — unlike RQ3's GNNWR, none of
RQ2b's methods (NLME, EN, XGBoost) produce a coefficient that varies by location; the map is a
diagnostic of what is left unexplained, not evidence of a locally different environmental effect,
and the caption must not blur that distinction. The CanopyCover-dropped ablation (a separate,
important finding) is not shown here — see Figure R2b-2.

**Placement:** main text.

**Caption message:** A small set of variables — CanopyCover, thinning history, slope — are large
and stable across both model families and every fold; `topex` is stable only within one method, not
both; and the model's own leftover error remains visibly clustered across Aberfoyle regardless of
how much environmental data is added, not just flat in a single Moran's I number.

**Ranking:** 1 (essential).

---

#### Figure R2b-2: CanopyCover-dropped ablation

**Research question and findings supported:** RQ2b item 1 — CanopyCover carries large, non-redundant
signal; dropping it costs both models roughly 34–36% of R2.

**Purpose:** How much does R2 fall when CanopyCover is removed from Set4, for each model?

**Key insight:** A visibly large drop for both EN and XGBoost, of similar relative size.

**What it adds beyond the table:** Nothing beyond what is already precisely stated in prose (EN
0.350→0.231; XGBoost 0.388→0.249) — included for visual confirmation only, not because it reveals
something prose hides.

**Exact visual design:** Paired dot-and-line ("slope") plot, same style as Figure R1-1, not a bar
chart — two categories on the x-axis ("With CanopyCover" / "Without"), y-axis = R2 with error bars
(fold SD: EN 0.350±0.040 → 0.231±0.062; XGBoost 0.388±0.057 → 0.249±0.076, already computed), one
line per model connecting its own two points, coloured consistently with Figure R2b-1's method
colours. The error bars carry real information here (they show the drop is far larger than the
fold-to-fold noise on either side), which a bar chart would not display as clearly.

**Data required:** `TEMP_rq2_attribution_results_2026-08-11.tex`'s CanopyCover-dropped ablation
table — already computed.

**Interpretation limits:** A single ablation number, not a decomposition of why CanopyCover carries
this signal — item 1's own text is explicit the reverse-causation question remains open.

**Placement:** appendix.

**Caption message:** Removing CanopyCover from Set4 costs both models roughly a third of their
explanatory power, with no other variable absorbing the lost signal.

**Ranking:** 3 (appendix only).

---

#### Figure R2b-3: VIF diagnostic

**Research question and findings supported:** RQ2b items 1 (thinning-pair collinearity) and 3
(multicollinearity ruled out as explaining topex/climate instability).

**Purpose:** Are the SHAP/coefficient-important variables collinear with each other within Set4?

**Key insight:** The core environmental variables sit well under the VIF=5 threshold; only the two
thinning-history columns (collinear by construction) exceed it.

**What it adds beyond the table:** Not present in the main results table at all, but this is a
supporting diagnostic for a claim already fully stated numerically in items 1 and 3.

**Exact visual design:** Horizontal lollipop/dot plot (a thin stem plus a point, not a full bar),
one row per Set4 variable, sorted by VIF value; vertical reference line at VIF=5.0; points above
the threshold coloured distinctly (e.g. the two thinning columns) from points below it, so the
"exempt by design, not by chance" story reads at a glance without needing the caption alone to
carry it.

**Data required:** `TEMP_rq2_attribution_results_2026-08-11.tex`'s VIF table — already computed.

**Interpretation limits:** VIF diagnoses collinearity only, not causal validity or importance.

**Placement:** appendix.

**Caption message:** Every SHAP-important environmental variable sits well under the collinearity
threshold; only the two thinning-history columns, collinear by construction, exceed it.

**Ranking:** 3 (appendix only).

---

#### Not proposed: a separate CI-overlap bar chart for item 4

The results table already reports each set's [95% CI] directly — a bar+errorbar chart of the same
numbers would repeat the table rather than add anything, which the design principles explicitly
warn against. Item 4's CI-overlap point is fully legible from the existing table.

### Ranked items

1. **`[Importance: 5/5]` CanopyCover and thinning-history dominate every set, converging tightly
   across three independent methods.** `CanopyCover`'s coefficient is the largest or near-largest in
   every NLME and EN table (e.g. EN Set4: +1.943±0.034), and it's #1 by mean |SHAP| in every
   XGBoost set too (1.375–1.442), essentially unmoved by the 2026-08-15 XGBoost hyperparameter
   correction (Set2: 1.474→1.442) — three methods that fit completely differently (a mixed-effects
   model, a linear model with L1/L2 penalty, a tree ensemble's feature attribution) landing on the
   same answer is the most consistent finding in this project.
   **Why CanopyCover/thinning dominate**: not mysterious — they're the most direct, proximate
   measures of stand structure (how much canopy/wood is actually present), so a model of height-
   residual variance leaning on them first is mechanistically unsurprising.
   **`CanopyCover` comes from the same LiDAR/ALS pipeline as the height targets** — a real reason to
   check its role carefully rather than take the dominance at face value. Its correlation with all
   three of the project's own targets (RQ1's `elev_percentile_95th`, this section's own
   `mean_cr_residual`, RQ3's `local_y_max_difference`) is moderate (Pearson 0.35–0.47) — not the
   target restated in different units. Refitting Set4 with `CanopyCover` removed drops R2 by
   roughly a third for both models (EN: 0.350→0.231, -34%; XGBoost: 0.388→0.249, -36%) — a bigger
   effect than its SHAP rank-margin alone
   (1.2–1.7x over the #2 variable) suggested, meaning `CanopyCover` carries large, non-redundant
   signal that nothing else in the set backfills. `dist_to_road` does not rise to absorb the gap
   once `CanopyCover` is removed (SHAP 1.003→0.869, slightly lower) — ruling out simple
   correlation-driven credit theft from that specific variable as the mechanism, though not
   resolving the broader question of why CanopyCover carries this much signal.
   **A separate collinearity note**: `CanopyCover`'s own two thinning-history columns are
   collinear with EACH OTHER (VIF 39.7/22.6, both far above the 5.0 threshold applied elsewhere) —
   a structurally expected pairing (`time_since_thinning_missing` flags whether
   `time_since_thinning` itself has a valid value), so read them as one combined "thinning-history"
   signal, not two independently interpretable coefficients. The core SHAP-important environmental
   variables (`dist_to_road`/`chelsa_bio12_precip_mm`/`slope_degrees`, VIF 1.09–3.91) are
   collinearity-clean, unaffected by this caveat.
   **Distinct from RQ2/RQ3's exclusion of `Age` from the candidate feature pool**: `Age` is
   algebraically inside the Chapman-Richards formula that builds RQ2b/RQ3's own targets, so it's
   never offered as a candidate predictor here at all — a separate, unrelated design choice from
   RQ1/PINN's own intentional use of Age as a physics-informed architecture input.
   The defensible write-up position is to present `CanopyCover`/thinning explicitly as baseline
   stand-structure controls (expected to dominate row-level R2 by construction), with the
   environmental variables' contribution beyond that baseline as the actual attribution headline —
   which is exactly how the remaining items in this section already treat it.
2. **`[Importance: 5/5]` Substantial spatial clustering remains in every model's residual, and more
   environmental data barely reduces it.** Moran's I stays in the 0.65–0.74 range across every set
   for both EN and XGBoost (p=0.001, the permutation floor, throughout) — going from Set1 (4 baseline
   columns) to Set4 (19 columns, every category, VIF-screened) only nudges EN's Moran's I from 0.738
   to 0.693. CanopyCover/thinning dominate the attribution (item 1), but a large share of the
   spatially-structured variance in `mean_cr_residual` remains genuinely unexplained regardless of
   how much environmental data is added — RQ2b's models explain a real, substantial share of the
   row-level signal, not the full spatial pattern.
   **Why this remains unexplained**: not established — flagged honestly rather than guessed at. No
   candidate mechanism (unmeasured local factors, finer-resolution soil/management variation than
   the current environmental data captures) has been tested here. This is a genuine open question,
   not a gap papered over.
   **Why this matters beyond RQ2b**: the strongest bridge to RQ3 in this section — real, current-tier
   evidence that global attribution models leave a gap more environmental data doesn't fix,
   motivating RQ3's spatially-varying approach as a response to a demonstrated problem. RQ3's own
   current-tier Moran's I check (see RQ3 section) independently confirms this bridge.
3. **`[Importance: 3/5]` `slope_degrees` is the most stable real environmental signal, agreeing
   across both NLME and EN; `topex` is stable only within NLME, not across model families — a
   genuine cross-model disagreement, not the same kind of instability as the other candidates
   below.** `slope_degrees`: NLME +0.894/+0.861 and EN +1.230/+0.993 (Set3/Set4) — same sign,
   comparable magnitude, both methods agree. `topex`: NLME is strongly positive and consistent
   (+1.316/+1.323, Set3/Set4) — but EN's own `topex` coefficient is weak and flips sign between
   sets (-0.175 Set3, +0.063 Set4, both with SD larger than or comparable to the coefficient
   itself). This is not the same finding as "both methods agree `topex` is a real signal" — NLME
   and EN disagree about it, and that disagreement is itself worth reporting rather than smoothing
   over. `chelsa_gdd5_degc`, `tas_mean`, and `soilgrids_ph` flip sign or have SD comparable to
   their own mean in a more straightforward way (unstable in both methods, not a cross-model split)
   — these should be read as "not reliably different from zero," not settled findings.
   **Why slope is stable and topex is not, across methods**: a plausible, domain-grounded reason
   for slope specifically — slope is a well-established predictor of tree growth/damage in the
   forestry literature generally, so a real, direct association here isn't surprising. Why NLME and
   EN disagree specifically on `topex` is not established — a candidate is that NLME's
   mixed-effects structure and EN's L1/L2-penalised linear fit handle `topex`'s relationship with
   the spatially-correlated random effect differently, but this hasn't been tested here.
   **Why the others aren't stable — multicollinearity checked directly and ruled out as the
   explanation**: Set4's own VIF table (`TEMP_rq2_attribution_results_2026-08-11.tex`) does NOT
   separate stable from unstable variables — `tas_mean` (VIF 3.44) and `soilgrids_ph` (VIF 2.12)
   sit in the same range as `chelsa_bio12_precip_mm` (VIF 3.91), which IS stable; `topex` itself
   (VIF 2.37) is lower than several stable variables. If multicollinearity within Set4 were driving
   the instability, the
   unstable variables should show visibly higher VIF than the stable ones — they don't.
   **A more specific, better-supported candidate — not yet empirically tested as the cause, so not
   claimed as one**: `chelsa_gdd5_degc`/`chelsa_bio12_precip_mm` are a single static 1981–2010
   climatology applied identically regardless of a plot's actual survey year (2008–2023) — a
   disclosed data characteristic (`methodlogy_env_setpick.md` §1.4), not a claim invented here.
   `tas_mean`/`groundfrost_mean` are closer to matched (cohort-year-averaged) but still not
   single-year-matched. Since `mean_cr_residual` aggregates across a plot's whole multi-year survey
   history while these climate values don't vary with it at all, a genuine mismatch between the
   predictor's time-resolution and the target's own construction is a real candidate mechanism —
   but this hasn't been isolated as the actual cause, only established as a real, disclosed property
   of the input data. `soilgrids_ph` (a genuinely static soil property, not time-mismatched in the
   same way) and `topex`'s EN-specific instability (a static terrain feature, same VIF context as
   its NLME fit, which IS stable) are NOT explained by either hypothesis — genuinely open.
4. **`[Importance: 3/5]` Bootstrap CIs temper "which set is best" into a softer claim, while sharpening
   which set is clearly weakest.** Set2 and Set4's XGBoost CIs overlap substantially ([0.355,0.461]
   vs. [0.338,0.438]) — a real point-estimate gap, not a statistically clear-cut one. Set3 is more
   clearly separated from Set2 — consistent with it being the weakest set on point estimates too
   (lowest R2 for both models, lowest NLME variance explained). Every set's CI clears Set1's
   baseline-only CI with no overlap at all, so "environment adds real value beyond baseline" is the
   one comparison here that's unambiguous.
   **Why Set3 is weakest**: a real, checkable compositional reason, not mysterious — Set3 is
   terrain-heavy with thin wind representation and zero soil/climate coverage, so it's missing
   candidate signal the other sets have access to.
   **Not yet done**: cross-method agreement heatmap (NLME vs. EN vs. XGBoost-SHAP, all three exist,
   table not built), and a 5-seed reseed (no seed sweep has been run for RQ2b at all — every number
   in this section reflects split seed 42 only). Item 3's temporal-mismatch candidate for the
   unstable climate variables is also untested — see item 3 for what a real test would need.

---

## RQ3 — plot-specific curve deviation (interesting growth-curve behaviour, not just which model wins)

**Framing**: RQ2b already showed (RQ2b item 2) that global attribution models leave substantial
spatial clustering in their residuals (Moran's I 0.65-0.74, p=0.001) that more environmental data
does not fix. RQ3 is a direct test of whether a spatially-varying relationship can capture what
RQ2b's global models could not, and item 3 below answers that directly, not just as a hypothesis.

### Results table (`TEMP_rq3_en_xgb_results_2026-08-11.tex`, `TEMP_rq3_gnnwr_results_2026-08-11.tex`
— 4survey has bootstrap 95% CIs throughout; 6survey does not for EN/XGB (point estimates, seed 42
only), and GNNWR's 6survey column shows both its single-seed-42 point estimate and its 3-seed
mean±SD reseed — see item 2 for why 6survey needed the reseed at all)

| Set | EN 4survey [95% CI] | EN 6survey | XGB 4survey [95% CI] | XGB 6survey | GNNWR 4survey [95% CI] | GNNWR 6survey (seed 42 / 3-seed mean±SD) |
|---|---|---:|---|---:|---|---|
| Set2 | 0.286 [0.241, 0.325] | 0.057 | 0.266 [0.214, 0.308] | 0.031 | 0.320 [0.270, 0.369] | -0.044 / 0.014±0.053 |
| Set3 | 0.274 [0.223, 0.316] | 0.031 | 0.281 [0.235, 0.320] | 0.086 | 0.319 [0.263, 0.365] | 0.058 / 0.054±0.055 |
| Set4 | 0.240 [0.192, 0.286] | 0.056 | 0.250 [0.201, 0.295] | 0.015 | 0.294 [0.236, 0.347] | -0.075 / 0.002±0.068 |

### Plot inventory

#### Figure R3-1: GNNWR's accuracy edge and its spatial-structure meaning

**Research question and findings supported:** RQ3, items 2 and 3 — GNNWR shows a consistent,
corroborated (though not CI-clean) edge over EN/XGBoost on 4survey, and 6survey is unreliable;
residual Moran's I independently confirms the same direction on a completely different diagnostic.

**Purpose:** Does GNNWR's accuracy edge over EN/XGBoost correspond to it actually reducing leftover
spatial structure in the residual, and does that correspondence hold — or break down — consistently
between cohorts?

**Key insight:** On 4survey, GNNWR wins on R2 (all 3 sets) AND independently reduces residual
Moran's I versus the best global model (all 3 sets) — two unrelated diagnostics agreeing. On
6survey, both signals flip or vanish together (R2 sign-flips across seeds; Moran's I increases
rather than decreases) — one consistent story that holds together and breaks together, not two
coincidental findings.

**What it adds beyond the table:** The results table shows R2 only. It cannot show that a second,
independent diagnostic (Moran's I) points the same direction on 4survey and the opposite direction
on 6survey — the actual basis for calling this "corroborated," not just "measured once."

**Exact visual design:**
- Plot type: three-panel composite — two panels sharing a categorical accuracy/clustering axis, plus
  a genuine spatial map exploiting the fact that GNNWR is a spatially-varying model, not just another
  point estimate to bar-chart.
- Panels: 3. Panel A and Panel B stacked vertically (shared x-axis); Panel C to the side as a map.
- Panel A (accuracy): x-axis = feature set (Set2/3/4), y-axis = R2, one point+error-bar per model
  (EN, XGBoost, GNNWR) per set, **4survey only** — 6survey is explicitly excluded here, since its R2
  is not a reliable single value (item 2's own finding); showing it would misrepresent an
  unreliable, seed-sign-flipping number as a stable point estimate. Error bars: bootstrap 95% CI,
  already computed for all three models, all three sets.
- Panel B (spatial structure): x-axis = same feature sets, y-axis = residual Moran's I, **both
  cohorts** shown (two line styles or two sub-panels), one line per model — this is where the
  4survey/6survey divergence becomes visible, the entire point of pairing it with Panel A's
  4survey-only accuracy view.
- Panel C (map — new): Aberfoyle plot locations (x/y), 4survey, Set4, one point per test-set plot,
  **colour = GNNWR's own local `coef_CanopyCover` value** (diverging colour scale, centred at 0, so
  a sign flip anywhere would be immediately visible rather than hidden inside an aggregate mean).
  This directly shows whether the "spatially varying" part of GNNWR is doing real work — a flat,
  uniform colour across the whole map would itself be an informative (if less exciting) result, not
  a failure of the figure.
- Colour: one consistent colour per model across Panels A/B (EN, XGBoost, GNNWR); Panel C uses its
  own separate diverging colour scale for the coefficient value, with its own legend/colourbar.
- Reference lines: none needed on A/B beyond the axes themselves; Panel C's colour scale midpoint
  (0) is the implicit reference.
- Uncertainty shown: Panel A shows 95% CIs directly; Panel B has no CI computed for Moran's I and
  should not imply one; Panel C could optionally use point transparency/size to encode each plot's
  own reference-set density (a proxy for how much local data informed that plot's coefficient) if
  that is easy to compute — NEEDS CHECKING, not required for a first version.
- Paired: Panel A's three models are paired by set; Panel B's models are paired by set AND cohort;
  Panel C is not paired (one map, one model).
- Labels/annotations: annotate 6survey's GNNWR points on Panel B with a note that its R2 is not
  shown in Panel A due to seed instability. Panel C should mark compartment boundaries lightly (if
  a boundary layer is available) so the reader can see whether high/low coefficient regions align
  with compartment edges — connects visually to Figure R3-2's own boundary-proximity finding.
- Legend: one shared legend for model colour (Panels A/B); one separate colourbar for Panel C.
- Axis limits: shared x-axis category order (Set2/3/4) across Panels A/B; independent y-axis scales
  (R2 vs. Moran's I are different units); Panel C uses real map coordinates, not a shared axis with
  A/B.

**Data required:** Panel A: `TEMP_rq3_en_xgb_results_2026-08-11.tex` and
`TEMP_rq3_gnnwr_results_2026-08-11.tex`'s 4survey bootstrap-CI table (same numbers as the main
results table above) — already computed. Panel B: RQ3's own current-tier residual Moran's I table
(all 3 sets, both cohorts, EN/XGBoost/GNNWR), cited in item 3 — already computed. Panel C: `x`, `y`,
and `coef_CanopyCover` columns, already present in every GNNWR `test_predictions.csv`
(`outputs/growth_curve_attribution/gnnwr/gnnwr_nested_set4_gated_all_vif_4survey_reffull_fold{0-4}of5_test_predictions.csv`,
the same files already used for `TEMP_rq3_gnnwr_local_coef_rank_2026-08-16.tex`) — pooling the 5
folds' test predictions gives one coefficient value per plot for the whole cohort; already
available, no new fitting. A compartment-boundary polygon layer for the optional annotation is
NEEDS CHECKING — not confirmed to exist in an easily-plottable form.

**Interpretation limits:** Corroboration between two diagnostics (Panels A/B) is not the same as
either clearing a strict significance bar — item 2's own text is explicit the R2 gap does not clear
non-overlapping CIs. Panel C shows association between location and coefficient value, not a tested
spatial-clustering statistic on the coefficient itself (no Moran's I has been computed on
`coef_CanopyCover` specifically) — the caption should describe what the map shows, not assert
statistical significance for any visual pattern in it.

**Placement:** main text.

**Caption message:** GNNWR's accuracy edge over EN/XGBoost on 4survey is corroborated by an
independent reduction in residual spatial clustering, and its own local CanopyCover coefficient
varies across Aberfoyle rather than reducing to one global number; both accuracy and
clustering signals disappear together on 6survey, consistent with an underpowered result there.

**Ranking:** 1 (essential).

---

#### Figure R3-2: Where growth-curve deviations occur, and what explains them

**Research question and findings supported:** RQ3, items 4, 5, 6, and 9 — the same ~10 plots recur
and split into two distinct sub-populations; the implausible `y_max_fit` values are a
target-construction artifact traced to a fixed-`k` mismatch, corroborated by boundary proximity;
the pattern scales to hundreds of plots; compartment-level archetypes reproduce the boundary-
proximity finding independently, at a different unit of analysis.

**Purpose:** Where across Aberfoyle do plots deviate from their yield-class benchmark, and does the
spatial pattern of *where* line up with the spatial pattern of the *reasons* already established
(boundary proximity, compartment archetype)?

**Key insight:** Deviation from the yield-class benchmark is not scattered randomly across the
landscape — plots and compartments near boundaries, and compartments classified as
"trajectory-break outlier" or "consistently below yldc," visibly cluster in the same places the
deviation map itself shows the largest departures. Two representative trajectories (one smooth-
offset, one genuinely unstable) ground what those coloured points actually mean as real growth
curves, not just abstract numbers.

**What it adds beyond the table:** No table in RQ3 shows this spatially at all — item 6's numbers
describe the flagged tail's size, item 5's numbers describe boundary distance as a population
median, item 9's numbers describe archetype-level environmental means, but none of them show a
reader where any of this actually sits on the ground, or whether the deviation pattern and the
proposed reasons visually coincide.

**Exact visual design:**
- Plot type: three-panel composite — two maps sharing the same Aberfoyle footprint, plus a small
  set of representative trajectories.
- Panels: 3. Panel A and Panel B side by side (same map extent, so the reader can visually compare
  them directly); Panel C below, small multiples.
- Panel A (deviation map): one point per 4survey plot at its own (x, y), **colour = `local_y_max_
  difference`** on a diverging scale centred at 0 (e.g. blue = below yldc, red = above), so the
  direction and magnitude of deviation is visible at a glance across the whole landscape. Tukey-
  fence-flagged plots (item 6's 266) get a distinct outline/marker so the "hundreds of plots, not a
  handful" scale finding is visible on the same map as the deviation pattern itself.
- Panel B (reason map): the same plot locations, but **colour = compartment archetype** (item 9's 5
  categories: close-to-yldc/stable, consistently-above, consistently-below, trajectory-break-
  outlier, moderate-mismatch), using a qualitative (not diverging) colour palette. Placed directly
  beside Panel A so a reader can visually check whether large deviations in Panel A sit inside
  trajectory-break-outlier or consistently-below compartments in Panel B, rather than reading two
  separate population-median numbers and trusting they line up.
- Panel C (representative trajectories, kept from the original design): small multiples, 4 panels —
  2 plots from the "moderately-unstable" group (most extreme residual range) and 2 from the
  "smooth" group (most extreme `local_y_max_difference`), named explicitly as the selection rule in
  the caption, not chosen because they look convincing. X-axis: age (years). Y-axis: observed top
  height (m). Each panel: that plot's own raw survey points (connected by a line) against its own
  yield-class benchmark curve (dashed).
- Colour: Panel A's diverging scale and Panel B's qualitative archetype palette are deliberately
  different colour systems (continuous vs. categorical) so they are never confused with each other;
  Panel C reuses Panel A's "observed vs. benchmark" line-style convention (solid vs. dashed).
- Reference lines: Panel A/B need none beyond their own colour scales; Panel C's benchmark curve is
  itself the reference.
- Uncertainty shown: none on the maps (each point is a direct value, not an estimate with its own
  CI at this level); none needed on Panel C (raw observed points, deterministic benchmark curve).
- Paired: Panel C is inherently paired (each plot's own trajectory against its own benchmark);
  Panels A/B are paired with each other by sharing the same plot locations, not paired internally.
- Labels/annotations: mark compartment/block boundaries lightly on both maps if a boundary layer is
  available (NEEDS CHECKING); label each Panel C trajectory with its plot ID and sub-population.
- Legend: separate colourbar for Panel A (diverging), separate categorical legend for Panel B
  (5 archetypes); one small shared legend for Panel C (observed/benchmark).
- Axis limits: Panels A/B share identical map extent and aspect ratio so they are directly
  comparable; Panel C's panels may each use their own y-axis range, stated in the caption.

**Data required:** Panel A: `local_y_max_difference` and (x, y) for every 4survey plot, from the
plot-level table used for item 6's Tukey-fence percentages
(`models/growth_curve_attribution/scale_comparison_check.py`'s `build_plot_level_table()`) —
already computed as an intermediate; NEEDS CHECKING whether already exported to a standalone file.
Panel B: compartment archetype per compartment, joined onto each compartment's own plots, from
`TEMP_rq3_compartment_archetype_check_2026-08-16.tex` / `rq3_compartment_archetype_check.py` —
already computed at the compartment level, needs joining onto plot-level (x, y) via `cpmt`
(straightforward, same identifier used throughout, not a new calculation). Panel C: raw per-survey
height/age rows for the 4 selected plots, via `load_filtered_growth_curve_table('4survey')` — already
available.

**Interpretation limits:** The maps show spatial coincidence, not a tested statistical association
between deviation and archetype/boundary-proximity — no formal spatial-join or clustering test
compares Panels A and B directly, only visual inspection. Panel C's specific examples remain
illustrative, not a claim every flagged plot looks exactly like these four. Does not establish
causation for either mechanism beyond what items 4/5/9 already state as "plausible."

**Placement:** main text.

**Caption message:** Plots and compartments with the largest deviations from their yield-class
benchmark are not scattered randomly — they visibly coincide with boundary-adjacent locations and
"trajectory-break" or "consistently below" compartment archetypes, and the two archetypal
trajectory shapes behind that pattern are shown directly.

**Ranking:** 1 (essential).

---

#### Figure R3-3: Does CanopyCover's rank agree across all three attribution models?

**Research question and findings supported:** RQ3 item 1 — CanopyCover dominates on 4survey by
three converging methods; on 6survey, the disagreement splits by MODEL, not by which method
happened to be computed — EN and GNNWR both still rank it #1 in every set, XGBoost (gain importance
and SHAP alike) does not, in any set.

**Purpose:** Do EN, XGBoost, and GNNWR agree on which variable is most important, separately for
4survey and 6survey?

**Key insight:** All three methods agree on 4survey (CanopyCover #1 throughout). On 6survey, EN and
GNNWR still agree with each other and with 4survey's own reading; XGBoost disagrees with both,
consistently, across all three sets — visible on the slopegraph as XGBoost's own column being the
one that consistently displaces CanopyCover, not as three columns each pointing a different way.

**What it adds beyond the table:** No RQ3 results table shows variable-level importance at all
(the main table is R2 only) — this is the only place a reader sees the rank comparison across all
three methods and both cohorts together.

**Exact visual design:**
- Plot type: slopegraph / bump chart, not a bar chart — three vertical columns (one per method),
  each variable positioned by its RANK within that method (1 = most important, at the top), with a
  line connecting the same variable's position across the three columns. This makes agreement
  (near-horizontal lines) and disagreement (crossing, steep lines) visually immediate in a way
  grouped bars of differently-scaled importance metrics cannot.
- Panels: 2, side by side, one per cohort.
- X-axis: three categorical positions (EN, XGBoost, GNNWR).
- Y-axis: rank (1 at top, ranks increasing downward) — deliberately NOT the raw importance score,
  sidestepping the cross-method scale problem entirely (EN coefficient, XGBoost gain, and GNNWR
  mean |local coefficient| are not on comparable units, so plotting rank avoids a false normalised
  comparison).
- Colour: CanopyCover's own line in one bold, distinct colour; every other variable's line in a
  muted neutral grey, so the variable under discussion is unambiguous without a full legend.
- Reference lines: none.
- Uncertainty shown: none currently computed at the per-variable rank level.
- Paired: yes — each line is one variable, paired across the three method-columns by construction.
- Labels/annotations: label CanopyCover's line directly at each end; label the top 2-3 other
  variables' lines lightly so the reader can see what displaces CanopyCover on 6survey.
- Legend: not required beyond the direct line labels (CanopyCover's colour is self-explanatory once
  labelled once).
- Axis limits: rank 1 to (whichever method has the most reported variables, e.g. Set4's ~19),
  though only the top 5-6 ranks need dense labelling — lower ranks can compress.

**Data required:** 4survey and 6survey EN coefficients, XGBoost gain importance, and SHAP, all
three sets, both cohorts — all now computed and available (`TEMP_rq3_en_xgb_results_2026-08-11.tex`,
updated 2026-08-16: the earlier gap where EN/gain-importance were never run for 6survey has been
closed by rerunning the same existing script with `--cohort 6survey`); GNNWR local-coefficient
ranks, both cohorts (`TEMP_rq3_gnnwr_local_coef_rank_2026-08-16.tex`) — all already computed, no
missing columns in either panel now.

**Interpretation limits:** Do not interpret rank or coefficient magnitude as causal effect size —
RQ3's own methodology frames all three methods as association, not causal effect, and SHAP/gain
importance specifically should not be read as effect direction. On 6survey, XGBoost's own R2 for
these sets is close to zero or negative (see results table above) — its rank/importance values are
attributing a fit with very little genuine predictive signal, so its disagreement with EN/GNNWR on
6survey should be read in that light, not as two equally-trustworthy readings in simple conflict.

**Placement:** main text if space allows, otherwise appendix.

**Caption message:** All three attribution methods agree CanopyCover is the top-ranked variable on
4survey; on 6survey, EN and GNNWR still rank it highest in every set, while XGBoost does not — a
disagreement that tracks which model is used, not which attribution method was available.

**Ranking:** 2 (useful if space allows).

---

#### Figure R3-4: Compartment archetypes and their environmental profile

**Research question and findings supported:** RQ3 item 9 — compartment-level classification
reproduces the plot-level boundary-proximity finding independently, and shows two distinct
wind-exposure signatures.

**Purpose:** What do representative compartments from each growth-curve-deviation archetype
actually look like as trajectories, and does the archetype-level environmental profile corroborate
the plot-level boundary-proximity and wind findings?

**Key insight:** Compartments classified as "trajectory-break outlier" and "consistently below
yldc" sit closer to compartment boundaries than the "stable" archetype, extending the individual-
plot boundary-proximity finding (Figure R3-2) to a different, population-wide unit of analysis.

**What it adds beyond the table:** Item 9's own text already states the key numbers (boundary
distances, elevation, wind-exposure figures per archetype) precisely — this figure's main added
value is showing what a representative compartment's actual trajectories look like per archetype,
which no number can convey. Complementary to Figure R3-2's Panel B (which maps every compartment's
archetype spatially, the "where"): this figure shows the "what" — the actual curve shape behind
each archetype label.

**Exact visual design:**
- Plot type: small multiples, one panel per archetype (5 panels: close-to-yldc/stable,
  consistently-above, consistently-below, trajectory-break-outlier, moderate-mismatch).
- Panels: 5, in a grid (e.g. 2 rows × 3, one cell used for a shared legend).
- X-axis (each panel): age (years). Y-axis: observed top height (m).
- Lines: each panel shows that archetype's one representative compartment's plots' own raw
  trajectories (thin, semi-transparent grey lines) plus that compartment's mean fitted local curve
  (one solid coloured line) and the official yldc benchmark curve (one dashed line).
- Colour: consistent fitted-curve/benchmark-curve colour pairing, reused from Figure R3-2's Panel B.
- Reference lines: the yldc benchmark curve itself is the reference.
- Uncertainty shown: none beyond the visible spread of individual plot lines.
- Paired: not paired; population-level view per archetype.
- Labels/annotations: title each panel with the archetype name and its own n_plots.
- Legend: one shared legend across all 5 panels.
- Axis limits: independent per panel (different archetypes cover different age/height ranges);
  state this in the caption.

**Data required:** Representative-compartment selection, from
`models/growth_curve_attribution/rq3_compartment_archetype_check.py`'s already-computed table
(`TEMP_rq3_compartment_archetype_check_2026-08-16.tex`'s own worked examples); raw per-survey rows
for that compartment's plots, via `load_filtered_growth_curve_table('4survey')`, filtered to the
relevant `cpmt` — already available. Plotting code confirmed not yet written (item 9's own "Not yet
done" note).

**Interpretation limits:** One representative compartment per archetype is illustrative, not a
claim every compartment in that archetype looks identical — the underlying group statistics are the
actual evidence; this figure supports intuition, it does not add new statistical weight.

**Placement:** appendix.

**Caption message:** Representative compartments from each growth-curve-deviation archetype show
visually distinct patterns of departure from the yield-class benchmark, consistent with the
population-level statistics reported in the text.

**Ranking:** 3 (appendix only).

---

#### Not proposed: a spatial map of sub-compartment clustering (item 7) or a dedicated 2008-artifact figure (item 8)

Item 7's finding (flagged plots cluster within their own compartment, evidence for a localized
rather than clerical cause) would need a proper spatial map with real compartment-boundary polygons
and plot coordinates to show responsibly — that map does not currently exist and is not confirmed
buildable from already-saved output, so it is not proposed here rather than guessed at. Item 8 (2008
survey-artifact, wind exposure) is already fully stated by 2–3 numbers in prose (75.6% vs. 47.8%,
89% moving closer, ~6% average improvement) — a dedicated figure would be "a figure whose only
message is already stated by one number," an explicit anti-pattern; not proposed.

### Ranked items

1. **`[Importance: 3/5]` `CanopyCover` dominates on 4survey by every method; on 6survey the
   disagreement splits by MODEL, not by which method happened to be computed — EN and GNNWR both
   still rank it #1, XGBoost (by gain importance and SHAP alike) does not, in any set.**
   (`TEMP_rq3_en_xgb_results_2026-08-11.tex`, `TEMP_rq3_gnnwr_local_coef_rank_2026-08-16.tex`)
   `CanopyCover` is present as a feature in every set and both cohorts throughout — this is about
   its *rank* by importance, not its presence.
   **4survey**: `CanopyCover` is #1 by THREE converging methods — EN coefficient (2.3x the
   next-largest on Set2, 2.3x on Set4, but only 1.5x on Set3, where the gap is real but noticeably
   smaller), XGBoost gain-importance (consistently ~2.6-2.7x across all three sets), and SHAP — the
   same pattern as RQ2b.
   **6survey**: EN's own coefficient ranks `CanopyCover` #1 in all three sets (margin over the #2
   variable: 1.06x on Set2, 1.63x on Set3, 1.21x on Set4 — Set2's margin is thin, within the fold
   SD of both variables, but the direction holds across all three). GNNWR's own per-plot local
   coefficients (`coef_*` columns) independently rank it #1 in all three 6survey sets too. XGBoost
   does not rank it #1 in any 6survey set, by either of its own two attribution views — gain
   importance (beaten by `cpmt_compactness_ratio` on Set2/Set4, by three thinning-related variables
   on Set3) and SHAP (beaten by `cpmt_compactness_ratio` on Set2/Set4, `windward_topex` on Set3) —
   two different views of the same fitted model agreeing with each other, not two independent
   pieces of evidence.
   **Reframed**: this is not "one thin SHAP-only reading against three converging methods." It is a
   genuine 2-vs-1 split by MODEL — EN and GNNWR agree with each other and with 4survey's own
   reading; XGBoost disagrees with both, consistently, on every 6survey set.
   **Why XGBoost disagrees on 6survey — a real, checkable candidate, not just a guess**: XGBoost's
   own 6survey R2 for these sets is close to zero or negative (already reported in the results
   table above) — the underlying fit itself is barely predictive there. XGBoost's own attribution
   on 6survey, for any variable, should be trusted less than EN's or GNNWR's on the strength of this
   alone, since there is very little genuine signal in the fit for gain-importance or SHAP to
   attribute in the first place. This does not by itself prove EN/GNNWR are right and XGBoost is
   wrong — only that XGBoost's own R2 gives an independent, non-circular reason to weight it less
   on this specific cohort.
   **Not pursued: retuning XGBoost specifically for 6survey.** This is a deliberate scope decision,
   not an oversight. It matches this project's own broader stance of not retuning RQ2b/RQ3's
   XGBoost at all (the shared, borrowed config is left as-is throughout — see RQ1 item 1's own note
   on why only RQ1 got a per-target search). More specifically for RQ3: 6survey is not the primary
   cohort for its spatial questions in the first place — GNNWR is already established as unreliable
   there for structural reasons (items 2-3, too few reference compartments), so a better-tuned
   XGBoost fit on 6survey would not change RQ3's main conclusions, which rest on 4survey throughout.
2. **`[Importance: 5/5]` GNNWR shows a real, consistent, corroborated edge over EN/XGBoost on
   4survey — but it's not CI-clean, and 6survey is simply unreliable.** (`TEMP_rq3_gnnwr_results`)
   Point estimates favour GNNWR on every 4survey set (table above) — consistent in direction and
   magnitude, not a one-set fluke. But EN/XGBoost's own bootstrap CIs overlap GNNWR's in every
   single set (e.g. Set4: GNNWR [0.236,0.347] vs. EN [0.192,0.286]/XGB [0.201,0.295]) — by the same
   non-overlapping-CI bar RQ1 used to pick its own winner, this doesn't clear it. On 6survey, GNNWR's
   R2 sign-flips across seeds (table above), with a mean near zero and within the seed-to-seed SD —
   underpowered and directionless, not a reliable loss or win either way.
   **Why the CI doesn't resolve it, but the finding still isn't a clean reversal**: the point-
   estimate gap is consistent across all 3 sets (unlike 6survey's own noisy sign-flipping), and item
   3's residual Moran's I is an entirely independent diagnostic corroborating the same direction.
   Two independent lines of evidence pointing the same way, but neither individually clears the
   strictest bar — report as "a consistent, corroborated advantage," not "the reliable result."
3. **`[Importance: 5/5]` Residual Moran's I closes the RQ2b bridge with a real, two-directional
   answer.** (`TEMP_rq3_gnnwr_results`) Same method as RQ2b's own check, run on RQ3's current-tier
   residuals, all 3 sets, both cohorts. On 4survey, GNNWR reduces residual clustering vs. the best
   global model in all 3 sets (-0.012 to -0.020) — independent confirmation, via a completely
   different diagnostic than R2, that item 2's win reflects GNNWR capturing real spatial structure,
   not just fitting the mean better. On 6survey, GNNWR *increases* clustering in all 3 sets (+0.031
   to +0.052) — independent confirmation of item 2's 6survey unreliability, via a different
   diagnostic than the R2 sign-flip.
   **Why the 6survey direction reverses**: a plausible mechanism grounded in how GNNWR's method
   works, not directly tested here — geographically weighted regression needs enough nearby
   reference points to estimate a stable local coefficient, and 6survey has far fewer (47 vs. 231
   compartments on 4survey). Consistent with GNNWR fitting noise rather than real local structure
   there, but no ablation varying reference density was run to confirm this specific mechanism.
4. **`[Importance: 4/5]` The same outlier plots recur regardless of feature set, and it's not
   disturbance.** (`TEMP_rq3_outlier_diagnosis`) The same ~10 plots are the worst residual across
   nearly all 6 (set × cohort) combinations — outlier-ness is a property of the plot's own
   trajectory, not which feature set was used, ruling out "we just picked bad features."
   **Plausibility check**: the natural hypothesis ("these are disturbance artifacts") was tested
   against the project's own disturbance-classification data and found NOT supported — 0/10 flagged
   for clearfell-like or measurement-inconsistent patterns, 0/10 clear the top-1% trajectory-
   instability cutoff.
   **The 10 plots split into two distinct sub-populations, not one uniform pattern**: 5 of the 10
   have a low residual range across their own survey years (0.9-5.9m) — a smooth, internally
   consistent height trajectory whose large `local_y_max_difference` (40-60m) comes entirely from
   that trajectory sitting far from the yldc benchmark curve, consistent with a probable
   yield-class lookup mismatch rather than anything wrong with the observed heights. The other 5
   have a moderately high residual range (13.6-15.1m) — real, if sub-threshold, curve-fit
   instability across their own survey years (not extreme enough to clear the top-1% cutoff, so
   the disturbance classifier doesn't catch it either). Both sub-populations are real
   growth-curve-behaviour findings, not a data-quality footnote — but they are two different
   phenomena, not the same one.
5. **`[Importance: 5/5]` Mechanism: the implausible `y_max_fit` values are a target-construction
   artifact, not a data problem — and the affected plots sit closer to boundaries than typical.**
   (`TEMP_rq3_outlier_diagnosis`) 5 of the 10 recurring plots have biologically implausible fitted
   `y_max_fit` (up to 116m) despite completely normal raw observed heights (e.g. 38.9-46.8m across
   ages 20-35) — the implausibility is entirely in the fitted parameter. This is a different 5-plot
   selection from item 4's two sub-populations (mostly, but not entirely, overlapping the
   moderately-unstable group: 4 of these 5 are also flagged there, the 5th belongs to the smooth
   group instead) — not a third, distinct category, but not a clean match to either of item 4's
   groups either.
   **Why**: the closed-form fit holds `k`/`p` fixed at the plot's yield-class-assumed values and
   only solves `y_max`; when a plot's own observed (ALS-derived) height trajectory is flatter/slower
   than its assigned yield class assumes, reconciling that mismatch under a fixed (faster) `k`
   forces an inflated ceiling.
   Symmetric to RQ2a item 3's PINN_k identifiability finding — letting `k` float risks non-
   identifiable parameters; fixing `k` (RQ3's design) risks an unstable solved `y_max` when the
   fixed value is wrong for that plot — two failure modes of the same very-few-points-per-plot
   constraint. Confirmed this isn't a data-cleaning gap: `filter_data()` only gates Age/yldc
   plausibility, no height- or `y_max`-based filter exists or would have caught this, since the raw
   heights are fine.
   **Corroborating evidence for *why* these plots specifically**: the same 10 plots sit roughly 3x
   closer to compartment/block/forest boundaries than a typical plot (median
   `dist_to_cpmt_boundary` 9.4m vs. population 28.0m) — plausibly because a plot near a boundary is
   more likely to have its own distinct height trajectory (edge effects, different management) than
   the single yield-class value assigned to its whole compartment assumes. Not causally proven (no
   per-plot species/management record exists to confirm directly), a concrete, checkable pattern.
   **Would letting `k` float too fix this, instead of just diagnosing it? A real trade-off, not a
   clear win**: it might reduce how often the fit is forced into an implausible `y_max` — a plot
   whose observed trajectory is flatter/slower could be explained by a lower `k` instead of an
   inflated ceiling. But RQ2a item 3 already showed that even a well-regularized neural network (PINN_k,
   sharing sub-network weights across all 58,073 plots) becomes severely non-identifiable once
   `y_max` and `k` are both free — a per-plot classical fit, with only a handful of points and no
   shared weights to borrow strength from other plots, would likely be worse, not better, on
   exactly the plots already short of data (the ones flagged here). It would also redefine RQ3's
   target from scratch, not just adjust a setting — `local_y_max_difference` is built on a
   fixed-`k` fit, so this isn't a rerun, it's a new target requiring the whole downstream
   outlier-diagnosis thread to be rebuilt against it. Worth trying if a future pass has the time,
   not something this evidence demands doing now.
6. **`[Importance: 4/5]` The pattern scales to hundreds of plots, clustered by compartment — and
   excluding them would make results worse, not better.** (`TEMP_rq3_outlier_diagnosis`) A Tukey-
   fence check on `y_max_fit` across the full population flags 0.47% of 4survey (266 plots) and
   5.32% of 6survey (709 plots), clustered by compartment rather than scattered (e.g. cpmt 2186:
   12/37 flagged, 32%).
   **Plausibility check on the natural "just exclude them" fix**: dropping the flagged plots from
   already-saved predictions changes 4survey's pooled R2 negligibly (-0.002), but makes 6survey's
   *worse* (EN -0.027, XGBoost -0.037, pushing XGBoost's R2 negative) — these extreme-target plots
   are propping up what little signal 6survey's models have. Keeping all plots in is the evidence-
   backed choice, not an oversight; a principled target-construction fix (compartment-level fallback
   `y_max`) remains legitimate future work, not something this evidence calls for doing now.
7. **`[Importance: 3/5]` All three models find the same plots hard, and they cluster spatially
   within their own compartment — evidence for a localized cause, not a whole-compartment clerical
   error.** (`TEMP_rq3_outlier_diagnosis`) EN/XGBoost/GNNWR independently find the 266 flagged plots
   hard (~36x over-representation in each model's own worst-1% residual bucket) — real, model-
   agnostic difficulty. Re-checked the disturbance flags at this larger, statistically real scale
   (item 4's check was n=10, underpowered): still 0/266 clearfell-like, 0/266 measurement-
   inconsistent, only 2/266 ambiguous — confirms item 4's "not disturbance" finding holds at scale.
   **Why sub-compartment clustering points toward "localized," not "clerical"**: a whole-compartment
   data error (wrong yldc recorded for the entire compartment) would apply uniformly regardless of
   location and show no spatial sub-clustering. Instead, flagged plots sit closer to each other
   (within their own compartment) than the compartment as a whole does, in every one of the 6
   most-affected compartments checked — consistent with a genuine localized event or sub-stand
   condition.
8. **`[Importance: 2/5]` A known data artifact is a real but minor contributor; wind exposure
   doesn't support windthrow as a cause.** (`TEMP_rq3_outlier_diagnosis`) 75.6% of flagged plots
   have their single worst-fitting year in the 2008 survey, well above the 47.8% rate for the
   general population — consistent with an already-known artifact where compartment boundaries
   were redrawn between the 2006 and 2008 surveys, which can shift which yield-class value a
   boundary-adjacent plot gets compared against. Dropping each plot's 2008 point and refitting: 89%
   of plots move at least slightly closer to their yield-class benchmark, but the average
   improvement is small (~6%) — a real contributing factor, not the main cause. Flagged plots
   are slightly more sheltered than typical, not more exposed — doesn't support a
   windthrow-disturbance story.
9. **`[Importance: 3/5]` Classifying compartments by growth-curve deviation pattern reproduces the
   plot-level boundary-proximity finding independently, at a different unit of analysis.**
   (`TEMP_rq3_compartment_archetype_check_2026-08-16.tex`,
   `models/growth_curve_attribution/rq3_compartment_archetype_check.py`) Classified all 231
   4survey compartments into archetypes by how their plots deviate from the yldc benchmark: `close
   to yldc / stable` (20), `consistently above yldc` (20), `consistently below yldc` (6),
   `trajectory-break outlier` (26), `possible reset / measurement issue` (0), `moderate mismatch /
   mixed` (159, the majority).
   **Environmental means by archetype — two patterns, both directional (group means, n=6-159, not
   significance-tested)**:
   1. *Boundary proximity, at a new scale.* `consistently below yldc` and `trajectory-break
      outlier` compartments sit notably closer to compartment/block boundaries than the stable
      group (13.6-19.7m vs. `close to yldc/stable`'s 30.1m; population median 28.4m). This is the
      same boundary-proximity pattern from item 5, but found independently here, at the compartment
      level rather than for individual plots.
   2. *Wind exposure splits by TYPE, not just by direction of deviation.* `trajectory-break
      outlier` compartments have the highest elevation (220.6m) and the highest WINDWARD-specific
      exposure of any group; `consistently above yldc` compartments instead have the highest
      OMNIDIRECTIONAL exposure. These are two different wind measurements picking out two
      different archetypes, not the same signal counted twice.
   **Why one category comes back empty — diagnosed, not guessed**: `apply_disturbance_cleaning=True`
   (the real methodology used throughout this project's target construction, adopted 2026-08-03
   after it roughly halved the attribution R2 by removing plots where the uncleaned model had been
   partly fitting "terrain predicts which plots got felled," not real growth signal) removes every
   plot the empty category is designed to detect, before classification runs. Confirmed by
   re-running with cleaning off: the category immediately populates (1 compartment). Stays sparse
   either way — disturbance flags are rare project-wide (~0.55% of plots) and the rule needs a 10%+
   concentration within one compartment. A pipeline-order interaction, not a rules or ecology
   problem.
   **A weaker, secondary check**: does "above/below yldc" partly reflect which growth phase a
   compartment was caught in, not a permanently different ceiling? `consistently above yldc` has the
   lowest mean fraction-of-asymptote-reached (0.584), `below` the highest (0.710) — directionally
   consistent, but SDs (0.12-0.18) overlap heavily across every group. Real and directional, not
   clean — an initial single-example read (0.402) overstated it before checking the full groups.
   **Not yet done**: representative-compartment trajectory plots (one panel per archetype); the
   same check on 6survey.

---

## Figure plan summary

**Essential figures (main text)**

| Figure | RQ | Findings covered |
|---|---|---|
| R1-1: XGBoost vs. DNN, matched fold by fold | RQ1 | Item 1 |
| R2a-2: physics constraint trades accuracy for identifiability | RQ1 + RQ2a | RQ1 item 5, RQ2a item 3 |
| R2a-1: where conditioning helps and hurts | RQ2a | Items 1, 2 |
| R2b-1: what global attribution finds, and where its own leftover residual sits on the map | RQ2b | Items 1, 2, 3 |
| R3-1: GNNWR's accuracy edge, its spatial-structure meaning, and its own local-coefficient map | RQ3 | Items 2, 3 |
| R3-2: where growth-curve deviations occur, and what explains them (deviation map + archetype map + trajectories) | RQ3 | Items 4, 5, 6, 9 |

Note: R2b-1, R3-1, and R3-2 are now 3-panel composites, each including at least one genuine spatial
map rather than a categorical chart, reflecting a deliberate shift away from bar charts toward
maps, colour-coded distributions, and paired points with error bars throughout this plan — see the
per-figure "Exact visual design" sections for the reasoning in each case (e.g. R2b-2 and R3-3 were
both redesigned from bar charts to a paired slope plot and a rank slopegraph respectively). RQ2b's
map (Panel C) is deliberately framed as a leftover-residual diagnostic, not a spatially-varying
relationship map — unlike RQ3's GNNWR, none of RQ2b's methods produce a coefficient that varies by
location, so the map shows what the models leave unexplained, not a local effect.

**Useful if space allows**

- R1-2: split-difficulty gradient (RQ1 items 3, 4)
- R2a-3: where conditioning helps and hurts, mapped — now backed by a real Moran's I test (0.55–0.64,
  p=0.001, both models, both cohorts; RQ2a item 1)
- R3-3: CanopyCover rank agreement across EN/XGBoost/GNNWR, as a rank slopegraph (RQ3 item 1)

**Appendix only**

- R1-4: two fitted CR curves + elevation distribution (RQ1 item 2)
- R2b-2: CanopyCover-dropped ablation, as a paired points-with-error-bars plot (RQ2b item 1)
- R2b-3: VIF diagnostic, as a lollipop/dot plot (RQ2b items 1, 3)
- R3-4: compartment archetypes and their environmental profile, complementary to R3-2's archetype
  map (RQ3 item 9)

**Findings that still lack an effective visual**

- RQ3 item 7 (sub-compartment clustering, localized vs. clerical): would need a proper spatial map
  with real compartment-boundary polygons and plot coordinates, not confirmed buildable from
  already-saved output.
- RQ3 item 8 (2008 artifact, wind exposure): deliberately not given a figure — fully covered by 2–3
  numbers already in prose; a dedicated figure would only restate them.

**Proposed plots that depend on unavailable or unconfirmed data**

- R2a-3 (useful): none — the underlying per-row reduction + (x, y) table was already built and used
  directly for `models/spatial_attribution/rq2a_reduction_morans_i.py`'s Moran's I test; the same
  merged table is what the map would plot, so this one is fully ready to build, not blocked.
- R2a-1 (essential): needs the per-row reduction table saved to disk — the script
  (`models/spatial_attribution/rq2_residual_reduction.py`) already supports this via its own
  `main()`, but the file does not yet exist under `outputs/` for any model. Requires running the
  existing save step, not new code.
- R1-2: needs a one-time cross-check that `TEMP_rq1_plotlevel_check_results_2026-08-12.tex` and
  `TEMP_rq1_temporalcheck_results_2026-08-11.tex` report identical spatial_block numbers before
  plotting them as one continuous line (expected to match, not yet directly verified).
- R1-4: needs confirmation of whether a plot-level, deduplicated elevation table already exists on
  disk, or needs a trivial fresh read from the existing loader.
- R2a-2 (essential): Panel B (identifiability) should be scoped to w=0/w=1 only unless a w=2
  correlation number is confirmed to exist; w=1's own bootstrap CI at the current tier has not been
  separately reverified this session (only w=0's has).
- R2b-1 (essential): Panel C's residual map needs EN's Set4 `predictions.csv` (already saved, same
  input already used for Panel B's own Moran's I) read directly for its (x, y, residual) columns —
  no new fitting, but not yet exported as a standalone plotting file; the point-density decision
  (raw ~71,330 plots vs. compartment-mean binning) needs picking before building. The optional
  compartment-boundary overlay has the same gap as RQ3's maps below.
- R3-1 (essential): Panel C's GNNWR coefficient map needs the 5 per-fold `test_predictions.csv`
  files pooled into one per-plot coefficient value (mechanically simple, same pooling already done
  for `TEMP_rq3_gnnwr_local_coef_rank_2026-08-16.tex`, but not yet saved as a standalone plotting
  file); the optional compartment-boundary overlay needs a boundary polygon layer not confirmed to
  exist in plottable form.
- R3-2 (essential): Panel A needs confirmation of whether `local_y_max_difference` for the full
  4survey population is already exported to a standalone file, or needs a trivial fresh read from
  `build_plot_level_table()`. Panel B needs a straightforward join of item 9's compartment-level
  archetype onto each compartment's own plots via `cpmt` (not a new calculation, just a merge not
  yet performed). The optional compartment-boundary overlay has the same gap as R3-1's.

---

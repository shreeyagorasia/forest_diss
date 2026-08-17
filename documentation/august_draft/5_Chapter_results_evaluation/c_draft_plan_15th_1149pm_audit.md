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

1. Two fitted CR curves (4survey vs. 6survey) on one age axis, next to an elevation-distribution
   comparison between cohorts — general-hypothesis visual for items 2/3 (data ready, not plotted).
2. Grouped-bar or dumbbell version of the results table above, model on the x-axis, cohort as
   colour/facet — the visual for item 1's headline (currently table-only).
3. Dumbbell: single-split spatial_block R2 → temporal R2, one row per model — the visual for item 3.
4. Dumbbell: plot_level (easy-split) R2 → single-split spatial_block R2, one row per model — the
   visual for item 4, makes the DNN-vs-PINN/PINN_k asymmetry visually obvious.
5. Loss curves by training epoch, physics weight as the colour, plus a bar+error chart of R2 across
   the weight range — the visual for item 5.

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

1. Quartile bar chart: mean reduction by CR-residual quartile (Q1 best-fit to Q4 worst-fit),
   colour = model — the central plot for item 1, data fully computed (5-seed DNN quartile table +
   XGBoost quartile table both exist), not yet built.
2. Spatial "help map": x/y already saved in every row, not yet plotted — where geographically does
   the reduction concentrate.

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

1. Coefficient forest plot (one row per variable, point = mean, whisker = ±SD across folds, NLME
   and EN side by side, faceted by set) — the central plot, serves items 1 and 3 at once (mean
   ±SD numbers already exist in the fitted tables, nothing new to compute).
2. Grouped bar: R2 and NLME spatial-variance-explained, Set1 (baseline) vs. Set4 (full) — the
   visual for item 1's CanopyCover-dropped ablation.
3. Bar chart: Moran's I by set, EN and XGBoost — the visual for item 2; flat/barely-moving bars
   across sets is itself the finding.
4. Bar+errorbar: pooled R2 by set with 95% CI, EN and XGBoost — the visual for item 4's CI-overlap
   point (Set2/Set4 whiskers visibly overlapping).
5. Bar chart: VIF per variable within Set4's 19-column context, 5.0 threshold line — supporting
   diagnostic for item 1's thinning-collinearity footnote and item 3's multicollinearity-ruled-out
   check.
   None of these are per-plot/example figures — RQ2b's target is a plot's *persistent, whole-history*
   departure, which fits a population-level story; a single-example figure belongs in RQ3's own
   outlier-diagnosis machinery instead, not here.

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

1. Coefficient/SHAP comparison plot (EN coefficients + XGBoost SHAP, 4survey and 6survey side by
   side, all 3 sets) — the central plot for item 1, and the one that would make the 6survey
   `CanopyCover` reversal visually obvious rather than buried in a table.
2. Bar+CI: pooled R2 by set and model (EN/XGB/GNNWR), 4survey — the visual for the results table
   and item 2's CI-overlap point.
3. Bar chart: residual Moran's I by set/cohort/model — the visual for item 3's two-directional
   result.
4. Histogram of `y_max_fit` (or `local_y_max_difference`) across the full population with Tukey
   fences marked, the 266/709 flagged plots as a visibly distinct tail — general-hypothesis visual
   for item 6.
5. Spaghetti plot: random ~30-50 plot sample's Age-vs-height trajectories (grey) against the yldc
   benchmark curve, one normal example highlighted, next to a small-multiples panel using the
   categories established in item 4 (smooth-but-offset / moderately unstable) plus item 8's
   2008-influenced pattern. Split role: grey spaghetti is general-hypothesis (population-level);
   small multiples is the specific/typology visual — pair them so the figure covers both registers.
6. Representative-compartment trajectory plots (one panel per archetype: observed height by age,
   mean fitted local curve, mean official yldc curve) — the visual for item 9's classification;
   data fully computed, plotting code not yet written.

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

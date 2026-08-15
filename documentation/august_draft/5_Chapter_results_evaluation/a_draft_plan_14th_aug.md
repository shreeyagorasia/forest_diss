# Results chapter — discussion primer and draft plan

Copied from `TEMP_results/README.md`'s discussion primer on 2026-08-15, to continue analysis here
directly. The full results index (all TEMP files, per-experiment status) stays in
`TEMP_results/README.md` — this is just the ranked discussion primer + migration-tracking table,
now living where the results/evaluation chapter draft is being worked on.

---

## Discussion primer — ranked, per RQ (2026-08-15)

Not a draft, not exhaustive — a prompt for you to react to. Per RQ, the results most worth
building discussion around, ranked by how much they matter for actually answering that RQ and how
genuine/non-obvious the finding is (not just "biggest number"). Pull whichever ones you want to go
deeper on.

**Appendix-only, not discussion**: the XGBoost hyperparameter bug-and-fix story (RQ1 baseline and
RQ2b both affected, RQ3 was never buggy) is real and worth reporting, but per your call it stays
out of the main discussion narrative in both RQ1 and RQ2b — cite the corrected numbers in an
appendix table, don't build a discussion section around the tuning story itself.

### RQ1 — raw height prediction, model comparison (throughline: 4survey vs. 6survey cohort)

1. **The winner is cohort-conditional, not absolute.** DNN's 4survey win is real and CI-confirmed
   (non-overlapping vs. every PINN variant); on 6survey, the apparent "PINN wins" doesn't survive
   scrutiny — DNN's CI fully contains PINN's, so it's a tie, not a loss for DNN
   (`TEMP_rq1_winner_reseed`). The honest answer to "which model wins" is "it depends which cohort
   you ask," not a single verdict — worth stating as the RQ1 throughline rather than picking one.
2. **6survey's negative bias is real and seed-independent** (`TEMP_rq1_winner_reseed`). Consistent
   across all 5 reseed seeds — not noise. **Now has a real, evidenced explanation, not just a
   question** (`TEMP_rq1_cohort_composition`, added 2026-08-15): 6survey's own frozen CR curve is
   a genuinely different shape from 4survey's (7.5m lower ceiling, more than double the growth
   rate), and 6survey is a compositionally different, lower-elevation, less-varied slice of
   Aberfoyle (mean 102m vs. 177m elevation, range capped at 351m vs. 561m — 6survey simply doesn't
   include 4survey's higher-elevation compartments at all). This reframes the bias as "6survey is a
   genuinely different population with its own growth relationship," not a data-quality flaw.
   **Also sharpens a methodology-chapter claim** (`TEMP_rq1_cohort_composition`): Linear regression
   is justified there by the age range sitting on the CR curve's near-linear segment. That's
   measurably less true for 6survey — Linear's own R2 drops across cohorts (0.580 to 0.509, current
   VIF-screened tier), and fitting a straight line directly to each cohort's own true CR curve shape
   confirms why: R2 of that linear approximation is 0.984 for 4survey but only 0.958 for 6survey, a
   direct, testable consequence of 6survey's own faster-growth-rate curve. Worth a one-line caveat on
   that methodology claim rather than stating it as true for both cohorts equally.
3. **Temporal forecasting degrades far worse on 6survey than 4survey** (`TEMP_rq1_temporalcheck`).
   PINN loses >3/4 of its R2 forecasting forward on 6survey vs. a much smaller drop on 4survey —
   ties model fragility directly to cohort/sample size, not just "temporal splits are harder." The
   same compositional evidence as item 2 plausibly extends here too: a smaller, more homogeneous
   population may generalise across time worse than a larger, more varied one, independent of row
   count alone.
   **Plot idea**: the two fitted CR curves (4survey vs. 6survey) plotted together on one age axis,
   next to an elevation-distribution comparison between cohorts — visualises "these are different
   growth relationships from compositionally different populations" directly, motivating both
   items 2 and 3 without a new model run (data ready, not yet plotted).
4. **No architecture helps both cohorts** (`TEMP_rq1_architecture_sweep`). `deeper` wins on 4survey,
   `small` wins on 6survey — architecture choice interacts with cohort rather than having a single
   right answer, reinforcing the "depends on which cohort" throughline rather than being a separate
   point.
   - Random Search and Reproducibility for Neural Architecture Search - Li, L. & Talwalkar, A.
     (2020). - closes to claim , general ML
   - Random Search for Hyper-Parameter Optimization (JMLR) -  Bergstra, J. & Bengio, Y. (2012). -
     highly cited
   - Process-Informed Neural Networks — Wesselkamp et al. 2024, Ecology Letters - closts to topic,
     need to check it actually says this about the architecture.
5. **Plot_level vs. spatial_block_kfold asymmetry** (`TEMP_rq1_plotlevel_check`). DNN inflates
   hugely under an easy split (+0.197 R2); PINN/PINN_k barely move (±0.006) — a real DNN-vs-PINN
   difference, less directly about cohort, but worth keeping if you want one non-cohort item.
   - Two splits, same DNN/PINN/PINN_k models, same data, same Set3, plot level split, held out
     nearest neighbour in training splits, so its memorising the data, under spatial split drop
     from 0.831 to 0.634. (PINN and PINN_k identical, split doesnt matter)
   - DNN can exploit but PINN cant is architecture. DNN age feeds each input (age and environment
     together), so has capacity of fit an arbitarily specific combination of features as a
     near-unique "signature" for a location if that minimizes loss, which is exactly what lets it
     exploit nearby-plot leakage. PINN's environmental features, by contrast, only ever pass
     through a narrow 16-unit sub-network that produces a single additive adjustment to y_max (and
     for PINN_k, a multiplicative adjustment to k), a much narrower bottleneck for how much
     fine-grained, location-specific information can influence the prediction at all. On top of
     that, the physics and trajectory losses actively pull those adjustments toward staying
     consistent with the shared curve's analytical derivative, further limiting how "idiosyncratic"
     any one plot's adjustment can become.
   - This is the same structural rigidity you've already seen cost PINN raw accuracy elsewhere in
     RQ1 (the physics-weight ablation) — it's the same mechanism, just showing up as a benefit here
     instead of a cost. PINN can't fit the data as flexibly as DNN, which is why it loses on
     plot_level (where flexibility pays off via leakage) but also why it doesn't lose anything
     moving to spatial_block_kfold (there's nothing to lose — it was never relying on that
     flexibility to begin with).
   - 1. It directly validates the choice of spatial_block_kfold as the primary split — and shows
     that validation is model-dependent, not a blanket methodological footnote. If you'd only run
     plot_level, you'd have concluded DNN generalizes far better than it actually does (0.831 is
     not a trustworthy number — it's an artifact of leakage, not real skill). For PINN, the choice
     of split barely matters, which is itself informative: it means PINN's own architecture is
     already doing some of the work that a careful spatial split exists to enforce.
   - 2. **Revised 2026-08-15, after auditing this whole "why PINN" case for bias**: an earlier
     version of this note claimed this was a "third" argument alongside "reproducibility" and
     "structural necessity (RQ2/RQ3 need PINN_k's y_max/k)." Both of those turned out to be
     overstated on closer check — reproducibility (RQ2a item 5) is a DNN-vs-XGBoost finding that
     doesn't test PINN at all, and RQ2/RQ3's actual attribution targets use a separate classical
     curve fit, not PINN_k's learned parameters (see RQ2a item 6's correction). What survives,
     on the actual evidence: this is a genuinely solid, independent argument on its own terms —
     not spin on "PINN is just less flexible." Reduced capacity preventing a model from exploiting
     spurious correlations (here, nearby-plot leakage) is the standard bias-variance/regularisation
     argument, not a reframed weakness. You need the careful spatial split to keep DNN honest;
     PINN's constraint does some of that work by construction. That is one solid, citable point
     about what physical constraints buy you methodologically — evaluate it as one piece of
     evidence among several (RQ2a §items 4-6 below), not as part of a three-legged case that
     doesn't all hold up.
6. **Physics-weight ablation — a real RQ1 finding in its own right, and the bridge into RQ2**
   (`TEMP_rq1_physicsablation`; expanded in RQ2a item 6). R2 decreases monotonically as the
   physics/trajectory loss weight increases (0→1→2), every model, every cohort, no exceptions —
   w=0 beats the default w=1 by +0.05 R2 on 4survey (real, outside both CIs) and +0.01 on 6survey
   (closer to noise). This is what justifies RQ1's headline comparison using w=1, not w=0 — but it's
   not just a methodology footnote: it's the first evidence in the dissertation that the physics
   constraint has a real cost, which is exactly the tension RQ2a/RQ2b then have to reckon with (does
   that cost buy something back — identifiability, leakage-resistance — worth paying?). Keep the
   full number here in RQ1 rather than only cross-referencing it, since it's this item's own result
   before it becomes anyone else's evidence.

### RQ2a — does environmental conditioning shrink the departure from the shared curve

**Base finding (items 1-3) — established using DNN/PINN/PINN_k, the original 90-combo sweep:**

1. **The core quartile pattern itself** (`TEMP_rq2a_residual_reduction`). The actual RQ2a answer:
   Q1 (CR already good) gets *worse* under environmental conditioning, Q4 (CR worst) improves a lot
   — a real trade-off, not a uniform improvement. This is the finding the whole RQ hinges on, and
   the reference point everything below is checked against.
2. **5-seed robustness of the quartile pattern** (`TEMP_rq2a_residual_reduction`, Step 4a). Confirms
   the Q1-hurts/Q4-helps shape isn't a single-seed fluke — tight SDs across all 5 seeds. Robustness
   check on item 1's own (DNN) numbers.
3. **Cohort asymmetry** (`TEMP_rq2a_residual_reduction`). 6survey's effect is real but much smaller
   than 4survey's (0.174 vs 1.501 mean reduction) — fits the broader 6survey-noisiness pattern
   across the whole project. Breakdown of item 1's own (DNN) numbers by cohort.
4. **XGBoost's own version of the quartile check** (`TEMP_rq2a_residual_reduction`). The most
   important corrective finding in RQ2: the "shrinks most where CR is worst" pattern is NOT
   PINN-specific — XGBoost shows it too, with comparable or larger average reduction. Forces the
   "why PINN" argument onto different (stability + structural) ground. **Conclusion and evidence**:
   the "shrinks the model's error most where CR already fits worst" pattern is a general property
   of giving any sufficiently flexible model environmental features — it is not something
   physics-guided architecture does specifically. XGBoost, which has no physics loss and no
   CR-curve embedding at all, reproduces the same pattern with a magnitude that matches or exceeds
   DNN's (item 1's numbers).
5. **Stability gap between DNN and XGBoost — NOT a PINN-specific finding, corrected 2026-08-15**
   (`TEMP_rq2a_residual_reduction`). A second, separate conclusion from the same XGBoost experiment
   as item 4 (not evidence for item 4 — a sibling finding): DNN's reduction is 5-14x more
   reproducible fold-to-fold than XGBoost's. **This does not test PINN at all — DNN has no physics
   guidance.** An earlier version of this note filed it under "why PINN"; that was a bucketing
   error caught during a bias audit of this whole section. At most this is evidence for "neural
   network over tree ensemble" (DNN vs. XGBoost), not "physics-guided network over plain network."
   Keep it as a real, separate finding, but don't cite it in defence of PINN specifically.
6. **The physics constraint trades accuracy for parameter identifiability**
   (`TEMP_rq1_physicsablation`, raw result stated in RQ1 §item 6). Removing the physics/trajectory
   loss (w=0) makes PINN_k *more* accurate, but its `y_max`/`k` become far less identifiable — they
   correlate at -0.93 (essentially interchangeable, a taller-ceiling/slower-growth pair fits about
   as well as a shorter-ceiling/faster-growth pair) vs. -0.45 at the default w=1. Without the
   constraint, PINN_k's own headline parameters stop being separately trustworthy numbers.
   **Scope, stated precisely (corrected 2026-08-15)**: this is not evidence that "RQ2/RQ3 need
   PINN's physics guidance" — RQ2b/RQ3's actual attribution targets are built from a separate
   classical curve fit, not PINN_k's learned parameters (see the RQ3 correction below). The real
   claim is narrower and self-contained: PINN_k advertises the ability to produce physically
   interpretable, plot-specific parameters as part of prediction; this finding shows that claimed
   capability is genuine (identifiable) only when the physics constraint is present, and illusory
   (parameters trade off near-perfectly) without it. That's one legitimate, accurately-scoped
   argument, not a claim that the rest of the dissertation's pipeline depends on it.
   **Correction caught while drafting this**: this is NOT evidence that RQ3's own `y_max` target is
   at similar risk — RQ3 fits `y_max` with `k`/`p` held FIXED from the yield-class lookup (a single
   free parameter, closed-form linear solve), not jointly estimated the way PINN_k's neural
   sub-network does.
   **Updated 2026-08-15, after RQ3's own outlier deep-dive — softens the "retrospective support"
   framing above.** An earlier version of this note treated RQ3's fixed-`k`/`p` choice as simply
   vindicated by this finding. RQ3's own outlier work (RQ3 item 6) since showed that choice has a
   real failure mode too, just a different one: when a plot's true growth genuinely diverges from
   its assigned yield class's `k`/`p`, the fixed-parameter closed-form fit can produce a wildly
   implausible `y_max` (up to 116m) — affecting a small but non-trivial share of the population
   (0.47% of 4survey, 5.32% of 6survey, RQ3 item 7's broader scan). **The honest framing is now
   symmetric, not one-sided**: floating `k` (PINN_k) risks non-identifiable parameters when the
   physics constraint is removed; fixing `k` (RQ3's design) risks an unstable solved `y_max` when
   the fixed value is wrong for a specific plot. Both are real, demonstrated consequences of the
   same underlying constraint — very few points (4-6) per plot forces a floating-vs-fixed design
   choice, and neither choice is free of failure modes. RQ3's choice is still defensible (a
   sensitivity check, RQ3 item 7, found excluding the affected plots isn't a net improvement — it's
   not a "fix" waiting to happen), but "retrospective support FOR that design choice" overstated
   the case; "a different, comparably-sized tradeoff" is the more accurate framing.

**Where the "why PINN despite losing on accuracy" case actually stands (restored/updated
2026-08-15)**: two legs survive, not three. (1) **Leakage-resistance** (RQ1 item 5) — PINN's narrow
environmental bottleneck and physics/trajectory constraints structurally prevent it from exploiting
nearby-plot leakage the way DNN can; a genuine bias-variance/regularisation argument, not spin. (2)
**Scoped identifiability** (item 6 above) — PINN_k's claimed physically-interpretable parameters are
only genuinely identifiable when the physics constraint is present; narrower than originally framed
(not evidence RQ2b/RQ3 need PINN's guidance), and now itself qualified further — RQ3's own
fixed-`k`/`p` design carries a comparably-sized tradeoff, so this isn't one-sided support for
physics-guided architecture over classical curve-fitting, just a real, self-contained claim about
what the physics constraint buys PINN_k specifically. What does **not** survive: reproducibility
(item 5) is a DNN-vs-XGBoost finding, not a PINN-vs-anything finding — don't cite it here. Net: a
smaller, more honest case than originally hoped, built on two independently-evidenced, narrowly-
scoped legs rather than a broad "physics helps" claim.

### RQ2b — attribution of the CR-residual to environmental/stand-structure variables

1. **CanopyCover/thinning dominance, converging across NLME/EN/XGBoost-SHAP** (`TEMP_rq2_attribution`).
   The actual headline: three independent methods agree, tightly, on the same answer — the single
   most consistent finding in the project.
   **Plot idea (the central plot for this whole section — general/population-level)**: a
   coefficient forest plot (one row per variable, point = mean, whisker = ±SD across folds, NLME
   and EN side by side, faceted by set) — the mean±SD numbers already exist in this item's own
   fitted tables, nothing new to compute. This single plot also directly serves item 4 (topex/slope
   show up as the tightest non-baseline whiskers) and item 5 (the variables that don't survive
   scrutiny show up as whiskers crossing zero or comparable to the mean) — build it once here,
   reference it from both rather than duplicating.
2. **CanopyCover baseline-only ablation, the necessary check on item 1** (`TEMP_rq2_attribution`,
   Set1 section). Item 1's dominance finding could look circular (CanopyCover is downstream of
   growth, not an independent driver) unless checked — this resolves that concern with actual
   evidence rather than assumption: environment adds real R2 beyond baseline, and — more decisively
   — explains real spatial variance the baseline explains almost none of. Reorder note: this is the
   response to item 1, so it's stated second, not first — reading order should be "here's the
   finding" then "here's why it isn't circular," not the caveat before the reader knows what it's
   caveating.
   **Plot idea (general/population-level)**: grouped bar chart, R2 and Moran's I side by side,
   baseline-only (Set1) vs. full set (Set4) — both numbers already exist in this item's own ablation
   table, directly visualises "environment adds real value beyond baseline" on two axes at once
   (accuracy and spatial-variance-explained) rather than requiring the reader to hold two separate
   claims in mind.
3. **Moran's I on the EN/XGBoost residual — computed 2026-08-15, moved up from item 9, the third
   piece of the headline story, not a trailing addendum** (`TEMP_rq2_attribution`). Residual
   clustering stays high (Moran's I 0.65-0.74, p=0.001 at the permutation floor) across every set,
   and adding more environmental information barely moves it (Set1 baseline-only: 0.738 → Set4 full
   set: 0.693 for EN). **This changes the overall RQ2b conclusion, not just adds a data point**: the
   honest complete story is now "here's what dominates (item 1), it's not circular (item 2), but
   substantial spatially-structured variance remains genuinely unexplained regardless of how much
   environmental data is added (this item)" — not "we've explained the CR-residual well." Also the
   strongest bridge to RQ3 in this whole list: real, current-tier evidence that global models leave
   a gap more environmental data doesn't fix, motivating RQ3's spatially-varying approach as a
   response to a demonstrated problem, not a fancier model tried for its own sake. **Update
   2026-08-15 — this comparison is now done, see RQ3 item 3**: the current-tier apples-to-apples
   check on RQ3's own GNNWR/EN/XGBoost residuals confirms the bridge with a real, two-directional
   result — GNNWR reduces this same kind of clustering on 4survey (-0.012 to -0.020 vs. the best
   global model) but increases it on 6survey (+0.031 to +0.052), independently corroborating both
   RQ3's 4survey win and its 6survey unreliability finding via a completely different diagnostic
   than R2.
   **Plot idea (general/population-level)**: bar chart of Moran's I by set (Set1-Set4, EN and
   XGBoost) — the numbers already exist in this item's own table, and the flat/barely-moving bar
   heights across sets are themselves the finding (visually reinforces "more environmental data
   doesn't fix this" without needing a spatial map). A residual map (colour = residual sign/
   magnitude, one set) is a possible second panel but not necessary to make the point — the bar
   chart alone carries it.
4. **`topex`/`slope_degrees` as the most stable real environmental signals** (`TEMP_rq2_attribution`).
   The actual environmental (non-baseline) attribution answer, distinct from the baseline-dominance
   story — both NLME and EN agree on direction/magnitude. **Plot**: no new plot needed — item 1's
   coefficient forest plot already shows this directly (tightest non-baseline whiskers).
5. **Variables that don't survive scrutiny** (`TEMP_rq2_attribution`). `chelsa_gdd5_degc`, `tas_mean`,
   `soilgrids_ph` flip sign or have SD comparable to their mean — honest "what we can't claim" list,
   useful for critical evaluation rather than overselling every coefficient. **Plot**: no new plot
   needed — item 1's coefficient forest plot already shows this directly (whiskers crossing zero or
   comparable to the mean).
6. **Set3 as the weakest set** (`TEMP_rq2_attribution`). Lowest R2, lowest NLME variance explained —
   ties directly to Set3's terrain-heavy, thin-wind, zero-soil/climate composition; a clean
   set-composition-to-outcome link worth stating explicitly. **Plot**: no new plot needed — item 7's
   R2-by-set bar+CI chart already shows this directly, just read for which set sits lowest.
7. **Bootstrap 95% CIs on RQ2b's pooled R2 — computed 2026-08-15, not previously available**
   (cluster-bootstrap on compartments, 1000 resamples, reusing `models/common/bootstrap_ci.py`
   unchanged, ~1 minute to run). RQ1 has had this from the start; RQ2b never did — a real,
   previously-flagged gap, now closed:

   | Set | EN R2 [95% CI] | XGB R2 [95% CI] |
   |---|---|---|
   | Set1 | 0.220 [0.180, 0.252] | 0.222 [0.183, 0.252] |
   | Set2 | 0.359 [0.310, 0.396] | 0.416 [0.355, 0.461] |
   | Set3 | 0.325 [0.272, 0.365] | 0.375 [0.321, 0.414] |
   | Set4 | 0.358 [0.305, 0.398] | 0.395 [0.338, 0.438] |

   **Tempers any reading of item 6's "Set3 is weakest" as implying a clean Set2/Set4 ranking above
   it, stated honestly rather than left as a clean point-estimate win**: Set2 and Set4's CIs overlap
   substantially (XGB: [0.355,0.461] vs [0.338,0.438]) — a real point-estimate gap, not a
   statistically clear-cut one. Set2 vs. Set3 looks more separated (supporting item 6's specific
   claim that Set3 underperforms). Every set's CI clears Set1's baseline-only CI with no overlap, so
   "environment adds real value beyond baseline" (item 2's ablation) is the one comparison here
   that's unambiguous; "which environmental set is best among Set2/Set4" is a softer claim than the
   point estimates alone suggest.
   **Plot idea (general/population-level — corrected 2026-08-15, the coefficient forest plot
   previously listed here belongs to item 1, not this item)**: bar+errorbar chart of pooled R2 by
   set (Set1-Set4, EN and XGBoost side by side, whisker = 95% CI) — this item's own table, plotted
   directly. This is the plot that actually shows the CI-overlap point (Set2 and Set4's whiskers
   visibly overlapping) that the coefficient forest plot cannot show, since that plot is about
   individual variable coefficients, not set-level R2 uncertainty.
8. **VIF check on the "core" attribution variables — computed 2026-08-15, found something more
   important than what was being checked for.** Set out to verify the SHAP-important core variables
   (`dist_to_road`, `chelsa_bio12_precip_mm`, `slope_degrees`, `CanopyCover`) aren't collinearity
   artifacts — they aren't (VIF 1.09-3.91, all well under the 5.0 threshold, computed within Set4's
   full 19-column context via `compute_vif_table()`). **But the baseline columns, which are exempt
   from VIF removal by methodological design, are not clean among themselves**:
   `time_since_thinning_missing` VIF=39.7, `time_since_thinning` VIF=22.6 — both far above the
   threshold every other column was screened against. This is a real, previously unstated caveat:
   these two coefficients (both large in every EN/NLME table, per item 1's headline) are almost
   certainly not separable from each other given VIF this high, and should be read as one combined
   "thinning-history" signal rather than two independently interpretable numbers — even though the
   pair together clearly carries real signal (the dominance finding itself is unaffected).
   **Plot idea (supporting/diagnostic, not a headline plot — narrower scope than items 1-3)**: bar
   chart of VIF per variable within Set4's 19-column context, with a horizontal line at the 5.0
   threshold — the two thinning columns visibly towering over it while the SHAP-important core four
   sit well under, in one glance. A small supporting figure for a caveat, not the section's main
   visual — same register as RQ3's item 9 (wind/2008 test), a narrow mechanism check rather than a
   population-level pattern.
9. **The map/per-plot-example idea from earlier this session belongs in RQ3, not here** — kept as a
   deliberate scope decision, not an oversight. RQ2b's target (`mean_cr_residual`) is a plot's
   *persistent, whole-history* departure, which fits a population-level "which variables converge
   across everyone" story (items 1-8 above) more naturally than a single-plot "why did this example
   improve" story, which is RQ3's own outlier-diagnosis machinery and already built there. **This is
   also the general-vs-specific boundary for the whole section, stated explicitly**: every plot idea
   above (items 1-3, 7-8) is general/population-level by design — RQ2b's own target construction
   makes a specific/single-plot-example figure a poor fit here, unlike RQ3 where it's the natural
   unit of analysis. If a specific/example figure is wanted for this material, it belongs in RQ3's
   outlier-diagnosis figures instead, not as a new addition here.

### RQ3 — plot-specific curve deviation (throughline: interesting growth-curve behaviour, not just which model wins)

**Reordered 2026-08-15** to lead with what most directly answers RQ3's own question, then its
honest caveat, then the outlier-diagnosis thread in its actual logical order (observe the pattern
before testing why), with the category-attribution side-analyses last since they're single-split,
single-set checks rather than the core 3-set x 2-cohort x 5-fold comparison.

**Framing bridge, added 2026-08-15, after RQ2b's Moran's I finding (RQ2b §item 3), now answered
not just hypothesised (item 3 below)**: RQ3 no longer needs to open as "we tried a spatially-varying
model and it happened to win" — RQ2b already showed its own global attribution models leave
substantial, statistically significant spatial clustering in their residuals (Moran's I 0.65-0.74),
and that giving them *more* environmental data barely reduces it. Item 1 below isn't an isolated
empirical win, it's a direct test of whether a spatially-varying relationship can capture what
RQ2b's global models demonstrably couldn't — and item 3, run the same day, shows GNNWR actually
does reduce that same kind of clustering, via a completely independent diagnostic from the R2
comparison.

1. **GNNWR beats EN/XGBoost on 4survey — real and consistent, but not CI-clean; audited and
   updated 2026-08-15** (`TEMP_rq3_gnnwr_results`). The primary, most direct answer to RQ3's actual
   question ("does letting the relationship vary spatially help"). Point estimates favour GNNWR on
   every set (Set2 0.320 vs. EN 0.286/XGB 0.266; Set3 0.319 vs. 0.274/0.281; Set4 0.294 vs.
   0.240/0.250) — consistent in direction and magnitude, not a one-set fluke.
   **Bootstrap CI check, computed 2026-08-15 (EN/XGBoost's own CI, closing a gap this item
   previously left open)**: `cluster_bootstrap_ci()` on the same already-saved predictions gives
   EN/XGBoost CIs that OVERLAP GNNWR's in every single set (e.g. Set4: GNNWR [0.236,0.347] vs. EN
   [0.192,0.286]/XGB [0.201,0.295]). By the same non-overlapping-CI bar RQ1 used to pick its own
   winner, this specific test does NOT clear it — "the one reliable positive result" is overstated
   as originally written, and the word "reliable" needs softening.
   **But this isn't a clean reversal either** — two things push back toward a real effect: the
   point-estimate gap is consistent across all 3 sets (not noisy direction-flipping the way
   6survey's numbers are, item 2), and item 3's residual Moran's I check is an entirely independent
   diagnostic that corroborates the same direction (GNNWR reduces spatial clustering EN/XGBoost
   leave behind, on 4survey specifically). **Honest synthesis**: a real, plausible edge supported by
   two independent lines of evidence (point estimate + spatial-residual diagnostic), but one that
   doesn't clear the single strictest bar (CI non-overlap) this project has applied elsewhere —
   report it as "a consistent, corroborated advantage" rather than "the reliable positive result."
2. **GNNWR 6survey reseed, correcting the original finding** (`TEMP_rq3_gnnwr_results`). The
   necessary honest caveat on item 1, not a separate model-comparison curiosity: "GNNWR loses on
   6survey" was wrong — R2 sign-flips across seeds, mean near zero. Keep the two together — item 1
   is the reliable half of this comparison, item 2 is the unreliable half, and both belong in the
   same breath rather than item 2 reading as a standalone aside.
3. **Residual Moran's I vs. EN/XGBoost — computed 2026-08-15, closes the RQ2b bridge with a real
   answer** (`TEMP_rq3_gnnwr_results`). Same method as RQ2b's check, run on RQ3's own current-tier
   residuals, all 3 sets, both cohorts. A clean, consistent, two-directional result: on 4survey,
   GNNWR reduces residual clustering vs. the best global model in all 3 sets (-0.012 to -0.020) —
   independent confirmation, via a completely different diagnostic than R2, that item 1's win
   reflects GNNWR actually capturing spatial structure the global models miss, not just fitting the
   mean better. On 6survey, GNNWR *increases* clustering in all 3 sets (+0.031 to +0.052) —
   independent confirmation of item 2's unreliability finding: not just an unstable accuracy number,
   but a sign that GNNWR is fitting noise rather than real local structure on the smaller cohort,
   consistent with its spatial-weighting mechanism needing more reference density (47 vs. 231
   compartments) than 6survey provides.
4. **Same outlier plots recur across nearly every (set, cohort) combo** (`TEMP_rq3_outlier_diagnosis`).
   Stated before item 5 because it's the observation that motivates testing why: outlier-ness is a
   stable property of the *plot's own trajectory*, not an artifact of which feature set was used to
   try to explain it — rules out "we just picked bad features" before asking what the real
   explanation is.
5. **RQ3 outlier diagnosis + disturbance cross-reference** (`TEMP_rq3_outlier_diagnosis`). Given
   item 4's finding that the same plots deviate regardless of feature set, tests *why*: a
   hypothesis ("these are disturbance artifacts") generated, then tested against the project's own
   disturbance-classification data, then found NOT supported (0/10 flagged). The revised
   explanation — a smooth, stable trajectory that's just far from the yldc benchmark, i.e. a
   probable yield-class lookup mismatch rather than a real disturbance event — is itself an
   interesting growth-curve-behaviour finding, not just a data-quality footnote.
6. **Mechanism behind the extreme `y_max_fit` outliers, plus a real boundary-proximity pattern —
   computed 2026-08-15** (`TEMP_rq3_outlier_diagnosis`). *Specific/example-level — the original
   10-plot finding.* Deepens item 5's "yldc lookup mismatch" read into something mechanistic rather
   than a label: 5 of the 10 recurring outlier plots have biologically implausible fitted `y_max_fit`
   (up to 116m). Their raw observed heights are completely normal (e.g. 38.9-46.8m across ages
   20-35) — the implausibility is entirely in the *fitted parameter*, not the data. It happens
   because the closed-form fit holds `k`/`p` fixed at the plot's yield-class-assumed values and only
   solves `y_max`; when a plot's true growth is flatter/slower than its assigned yield class
   assumes, reconciling that mismatch under a fixed (faster) `k` forces an inflated ceiling.
   **Symmetric to RQ2a item 6's identifiability finding**: letting `k` float (PINN_k) risks
   non-identifiable parameters; fixing `k` (RQ3's own design) risks an unstable solved `y_max` when
   the fixed value is wrong for that specific plot — two failure modes of the same constraint (very
   few points per plot forces a floating-vs-fixed design choice).
   **New, independent evidence for *why* these specific plots might be yldc-mismatched**: the same
   10 recurring outlier plots sit roughly 3x closer to compartment/block/forest boundaries than a
   typical plot (median `dist_to_cpmt_boundary` 9.4m vs. population 28.0m; `dist_to_block_boundary`
   12.1m vs. 29.3m; `dist_to_forest_perimeter` 29.7m vs. 165.7m) — plausibly because boundary-
   adjacent stands are more likely to have edge effects or a management history the compartment-
   level yldc lookup doesn't capture. Not causally proven (no per-plot species/management record
   exists to confirm directly), but a concrete, checkable pattern, not a hand-wave. **Also answers
   a natural "was this a data-cleaning gap?" question**: no — `filter_data()` only gates Age/yldc
   plausibility, and no height- or y_max-based filter exists or would have caught this, since the
   raw heights are fine; this is a target-construction artifact, not a raw-data one.
7. **Broader scan for implausible `y_max_fit`, and a rerun sensitivity check — computed 2026-08-15**
   (`TEMP_rq3_outlier_diagnosis`). *General/population-level — scales item 6 up from 10 hand-picked
   plots to the whole population.* A standard Tukey-fence check on `y_max_fit` across the full
   population flags 0.47% of 4survey (266 plots) and 5.32% of 6survey (709 plots), clustered by
   compartment rather than scattered (e.g. cpmt 2186: 12/37 flagged, 32%) — a compartment-level
   yldc-misassignment pattern, not isolated flukes.
   **Rerun sensitivity check, no retraining needed**: dropping the flagged plots from the
   already-saved predictions changes 4survey's pooled R2 negligibly (-0.002), but makes 6survey's
   *worse*, not better (EN -0.027, XGBoost -0.037, pushing XGBoost's R2 negative) — these extreme-
   target plots are propping up what little signal 6survey's models have. **Conclusion: exclusion
   is not a fix and shouldn't be applied** — keeping all plots in is the evidence-backed choice, not
   an oversight. A principled target-construction fix (compartment-level fallback `y_max`) is a
   legitimate future-work item, not something this evidence calls for doing now.
   **Plot idea**: histogram of `y_max_fit` (or `local_y_max_difference`) across the full population
   with the Tukey fences marked, showing the 266/709 flagged plots as a visibly distinct tail —
   directly visualises "this is a real, population-level pattern," a general-hypothesis plot rather
   than an outlier anecdote.
8. **Outlier typology: cross-model agreement and sub-compartment spatial clustering — computed
   2026-08-15** (`TEMP_rq3_outlier_diagnosis`). *General/population-level — also concerns the full
   266/709-plot flagged set, a distinct diagnostic axis from item 7's scan.* All 3 models
   (EN/XGBoost/GNNWR) independently find these same flagged plots hard (~36x over-representation in
   each model's own worst-1% residual bucket) — real, model-agnostic difficulty, not an XGBoost
   quirk. **Disturbance cross-reference re-checked at this larger, statistically real n=266 scale**
   (the original item 5 check was n=10, underpowered): still 0/266 `any_clearfell_like`, 0/266
   `any_measurement_inconsistent`, only 2/266 `any_ambiguous_disturbance` — confirms item 5's "not
   disturbance" finding holds at scale, not just for the original hand-picked plots. Flagged plots
   cluster spatially WITHIN their compartment (smaller mean distance to their own centroid than the
   compartment as a whole, every one of the 6 most-affected compartments checked) — evidence for a
   localized event/sub-stand condition, not a whole-compartment clerical error.
9. **Wind exposure, storm-year signature, and the known 2008 survey-boundary artifact — computed
   2026-08-15** (`TEMP_rq3_outlier_diagnosis`). *Specific/mechanism-test — the narrowest-scope item
   in this thread, testing two named candidate causes rather than characterising the population.*
   Wind exposure does NOT support a windthrow story (flagged plots are slightly more sheltered, not
   more exposed). **A known, already-documented data-quality issue is a real but minor
   contributor**: 75.6% of flagged plots have their worst residual in the 2008 survey (vs. 47.8%
   population) — ties to the already-known 2006->2008 survey-boundary re-delineation artifact.
   Refitting without each plot's 2008 point moves 89% closer to `y_max_yldc`, but only ~6% on
   average — real, but not the dominant driver.
   **Plot idea**: spaghetti plot of a random ~30-50 plot sample's Age-vs-height trajectories (grey)
   against the yldc benchmark curve, one normal example highlighted, next to a small-multiples panel
   using the categories established across items 5/6/9 (smooth-but-offset / genuinely unstable /
   2008-influenced) rather than examples picked by eye. Note the split role: the grey-spaghetti panel
   is a general-hypothesis visual (population-level, not outlier-specific); the small-multiples panel
   is the specific/typology visual — pair them so the figure covers both registers, not just the
   more visually "interesting" outlier cases.
10. **The Moran's I contradiction** (`TEMP_rq3_category_attribution`). Demoted below the primary
   GNNWR/outlier findings — this is a single spatial_block split, Set4/4survey-only side-analysis,
   not the core comparison — but still the best-suited item for a real expectation-vs-evidence
   discussion: the working assumption was "removing a category that genuinely explains local
   deviation should raise residual spatial clustering." Removing `terrain` — the largest real
   contributor besides stand_structure — instead *lowers* clustering (0.153 → 0.042), the opposite
   direction. Structure to use: state the expectation, show the diagnostic, name the contradiction,
   offer competing explanations (VIF screening changed what "terrain" even captures? spatial
   structure was never really carried by terrain in the first place?), and don't force a resolution
   if the evidence doesn't support one.
11. **Category attribution rerun changed after VIF screening** (`TEMP_rq3_category_attribution`).
    The earlier "every category removal improves R2" pattern disappeared once VIF screening fixed
    collinearity — connects directly to item 10's contradiction, since VIF screening is also the
    most likely explanation for why `terrain`'s Moran's I behaviour looks different from expected.

---

## Methodology → Results content migration (flagged during methodology review, 2026-08-15)

Moved here from `TEMP_results/README.md` (2026-08-15) since it cross-references this primer's own
item numbers directly — kept alongside the primer rather than in the raw-results index. Specific
sentences identified as living in the wrong chapter — methodology currently states them as
findings, but they're results. Logged here so they land in the right subsection when the results
chapter is actually drafted, not lost between chats.

| Sentence/claim | Target location | Status |
|---|---|---|
| "Testing four alternative network sizes confirmed that this shared design performs best... guided models proving the least sensitive" | Results, **RQ1 architecture-sweep subsection** (primer item 4 above) | Matches the real finding: PINN/PINN_k's architecture-sensitivity spread is 5-10x *smaller* than DNN's — "guided models least sensitive" is accurate, not just DNN robustness. Source: `TEMP_rq1_architecture_sweep_results_2026-08-13.tex`. |
| "Because no checked plots triggered these exclusion criteria, all were retained as genuine empirical departures" (flagged `\fc{check this is true}`) | Results, **RQ3 outlier-diagnosis subsection** (primer item 5 above) | **`\fc{}` resolved — verified true.** 0 of 10 outlier plots trigger any disturbance/measurement exclusion flag, and 0 of 10 clear the top-1% trajectory-instability cutoff either. Re-checked at the larger, statistically real n=266 scale (item 8's typology section) — still 0/266 clearfell-like, 0/266 measurement-inconsistent, only 2/266 ambiguous-disturbance. Source: `TEMP_rq3_outlier_diagnosis_results_2026-08-12.tex`'s "Outlier typology" section. Safe to state as fact, drop the flag. |
| GNNWR 6survey sign-flip limitations bullet: "centred near zero... underpowered... should be read as inconclusive" | Results/Discussion, **RQ3 GNNWR subsection** (primer item 2 above) | Matches `TEMP_rq3_gnnwr_results_2026-08-11.tex`'s reseed section exactly. **Split**: methodology keeps only the general "small-cohort split/seed instability" caveat (applies broadly, e.g. also motivates RQ2b's 4survey-only scope); this specific sign-flip finding and its numbers belong in results only, not duplicated in methodology. |

**Pattern to watch for**: methodology sections that state a specific numeric outcome ("X performed
best," "no plots were excluded," "the result was inconclusive") are usually results content that
drifted upstream during drafting — methodology should describe the *procedure* checked, results
should report *what the check found*. Worth a pass over the rest of the methodology draft for the
same pattern before it's finalised.



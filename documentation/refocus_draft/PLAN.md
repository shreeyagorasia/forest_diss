# Refocus plan — outline, not prose

## Results chapter rewrite — locked decisions (2026-08-20)

All below confirmed by user, execute against these, don't re-litigate:

1. **RQ2a placement**: whole unit (prediction comparison + row-level curve-departure bridge)
   moves to Q3 intact, positioned last, supplementary. Not split.
2. **Scope**: reorganize existing prose into Q1->Q2->Q3 order. Don't rebuild from scratch --
   current text is already fact-correct (Moran's I already fixed, numbers already verified).
3. **Depth**: trim to essentials. Cut what doesn't change or defend a claim actually being made.
4. **Balance**: Q1 + Q2 dominate chapter length. Q3 short, supplementary.
5. **Figures**: mixed effort. Build the already-flagged quick wins (R3-3's real ranks swapped
   in, R2b-1's Moran's I axis rescale) since the code exists and works. Everything else stays
   a `\fig{}` placeholder description, no execution attempt tonight.
6. **Ablation placeholder**: literal gap, one sentence stating what will go there and why. No
   guessed numbers anywhere, even tagged.
7. **Per-finding template, applied uniformly**: comparison/number -> why it happened -> what it
   means for the Q. Every major finding gets this shape.
8. **Cut vs keep**: drop secondary diagnostics that are purely confirmatory (candidates: the
   266-plot population-scale check, the 2008 survey-boundary artifact) unless they're actually
   load-bearing for a claim being made -- judge each on that basis, not "everything reordered."
9. **Q3/PINN depth**: trim to essentials -- headline table, XGBoost-vs-DNN result, PINN
   forward-pass limitation. Cut split-leakage detail and the cohort-sensitivity paragraph
   unless load-bearing.
10. **Methodology cross-references**: write Results assuming Methodology's sections get renamed
    to match Q1/Q2/Q3 later. Creates a known follow-up task -- Methodology still uses
    RQ2a/RQ2b/RQ3 internally right now, not yet touched.

## Results chapter rewrite — DONE (2026-08-20, this pass)

Executed against the locked decisions above. Structural check clean: no duplicate labels, no
new dangling refs (only pre-existing unresolved appendix stubs remain).

- Renamed: `RQ2` -> `Q1: Persistent curve-departure attribution`, `RQ3` -> `Q2: Spatial
  attribution`, `RQ2a` -> `Q3: Prediction and physics guidance`. All prose mentions of old
  RQ-numbering inside Results batch-renamed to match (labels/refs left untouched, only visible
  text and section titles changed).
- Chapter intro paragraph rewritten to state the Q1->Q2->Q3 argument directly.
- Row-level curve-departure bridge relocated from Q1's opening to Q3 (per the locked decision),
  now sitting after the XGBoost-vs-DNN discussion as "Does environmental conditioning correct
  the curve at row level?" -- kept intact, not trimmed.
- Q1's multicollinearity/spatial-confounding block: unchanged (already tight, was reviewed and
  judged load-bearing, not cut).
- Q3 trimmed: cut the "Cohort sensitivity" paragraph entirely (not Q3-specific, applies
  dissertation-wide -- candidate for a footnote in Data instead, not done yet); compressed
  "Effect of the evaluation split" from 2 paragraphs to 1; compressed the physics-weight
  ablation's speculative second paragraph to one sentence. Reattached the
  `sec:results-rq1-cr-reduction` label here (Methodology still points to it) after the original
  location was cut.
- Reference-density ablation: converted from an `\ai{}` proposal to a literal `[PENDING ...]`
  gap, no numbers guessed, states exactly what will land there (8 cluster jobs, still running).
- Floating-growth-rate-refit `\ai{}` proposal: deleted -- matches the "parked, out of scope"
  status already recorded above.
- 266-plot / 2008-artifact paragraph: compressed (kept the load-bearing claims -- scale,
  compartment clustering, removing plots hurts not helps, zero disturbance-related, 2008
  artifact's real-but-minor share -- cut the "dropping 2008 moves 89% closer by 6%" detail and
  the windthrow-ruling-out sentence, both non-load-bearing).

**Not yet done this pass**: the two quick-win figures (R3-3 real ranks, R2b-1 axis fix) --
those live in the notebooks, not this `.tex`, still pending. Q2's own body text (the GNNWR
edge, category attribution, local-coefficient analysis) was left as-is -- already tight, matched
the locked decisions without requiring cuts.

## RULE: PINN mentions — where allowed, where not

PINN only appears in its dedicated Q3 method section, and in Results/Discussion when Q3 is
being discussed. Not in Motivation, not in the research-gap framing, not scattered elsewhere.
This is the whole reason for the refocus draft — keep it out of anywhere that isn't Q3 itself.

## URGENT — full dissertation due tomorrow

Everything below this line is re-prioritized around that. Rule, corrected 2026-08-20: **not
"no more running" — running is fine if it directly serves the argument being built.** The
earlier version of this rule ("narrative beats polish, no more notebook execution") overcorrected.
Right test for any further run/analysis: does it change or strengthen a specific claim in the
Q1/Q2/Q3 story, not "is it interesting" or "is it easy." Quick wins that pass this test (real
ranks into R3-3, the axis fix, extending the local-coefficient table) are worth doing. Anything
that's exploratory-only or doesn't sharpen a claim already being made isn't, purely on time
grounds tonight.

**Two gaps found that are more urgent than anything else on this list**: `\abstract{TODO: FILL
IN}` (Ch1, literally unwritten) and Ch6 Conclusion is an empty stub (`\chapter{Conclusion and
Future Work}` with no body at all). A dissertation missing these outright is a bigger problem
than any ordering/consistency issue below. Flagging for a priority call, not deciding
unilaterally.

**Revised priority order for remaining time**:
1. Abstract + Conclusion — currently *empty*, not just imperfect
2. Results narrative — nearly done (this session's main work)
3. Methodology/Background consistency passes (PINN trim, Q1/Q2/Q3 reorder) — polish, do if time
4. Figures — only if everything above is done and time remains


`19th_1038_rq2brq3_refocus.tex` = full redraft. Complete document, every chapter present,
compiles standalone. Not a diff/patch file.

This file = the plan behind it. Status tracker + what's left, chapter by chapter.

---

## Q1 / Q2 / Q3 — clean content labels, replacing RQ1/RQ2a/RQ2b/RQ3

Old numbering (RQ1, RQ2a, RQ2b, RQ3) carries history and doesn't match story order any more.
Fresh labels, defined by content, ordered as they'll actually appear in the redraft:

- **Q1 — Persistent curve-departure attribution.** Which environmental/management conditions
  associate with a plot's whole-history departure from the shared growth curve, and do
  independent model families (Elastic Net, XGBoost, LMM) agree on which ones. = old RQ2b.
- **Q2 — Spatial attribution.** Does letting that environment-deviation relationship vary
  spatially (GNNWR) capture structure the global Q1 models leave unexplained. = old RQ3.
- **Q3 — Prediction and physics guidance.** Does flexible ML / a physics-informed architecture
  predict top height better than the shared curve alone, and does the attribution signal found
  in Q1/Q2 translate into a predictive advantage. = old RQ1 / RQ2a.

**Order: Q1 -> Q2 -> Q3, in every chapter, not just Results.** Rationale: Q1 establishes real
but incomplete attribution and leaves real spatial structure unexplained -> Q2 tests whether
spatial variation resolves that -> Q3 closes by asking whether any of it pays off as
prediction, PINN's limitation included honestly. This is the "best story" order if writing
fresh, not just a patch on the old structure.

**Consistency implication, not yet executed**: Q1/Q2/Q3 both targets are CR-residuals, and Q3
uses the CR curve as a baseline model — so Chapman-Richards curve-fitting is genuinely shared
prerequisite infrastructure for all three, not "part of" any one Q. Proposal: pull it out as
its own short Methodology subsection *before* Q1 begins, then present Q1/Q2/Q3 methodology in
that same order — so Background, Methodology, Objectives table, and Results all use the
identical Q1/Q2/Q3 order throughout, not just Results (which is where the reorder currently
only lives, per the section below). This is a bigger change than what's been executed so far
(only Results has been reordered) — needs sign-off before touching Methodology/Background/Ch1
again.

---

## What's already fillable — the argument can be built end to end with zero further training runs

User confirmed: no more cluster runs planned beyond the 8 already-submitted reference-density
jobs. Everything below needs no new training, cluster or otherwise — either already computed,
or a cheap local analysis pass on data already sitting on disk.

**Q1 (persistent attribution) — 100% fillable now.** EN/XGBoost/LMM coefficient table,
corrected Moran's I, CanopyCover reverse-causation check (honestly unresolved), `topex`
instability (EN vs LMM sign flip) — all done, all solid.

**Q2 (spatial attribution) — 100% fillable now for the headline argument**, plus optional
cheap extras. Solid and done: EN/XGBoost/GNNWR headline table + bootstrap CIs, corrected
Moran's I (4survey edge stronger than first thought, 6survey now "uninformative" not "wrong
direction"), category attribution, GNNWR local-coefficient analysis on Set4/4survey (the
`topex` corroboration with Q1), outlier-plot fixed-rate mechanism, the 266-plot population
check, GNNWR's own hardware/implementation constraints disclosed. **Bonus, not a dependency**:
the 8 reference-density jobs — build the Q2 argument so it stands complete without them; if
they land, they strengthen (or complicate) the 6survey account, they don't create it.
**Optional, cheap, no training needed** (post-hoc analysis of data already saved to disk):
extending the local-coefficient table to Set2/Set3 and 6survey; the EN/XGBoost-SHAP/GNNWR
cross-method Spearman rank correlation. Both are pandas passes over existing CSVs, not new
runs — worth doing if there's write time, genuinely optional if not.
**Parked, needs new code + a run, skip given no more training**: `compartment_mixed_dnn`
redo, the floating-growth-rate refit, the CanopyCover lagged/temporal-precedence check. All
three were "would strengthen this, not required" when proposed — treat as out of scope now.

**Q3 (prediction and physics guidance) — 100% fillable now, nothing pending.**
CR/Linear/RF/XGBoost/DNN/PINN/PINN-k table, the XGBoost-vs-DNN ablation (5 causes ruled out),
the PINN forward-pass limitation (found-not-designed), the physics-weight ablation, the
random-vs-spatial-split leakage finding, cohort sensitivity, and the row-level
curve-departure-correction bridge (environmental conditioning partially, not fully, corrects
the worst-fitting rows, permutation-confirmed) — this last one is the actual evidence for
Q3's closing question ("does the Q1/Q2 signal help prediction"), and it's already done.

**Bottom line**: the full Q1 -> Q2 -> Q3 argument is already buildable from what's on disk.
Nothing in the core story depends on a run that hasn't happened yet.

---

## Status: what we have (real, computed, not fabricated)

**RQ2a (prediction + curve-departure correction)**
- Table: CR/Linear/RF/XGBoost/DNN/PINN/PINN-k, 4survey + 6survey, spatial-block CV — done
- XGBoost-vs-DNN ablation block (5 ruled-out causes) — done, solid, untouched by PINN bug
- PINN forward-pass limitation disclosed as found-not-designed — done
- Row-level curve-departure bridge (was RQ2a's own subsection, now 1 paragraph) — done
- Temporal holdout — dropped entirely (results + methodology + Ch1 objective)

**RQ2b (persistent departure attribution)**
- EN / XGBoost / LMM (was "NLME" — wrong name, fixed) coefficients table — done
- Moran's I — corrected to semivariogram method (was stale k=8 numbers, now fixed everywhere)
- CanopyCover reverse-causation check (baseline-only R2, drop-CanopyCover ablation) — done,
  explicitly unresolved, treated as baseline control not attribution finding
- topex instability (EN vs LMM sign flip, spatial confounding partial account) — done

**RQ3 (spatial attribution)**
- EN / XGBoost / GNNWR headline table, Set2-4, both cohorts, bootstrap CIs — done
- Moran's I — corrected (was stale, 4survey edge now stronger, 6survey now "no structure"
  not "wrong direction") — done
- Category attribution (terrain/wind/soil/climate/edge/stand) — done, re-verified correct method
- GNNWR local-coefficient analysis (CanopyCover stable+spatial, topex unstable, corroborates
  RQ2b's own topex finding) — done, first time anyone looked at these columns
- Outlier-plot mechanism (fixed-rate forces inflated asymptote, traced on 5 plots) — done
- Population-scale check (266 flagged plots, 2008 artifact) — done
- GNNWR implementation constraints (hardware, hat-matrix, reference-set cap) disclosed — done
- GNNWR background: kernel/bandwidth theory, named variants (mixed/multiscale GWR, GTWR/GTNNWR),
  why GNNWR not alternatives — done

**Figures**: not built as images, but the *code* already exists and is mostly real (checked
2026-08-20, not previously known to this plan):
- `notebooks/results_figures_rq2b.ipynb` (Q1) — Figure R2b-1, 3 panels (EN vs LMM coefficient
  forest plot / Moran's I by set / Set4 residual map). Live-computed, not fabricated. **Needs
  one fix before running**: Panel B's y-axis is hardcoded `ax_b.set_ylim(0, 0.8)`, calibrated
  for the old stale k=8 Moran's I range — corrected values are 0.03-0.15, so this would plot
  everything squashed near zero. Rescale to roughly `(0, 0.2)` before running.
- `notebooks/results_figures_rq3.ipynb` (Q2) — 3 figures. R3-1 (GNNWR edge + Moran's I both
  cohorts + local CanopyCover map): Panels A/B transcribed from TEMP files but the notebook's
  own comment confirms they're already the corrected 2026-08-19 semivariogram numbers, not
  stale; Panel C live-computed. R3-2 (deviation map + compartment archetype map + 4 real
  representative trajectories, using the actual cross-model outlier crossref CSV, not a proxy
  rule): fully live. R3-3 (cross-method CanopyCover rank agreement slopegraph): **still
  placeholder data**, explicitly flagged by its own author-comment as needing the real ranks
  swapped in from `TEMP_rq3_en_xgb_results_2026-08-11.tex` / `TEMP_rq3_gnnwr_local_coef_rank_2026-08-16.tex`
  (now under `documentation/refocus_draft/` sibling `TEMP_results_attribution/`).
- Do not edit these notebooks in place — copy to new files if actually building images, per
  standing instruction not to touch anything that might need to be reverted to.
- Given the deadline, running these is optional/last-priority (see URGENT section above), but
  the fact they already exist and are mostly real is worth knowing — much cheaper to finish
  than to build from scratch if time allows.

**Quick wins — genuinely cheap, worth doing if any writing time is left over**:
- R3-3's placeholder ranks -> real ranks: not even a new computation, just correct transcription
  from two files that already have the numbers.
- R2b-1's Panel B axis fix (one line).
- Local-coefficient table extension to Set2/Set3/6survey, and the EN/XGBoost-SHAP/GNNWR Spearman
  rank correlation (both already flagged above as free, no-training-needed extras).

---

## New analysis — status of each idea raised

| Idea | Status |
|---|---|
| Reference-density ablation (does capping 4survey to ~6survey's scale reproduce instability) | **submitted to cluster** (A6000, 8 jobs, 7500/13000/20000/full x Set3/Set4), pending |
| GNNWR local-coefficient analysis | done (Set4/4survey only so far; Set2/Set3 + 6survey not yet) |
| Category-attribution Moran's I method | verified correct, no caveat needed |
| Cross-method rank correlation (EN/XGB-SHAP/GNNWR-coef, Spearman) | proposed, not run |
| Floating-growth-rate refit on outlier plots | proposed, not run, cheap (no cluster) |
| `compartment_mixed_dnn` vs GNNWR (blocky vs smooth) | **parked, redo if time allows** — evaluation protocol flaw found: test compartments never get a random intercept under spatial-block CV, so the comparison never actually tests the random effect. ICC finding itself (0.399/0.188) still valid, kept as motivating context only. Fix if revisited: impute each held-out test compartment's intercept as a distance-weighted average of nearby *training* compartments' shrunk intercepts (needs new code in `compartment_mixed_dnn_check.py`) — that's what would make it a fair blocky-vs-smooth test under the same spatial-block generalization regime GNNWR is evaluated under. |
| CanopyCover lagged/temporal-precedence check (does earlier-survey CanopyCover predict later growth) | proposed, not run, real repeated-survey data supports it, would need new code |

---

## Results chapter — story and order (locked in)

Not three independent boxes. One argument: global attribution models leave real spatial
structure unexplained no matter how much environmental data they get (RQ2b) -> motivates
testing whether letting the relationship itself vary spatially does better (RQ3) -> RQ2a
closes by asking whether any of that attribution signal actually helps prediction.

**Order: RQ2b -> RQ3 -> RQ2a.** RQ2a (prediction baselines, XGBoost-wins, PINN
comparison/limitation) moved to the END of the chapter — reframed as the payoff/validation
question ("does the attribution signal we found translate into better prediction"), not
throat-clearing before the real content. Executed in the .tex already (section moved,
renamed RQ1 -> RQ2a, new framing paragraph added, "Answer to RQ2a" paragraph's closing
sentence rewritten to point back at RQ2b/RQ3 instead of forward). Methodology chapter order
does NOT need to match — it can keep the logical build order (CR baseline -> RQ2a -> RQ2b ->
RQ3) since RQ2b/RQ3's methodology already depends on CR-curve concepts defined early in Ch4,
not on RQ2a's Results.

Cross-cutting threads to keep visible across RQ2b/RQ3, not scattered as isolated footnotes:
- Model families disagree in informative ways, not just noise: `topex` flips sign EN vs LMM
  (RQ2b) AND has by far the least stable local coefficient in GNNWR (RQ3) — three
  independent methods, same instability. State once, reference twice, not three unrelated
  observations.
- Distinguish real signal from artifact, stated as a stance early, cashed in twice: CanopyCover's
  reverse-causation ambiguity (RQ2b, honestly unresolved) and the outlier plots' fixed-rate
  asymptote artifact (RQ3, mechanically traced, not ecological).
- 6survey is underpowered for spatial questions specifically, not "broken" — every diagnostic
  (R2 sign instability, semivariogram failing to resolve, possibly the pending reference-density
  ablation) points at the same root cause (too few compartments). Say once, don't re-litigate.

Still open / needs a decision later: does the terrain-removal Moran's I anomaly (removing
terrain LOWERS clustering) get folded into the "disagreement is informative" thread above, or
stay a standalone open question? Leaning toward folding in once we write the actual prose.

## Methodology facts — the "how", for writing the new Methodology chapter

Concrete implementation details found by reading the actual code this session, not from the
existing draft prose. Organised by Q, "how it was actually computed" only.

**Shared, applies everywhere**
- Spatial-block CV: whole compartments assigned to folds; 60m buffer removes near-boundary
  *training* points only (val/test plots never removed); 5 folds, rotating test/val/train.
  Random split (separate, used only to show the leakage effect): 60/20/20, seed 42.
- Bootstrap CIs: 1,000 resamples of *whole compartments* (cluster bootstrap), not individual
  rows — preserves spatial clustering in the CI itself.
- Moran's I: semivariogram-informed distance-band method, 999 permutations, search window
  5,000m widened to 10,000m if unresolved. Replaced an earlier k=8-nearest-neighbour method
  project-wide — k=8 on this project's dense ~20m plot grid tests near-immediate-neighbour
  similarity, not real regional clustering; deliberate change, not a bug fix.

**Q1 (LMM / persistent attribution)**
- Two-stage: Stage 1 = pooled Chapman-Richards curve fit once (fixed, frozen). Stage 2 = model
  the residual as `statsmodels.MixedLM.from_formula`: fixed effects (standardised, mean 0 sd 1,
  every column) + one random intercept per compartment, no random slopes.
- Random-effect distribution: Normal by construction (`u_c ~ N(0, sigma_u^2)`, residual
  `~N(0, sigma^2)`) — not a configurable choice, inherent to what `MixedLM` is. Checked
  directly against the data (`mean_cr_residual`: skew -0.48, excess kurtosis 0.74 — moderate,
  not severe); GAMLSS considered and rejected (thin Python tooling relative to the size of the
  violation). Consequence disclosed: coefficients treated as robust to this, standard
  errors/p-values treated as approximate, cross-checked against XGBoost's distribution-free
  refit-ablation for the same variable rather than trusted alone.
- `reml=True` for final coefficient reporting (standard for one already-decided model);
  `reml=False` specifically when comparing variance components between models with *different*
  fixed-effects sets (null vs. full) — REML profiles depend on the fixed-effects design matrix,
  so comparing REML variance estimates across differing fixed-effects structures is invalid
  (textbook Pinheiro & Bates guidance, the reference behind R's own `nlme`).
  "Spatial variance explained" = proportional reduction in the compartment random-intercept
  variance (`result.cov_re.iloc[0,0]`) from a null (intercept-only) model to the full model.
- Elastic Net: nested CV for the penalty/`l1_ratio`.
- XGBoost fixed config for Q1/Q2 (not tuned per-target): `n_estimators=500, max_depth=4,
  learning_rate=0.04`, real early stopping against a genuine validation split. History worth
  knowing: this specific config traces back to a full-population SHAP-interpretation fit
  (`explain_signal.py`) that itself was never the product of a tuning search — a reasonable
  hand-picked config that got reused project-wide, not independently validated per use.
- CanopyCover check: same LiDAR flight/processing step as the height target itself; Pearson
  correlation with all 3 of this project's own targets 0.35-0.47 (moderate — rules out literal
  duplication, does not resolve causal direction).

**Q2 (GNNWR / spatial attribution)**
- GNNWR headline results use the FULL training population as reference points
  (`reference_set_size=0`), matching what EN/XGBoost/DNN also see — not a capped/handicapped
  version. Getting there took two separate, disclosed hardware fixes: (1) the spatial-weighting
  sub-network's first layer width equals the reference-set size (not a fixed constant like a
  normal MLP), OOM'd a 10.57GiB generic GPU at full scale — moved to a bigger card (H200 MIG
  slice, later A6000) rather than shrinking the reference set; (2) a *separate* bottleneck in
  gnnwr's own diagnostics class (a classical-GWR hat matrix, O(n^2 x features), used only for a
  cosmetic training-time AIC number) — patched to skip above a row threshold; confirmed this
  never touches the gradient step or the reported R2/RMSE, only AIC/F-test numbers (now
  approximate/unavailable, disclosed).
- Batch size 32 (vs. the DNN's 256) — one of two reasons (with the reference-set-width point
  above) GNNWR trains far slower than every other model in this project; worth stating plainly
  in Methodology rather than leaving as an unexplained runtime difference.
- Category attribution: grouped permutation importance + before/after category-removal Moran's
  I, single spatial-block split (not pooled 5-fold — 7 refits per call already), Set4/4survey
  only (the only tier with at least one column from every category).
- Local (`coef_<variable>`) columns are saved automatically per plot per fold by the existing
  pipeline — not a new computation, just previously unanalysed output.
- Outlier/target-construction mechanism: closed-form weighted least-squares through the origin,
  `y_max_hat = sum(height x shape_term) / sum(shape_term^2)`, `shape_term =
  (1-exp(-p4*Age))^p5`, with `p4`/`p5` held FIXED at the plot's yield-class value — this fixed-
  rate assumption is mechanically why a plot whose real trajectory is flatter/slower than its
  assigned yield class gets forced into an inflated ceiling. Tukey fence: 1.5xIQR, computed
  independently per cohort.

**Q3 (prediction, incl. PINN)**
- PINN forward-pass limitation, precisely: `y_max^(i)`/`k^(i)` are computed by the auxiliary
  sub-networks but the trunk's `forward()` never evaluates the CR equation with them — they
  only enter the physics/trajectory loss terms, never the prediction itself. Confirmed by
  reading the code directly, not inferred from results.
- XGBoost tuning (Q3 only, not Q1/Q2's shared fixed config): 27 configs x 5 folds x 2 cohorts =
  270 fits, model selection strictly on validation R2, test sets untouched until final eval.

## Chapter-by-chapter plan (working backwards from results)

**1. Results (Ch5) — mostly done**, remaining:
- Build actual figures (currently all placeholders)
- Swap in reference-density ablation numbers once cluster jobs land
- Decide: run floating-rate refit / cross-method correlation, or leave as noted-not-run

**2. Methodology (Ch4) — mostly done**, remaining:
- Confirm RQ2a/RQ2b/RQ3 section headers match new Objectives table exactly
- Strip PINN loss-function apparatus down to factual mention (currently still has full
  equations/ablation description — this hasn't been trimmed yet, only Results/Discussion has)

**3. Data (Ch3) — should barely change**. Nothing about the refocus touches cohort construction,
target definition, or environmental data sources. Skim only, no rewrite expected.

**4. Background (Ch2) — partially done**:
- PINN section (§2.2) — already shrunk once (cut integration-strategies table). May shrink
  further once Methodology's PINN section is trimmed, to match.
- GNNWR/spatial section (§2.3) — already expanded (kernel/bandwidth, named variants,
  alternatives ruled out)
- **Not yet done**: reorder so attribution-methods theory (LMM, GNNWR) comes before/gets more
  weight than the prediction-methods theory (DNN/PINN) — currently still in original order
  (prediction theory §2.2 before attribution theory §2.3)
- **Not yet done**: LMM citation (Pinheiro & Bates or equivalent) — currently cites nothing for
  the mixed-model choice specifically

**5. Intro (Ch1) — in progress, 2026-08-20**:
- Objectives/RQ table — **re-done**: relabelled Q1 (persistent attribution) / Q2 (spatial
  attribution) / Q3 (prediction + physics guidance), matching the Q1->Q2->Q3 story order.
  Dropped the standalone "Dataset construction" row (folded into one prose sentence instead) —
  3 rows now, not 4.
- Motivation section (§1.1) — being rewritten: para 3's closing sentence and para 5 (research
  gap) reordered attribution-first; para 4 (DNN/PINN) cut from a full paragraph to two
  sentences, PINN reduced to "tested as one baseline, discussed not centred."
- **Still not done**: dissertation title unresolved ("Physics-informed prediction and spatial
  attribution..." no longer matches the actual center of gravity) — flagged, not decided.

**6. Conclusion (Ch6) — stub, not started**. PINN lessons-learned content (scalability/
efficiency claims flagged as unmeasured, CR-vs-data loss tension measured, forward-pass
limitation as fragility lesson) sketched in conversation, not yet written into the chapter.

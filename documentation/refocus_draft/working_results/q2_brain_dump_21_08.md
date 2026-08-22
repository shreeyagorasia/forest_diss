# Q2 brain dump — what's actually true, what's shaky, what's new

Written 2026-08-21, updated 2026-08-22. Plain language, for you to read and decide what's
convincing before any of this goes in the dissertation. Marked **NEW** where I ran a real test
this pass that wasn't in the draft before, **UPDATE (08-22)** where the 08-22 pass changed or
corrected an 08-21 finding. Everything else is re-checked against saved files, not re-derived from
scratch.

**Queued, partially back** (submitted to the cluster 2026-08-22): GNNWR CanopyCover-only is 3/5
folds back (partial result in section 1 below, not final); the 2-seed reseed check and the
saturation-transform check are still fully pending. Nothing below treats the partial CanopyCover-
only result as final, or assumes anything about the still-pending experiments. Come back to this
file once the rest land.

## The story Q2 is supposed to tell

Q1 found: global models (same formula everywhere) leave real spatial clustering in their errors.
Q2 asks: does letting the model change *by location* (GNNWR) pick up what the global models miss?
And separately: of all the environmental variables, which ones actually drive the plot-to-plot
gap between a stand's own growth ceiling and what its yield class assumes ($\Delta y_{max}$)?

Two sub-stories, and they're not equally solid. The "does location-aware help" story is weak but
honest (small edge, overlapping confidence intervals, corroborated two ways). The "what drives it"
story turns out to hinge overwhelmingly on one variable — more than anyone had written down yet.

---

## Headline story map — re-audited 08-22, after everything above

The chapter has accumulated a lot of findings across many passes. This is the reflection pass: what
should the actual sequence of headline claims be, now that every earlier finding has to survive
contact with the later, sharper ones? Ordered as they should appear in the Discussion, each with its
evidence anchor and, where it exists, its Q1 link. Wording strength matches verdict (clear answer /
suggestive / ruled out / no answer yet), per the review framework.

**0. $\Delta y_{max}$ is not a clean ceiling measurement — it is entangled with growth-rate mismatch
across the whole population, not just its outliers (section 4).** Comes first because everything
downstream depends on it: the curve-fitting procedure holds each plot's growth rate fixed to its
yield class, so a rate mismatch leaks into the fitted ceiling instead of disappearing. Evidence:
r=0.51 Pearson across all ~56,000 plots, confirmed not a tail-only artifact (0.509→0.525 excluding
flagged plots). Clear answer, not suggestive — this one is solid.

**1. GNNWR outperforms both global baselines on every 4-survey set by point estimate, but the
advantage is not statistically distinguishable — confidence intervals overlap in every set.**
Unchanged from the original draft, still the right opening empirical claim once the target caveat
is on the table. Suggestive, not confirming — CIs overlap, said plainly.

**2. GNNWR's advantage is entirely explained by CanopyCover, not a general ability to capture
spatially-varying environmental relationships (section 1, Test B).** The single most important new
finding this pass, and it should be a headline of its own, not folded into a caveat: without
CanopyCover, GNNWR collapses to $R^2=0.049$ — statistically the same floor as EN (0.042) and
XGBoost (0.061). Clear answer, decisively tested (ablation), not inferred.

**3. Link to Q1 — the same 15 non-canopy environmental variables that added real signal in Q1 add
almost nothing in Q2, and the target explains why, not weaker biology (new synthesis, ties
sections 2 and 4 together).** Q1's own baseline-vs-full comparison (CanopyCover+thinning only vs.
Set4) shows +0.138/+0.173 R² from the 15 terrain/climate/soil variables. The identical comparison
for Q2 (computed this pass: baseline EN=0.201/XGB=0.209 vs. full Set4 EN=0.240/XGB=0.250) shows
only +0.039/+0.041 — roughly a quarter of Q1's gain. This isn't the variables mattering less
biologically; headline 0 explains it structurally: Q2's target is far noisier, so there's less
clean signal left for anything but the one variable (CanopyCover) most directly tied to the real
LiDAR-derived structure. This is exactly the kind of cross-chapter connective claim worth stating
explicitly rather than leaving Q1 and Q2 as two disconnected attribution stories.

**4. Terrain variables show a real, if modest, non-linear association with $\Delta y_{max}$; soil
is indistinguishable from noise — but this entire ranking is read off the small remaining budget
after CanopyCover (sections 1's SHAP-dependence check + section 2).** Softened from the original
draft's plainer "terrain matters moderately" — now specifically says non-linear (windward_topex,
slope_degrees both show clean saturating SHAP-dependence curves, not noise) and specifically says
small budget (CanopyCover carries ~80-90% of the model's total signal). Both halves matter: real,
but small and shape-constrained by GNNWR's own structural linearity.

**5. Link to Q1 — where local coefficients can be compared, the stability pattern matches: topex is
the least stable terrain variable in both chapters, slope_degrees is consistently stable in both
(section 3).** GNNWR's own per-plot coefficient sign-stability (topex ~72%, slope ~95-97%) lines up
with Q1's EN/LMM sign-disagreement finding for the same two variables. Worth stating as
cross-chapter consistency, not coincidence — the same underlying variables behave the same way
regardless of which target (curve-residual or ceiling-difference) or which method (EN/LMM vs.
GNNWR) is used to look at them.

**6. The outlier story scales beyond a handful of plots, and it's the extreme tail of headline 0,
not a separate phenomenon (section 5).** 266 (4-survey, 0.47%) / 709 (6-survey, 5.32%) plots via
Tukey fence; the full-population split test found 71.6% fit the fixed-$k$ artifact signature
(exposed terrain), 28.4% look like genuinely fast, real growth (sheltered terrain). This is the
best-evidenced part of the chapter — say so, and say explicitly that it's headline 0's mechanism at
a smaller, more extreme scale, not a disconnected finding.

**7. 6-survey's collapse is acknowledged but not chased (section 8) — a disclosed limitation, not
a resolved one, and deliberately not a headline.** Sample size ruled out as the cause; true cause
(likely fewer compartments) left untested by your own call, since 6-survey carries no narrative
weight either way (R²≈0 there for all three methods already). One sentence in the chapter, not a
sub-story.

**Conclusion shape this suggests**: location-aware modelling helps, but only through the one
variable most directly tied to real observed structure (CanopyCover) — genuine environmental
heterogeneity, if it exists at all beyond that, is out of reach for every method tested here, for
two compounding and disclosed reasons: the target's own estimator noise (headline 0) and GNNWR's
structural linearity (section 1's two-linearities discussion). Neither is fully resolved, both are
tested and quantified rather than asserted, and the future-work section (7) says precisely what a
real fix would cost. This is a defensible, honest chapter conclusion — it just needs to say clearly
that it found the *limits* of spatial attribution here, not spatial attribution itself.

---

## Verdict map

| Claim | Verdict | Confidence after this pass |
|---|---|---|
| GNNWR beats EN/XGBoost on every 4-survey set (point estimate) | **True** | Confirmed, unchanged |
| ...but the edge is statistically solid (CIs don't overlap) | **False** | Confirmed false — CIs overlap everywhere |
| Moran's I drop backs up the R² edge | **True, but weak corroboration** | Still just "suggestive" — and now (UPDATE 08-22) likely reflects the same CanopyCover-specific effect as the R² edge, not a broader spatial-non-stationarity signal, since the ablation shows GNNWR has nothing else to draw on |
| GNNWR's edge = genuine spatially-varying environmental relationships | **Mostly false** | **UPDATE (08-22)**: GNNWR without CanopyCover scores 0.049 — statistically the same as EN/XGBoost without it. GNNWR's entire edge over global models is riding on CanopyCover specifically, not on environmental variables in general — see "why GNNWR wins" below |
| CanopyCover is the top variable, not circular (not just the target restated) | **True** | Confirmed, correlation 0.35–0.47 range (Q1); **NEW for Q2**: 0.47 |
| CanopyCover "matters a lot" for Q2 | **Understatement** | **NEW**: it's ~80% of the model's entire explanatory power, worse than Q1 — and this is now confirmed true for GNNWR too, not just EN/XGBoost |
| Terrain matters moderately, soil is noise | **True but needs a huge caveat** | Only true within the ~5% of R² left after CanopyCover — see below |
| GNNWR's linear-per-location structure is why it can't use terrain/climate signal | **Plausible, live — corrects an 08-21 overstatement** | **UPDATE (08-22)**: SHAP-dependence plots show real, clean, *non-linear* (saturating) structure for windward_topex and slope_degrees — my earlier "linearity probably isn't the blocker" read was premature, see below |
| $\Delta y_{max}$ cleanly measures "ceiling difference" | **False** | Confirmed: entangled with growth-rate mismatch across the whole population, not just outliers |
| 6-survey collapse = sample size | **False, ruled out** | Confirmed ruled out; true cause still unknown |
| The ~10/266/709-plot outlier story (fixed-$k$ artifact) | **True, well-tested** | Solid — this is the best-evidenced part of the chapter |
| 71.6%/28.4% split (artifact vs. real fast growth) | **True, tested on full flagged population** | Solid |
| Q1's "environmental variables add real signal beyond CanopyCover" finding transfers to Q2 | **False** | **NEW**: Q1's baseline→Set4 gain is +0.138/+0.173 R²; Q2's identical comparison (computed this pass) is only +0.039/+0.041 — about a quarter. Same variables, very different targets — doesn't transfer |
| Freeing $k$ per plot (no shrinkage) would fix the entangled-target problem | **False, tested directly** | **NEW**: 9x more implausible ceilings (42.4% vs. current 4.6%), no R² gain. Shrinkage, not free fit, is the only version worth pursuing |

---

## 1. Why does GNNWR actually win? (NEW investigation, UPDATE 08-22: now a decisive answer)

The draft says GNNWR's R² edge is "corroborated but not confirmed" by its bigger Moran's I drop,
with an honest caveat that this doesn't prove genuine spatial non-stationarity. I wanted a sharper
answer than "maybe." Two tests, run a day apart — the second one settles it.

**Test A (08-21, diagnostic only)**: if GNNWR is mostly just learning "this compartment tends to
run high/low" — a location-level *average* effect, similar to what a compartment random intercept
would do — then most of its edge should be recoverable just by shifting each compartment's
predictions by that compartment's own average error. Took Elastic Net's actual Set4 predictions
and computed what $R^2$ would be with a "cheating" per-compartment correction (uses test-set
answers directly — an upper bound, not an achievable score):
- EN's real Set4 $R^2$: 0.240
- Diagnostic ceiling (perfect compartment-average correction): **0.538** — more than double
- GNNWR's real Set4 $R^2$: 0.294 — only about a fifth of the way from 0.240 to that ceiling

This established there's a lot of real compartment-level structure sitting unexplained, and that
GNNWR only captures a modest slice of it — but it couldn't say *what* GNNWR's slice actually is.

**Test B (08-22, decisive)**: you ran the CanopyCover-dropped GNNWR ablation on the cluster and
rsynced it back. Result: **GNNWR without CanopyCover scores $R^2=0.049$** (fold-mean
0.026±0.067) — statistically the same as EN (0.042) and XGBoost (0.061) without CanopyCover. All
three collapse to the same near-zero floor.

This is the direct answer Test A couldn't give: **GNNWR's entire edge over the global models is
riding on CanopyCover specifically.** Take it away, and GNNWR gains *zero* advantage from being
spatially aware — it does no better than a plain global model on the remaining 18 terrain/climate/
soil variables. So the honest story isn't "environmental relationships vary spatially and GNNWR
captures that" (what the current draft implies) — it's closer to "how CanopyCover's relationship
to ceiling height varies by place is the one thing GNNWR is demonstrably picking up; for genuine
environmental variables (terrain, climate, soil), there's barely any signal for any method, local
or global, to work with."

**CanopyCover-only GNNWR — partial result (08-22, 3 of 5 folds back, not final).** Completes the
ablation matrix from the other direction: does CanopyCover *alone* reach close to the full Set4
$R^2$? Partial 3-fold pooled result: **$R^2 \approx 0.246$** — much closer to the full-Set4 number
(0.294) than to the without-CanopyCover floor (0.049). CanopyCover alone recovers roughly 80% of
the total gain over that floor; the other 18 variables add about +0.048 on top of CanopyCover
(0.246→0.294) — almost exactly the same size as what they contribute *alone*, without CanopyCover
present (0.049), suggesting they contribute a small, roughly fixed, independent amount of signal
either way, not something CanopyCover is masking or absorbing. Reinforces the existing conclusion
rather than complicating it. Individual folds ranged 0.225–0.270, so treat 0.246 as directional,
not final — folds 3–4 land tomorrow, revisit then.

**Does GNNWR's CanopyCover coefficient actually vary, or is it flat?** Checked directly (GNNWR's
own saved per-plot `coef_CanopyCover` values, Set4, pooled 5 folds): mean 23.6, SD 5.7, always
positive, coefficient of variation ≈ 0.24 (swings by roughly ±24% of its average value across
plots). Not a frozen constant dressed up as "local" — the strength of the CanopyCover-to-ceiling
relationship genuinely differs from place to place, it just never flips sign. Doesn't resolve
whether that 24% swing is genuine environmental heterogeneity or GNNWR's nearest-neighbour
weighting smoothing over locally similar plots — but it's real variation, not nothing.

**Is GNNWR's structural linearity part of the problem? (UPDATE 08-22 — corrects an 08-21 miss)**
The 08-21 pass of this file argued the opposite of what I now think: it compared XGBoost's R²
without CanopyCover (0.061) to EN's (0.042), both near zero, and used that as evidence AGAINST
"linearity is the blocker" — reasoning that a flexible non-linear global model should have found
more signal than a linear one did, if there were much to find. **That was premature.** A quick
SHAP-dependence check (plotting each plot's SHAP value against its own raw feature value, from the
same no-CanopyCover XGBoost fit) shows `windward_topex` and `slope_degrees` both have clean,
tight, clearly *non-linear* (saturating) relationships — slope's effect rises steeply from 0–15°
then flattens completely above that; windward_topex has a similar S-shaped curve. Not noise.

So there is real non-linear structure in the data that XGBoost can see and GNNWR structurally
cannot (GNNWR is local-*linear* — one straight line per location, just a different straight line
in different places; a saturating curve like slope's gets mangled by any single straight-line fit,
local or global). This doesn't overturn the CanopyCover-dominance finding above, but it does mean
"GNNWR's spatial machinery isn't finding environmental signal" and "GNNWR's linearity is part of
why" can both be true simultaneously. The two aren't competing explanations — they likely compound:
even the modest environmental signal that exists is shaped in a way a local-linear model handles
poorly. **Checked**: building a genuine non-linear + spatially-varying model isn't available in
this project's toolkit (the `gnnwr` package only ships linear-in-features architectures) and would
be new methodology development — realistically days of work, not recommended given the timeline.

**Why did we expect GNNWR to work at all, given Q1 already found non-linear relationships?**
(genuinely worth stating explicitly, since it's easy to conflate two different assumptions.)
There are two separate "linear" questions here, and GNNWR only relaxes one of them:
1. *Is a feature's effect on the target a straight line, at a given location?* (e.g. does height
   rise proportionally with slope, or saturate?) GNNWR does **not** fix this — at every location it
   still predicts a linear combination of the raw features, same limitation as EN.
2. *Does a coefficient's strength change smoothly/simply as you move across the map, or in a more
   complex pattern?* This is what GNNWR's spatial-weighting sub-network was built to relax — it's
   a genuine neural network (not a fixed distance-decay kernel like classical GWR), so it can learn
   flexible, non-linear spatial weighting patterns.

GNNWR was chosen to test #2 — whether the *strength* of a relationship differs by place — not to
fix #1. That's a legitimate reason to try it, independent of the non-linearity already known from
Q1. What likely limits it in practice: if a relationship (like slope's saturating curve) is
non-linear *and* both ends of that curve show up within the same local neighborhood (very plausible
— slope varies within a compartment, not just between compartments), a single local straight line
can only approximate the neighborhood's *average* sensitivity, not the curve's shape. So a
well-justified spatial model can still underperform when the underlying relationship is curved and
that curvature isn't cleanly separated by geography. Worth stating both halves in the dissertation:
why GNNWR was a reasonable choice, and precisely what kind of signal it structurally cannot reach
even when the choice was reasonable.

**Honest summary for the dissertation**: GNNWR's edge is real, small, and now decisively shown to
depend entirely on CanopyCover — not on any broader ability to capture spatially-varying
environmental relationships. Whatever non-linear terrain signal exists (windward_topex, slope) is
real but structurally out of GNNWR's reach either way. This is a much sharper, more specific
finding than the current draft's "suggestive not confirming" line — worth rewriting the whole
"why GNNWR wins" framing around, not just swapping in a caveat.

---

## 2. Is CanopyCover really holding the model together? (NEW — and yes, more than Q1)

This exact question was asked and answered for Q1's target. It was never asked for Q2's target,
even though Q2's target uses the same variable. It should have been — the answer is more extreme,
and (UPDATE 08-22, see section 1 above) it's now confirmed true for GNNWR too, not just EN/XGBoost
— GNNWR without CanopyCover also collapses to $R^2=0.049$, the same near-zero floor as the global
models. CanopyCover isn't just the top variable in this chapter's global models — it's the entire
reason any of the three methods, including the spatially-aware one, gets meaningfully above zero.

**Ablation (already run this session, verified again here)**: dropping CanopyCover from Set4 and
refitting on Q2's own target ($\Delta y_{max}$):
- Elastic Net: 0.240 → **0.042** (an 83% relative collapse)
- XGBoost: 0.250 → **0.061** (a 76% relative collapse)

Compare to Q1's own ablation on its target: EN 0.358→0.231 (35% drop), XGBoost 0.395→0.249 (36%
drop). Q2's collapse is more than twice as severe.

**Is it circular (literally the target restated)?** No — checked directly, NEW this pass:
Pearson correlation between raw CanopyCover and $\Delta y_{max}$ is **0.468** ($r^2 \approx 0.22$).
Moderate, same range as Q1's own check (0.35–0.47). Not near 1, so it isn't literally a disguised
copy of the target. But moderate correlation + near-total ablation collapse means: nothing else in
the 19-variable set is carrying independent weight. Without CanopyCover, the fold-by-fold $R^2$
swings from $-0.10$ to $+0.14$ — that's a model that's basically not working, not a model that
lost a third of its power.

**Does anything "absorb" CanopyCover's role when it's removed? (NEW check, SHAP before/after)**

| Variable | SHAP with CanopyCover | SHAP without | Change |
|---|---|---|---|
| CanopyCover | 3.28 | — | (removed) |
| chelsa_bio12_precip_mm | 1.43 | 1.05 | down |
| windward_topex | 0.82 | 1.13 | **+38%** |
| slope_degrees | 0.64 | 0.92 | **+44%** |
| tas_mean | 0.58 | 0.92 | **+59%** |
| dist_to_road | 0.56 | 0.62 | flat |

Unlike Q1 (where the "next most likely" variable, dist_to_road, did *not* rise — clean evidence
against credit transfer), here several variables *do* rise 40–60% once CanopyCover is gone. This
is a murkier picture than Q1's clean story, and should be described that way, not forced into the
same "ruled out" framing. The honest read: some of what windward_topex/slope/tas_mean "explain"
with CanopyCover present might partly be picking up scraps of a correlated signal that CanopyCover
would otherwise claim — but since removing CanopyCover crashes $R^2$ to near zero anyway, whatever
they're picking up isn't translating into real predictive power. It looks more like the model
redistributing importance among near-noise features once the one dominant signal is gone, not
"terrain quietly does real independent work that CanopyCover was masking."

**What this means for "terrain matters moderately, soil is noise"**: technically still true as
stated (within the model that includes CanopyCover), but needs two caveats stacked together, not
one. First, this whole ranking exists inside a very small remaining budget — CanopyCover is
carrying ~80–90% of the model's total signal, and the "terrain vs. soil" ordering is being read off
what's left in the other ~10–20%. Second (UPDATE 08-22, see section 1's SHAP-dependence check):
what terrain signal exists there is genuinely non-linear (clean saturating curves for
windward_topex and slope_degrees, not noise), which caps how much of it GNNWR's local-*linear*
structure can actually use even where it's real. "Terrain matters moderately" should read as "a
real but small and shape-constrained non-linear signal," not a plain magnitude statement. Worth
saying explicitly, the same way Q1 now does for its own CanopyCover finding.

---

## 3. Which variables are the "real" key ones, and why (consolidated)

Using GNNWR's own saved per-plot local coefficients (not previously in any draft table), sign
stability across compartments, Set4/4survey, pooled 5 folds:

| Variable | % compartments same sign | Read |
|---|---|---|
| CanopyCover | 100.0% | Always positive — dominant, stable |
| slope_degrees | ~95–97% | Very stable, real local effect |
| windward_topex | ~92% | Stable — a better "stable terrain variable" example than plain `topex` |
| chelsa_bio12_precip_mm | ~75% (dominant direction) | Moderately stable |
| topex | ~72% | The least stable of the terrain variables actually plotted in the current draft — an odd choice to feature as the terrain example when better-behaved options exist |
| tas_mean, dist_to_road, soilgrids_ph | 50–57% | Essentially coin-flip — not real local effects by this measure |

**This directly matches Q1's own finding** (topex is genuinely unstable, precip more stable) —
good cross-chapter consistency, worth saying explicitly rather than leaving Q1 and Q2 as two
disconnected stories about overlapping variables.

**Practical suggestion**: if Q2 gets a coefficient-map figure (the "local coefficient" figure
already planned), `windward_topex` or `slope_degrees` are better candidates than plain `topex` —
topex is specifically flagged in the planning notes as "the least stable variable actually
plotted," which undercuts the punch of showing it as a coefficient map.

---

## 4. Is $\Delta y_{max}$ actually a clean target? (verified, not just asserted)

This is the least "in the text yet" and most structurally important finding sitting in this file's
old planning notes. I looked at the actual figure behind the claim (currently just a loose PNG,
`TEMP_results_attribution/TEMP_q2_growth_mismatch_vs_delta_ymax_2026-08-21.png`, not yet a
numbered chapter figure) rather than trusting the number alone.

**What it shows**: x-axis = growth-rate mismatch (observed growth minus what the yield class
predicts). y-axis = $\Delta y_{max}$ (the actual Q2 target). There's a clear, continuous, rising
relationship across the *entire* ~56,000-plot population (Pearson r=0.51, Spearman r=0.61) — not
just among the flagged outliers. A binned median line rises steadily from about $-4$m to $+20$m as
growth mismatch goes from negative to positive.

**Already tested and ruled out**: this is not just the flagged plots dragging the correlation —
excluding all 266 flagged plots barely moves it (0.509 → 0.525). It's a population-wide pattern.

**What it means, in plain words**: the curve-fitting procedure holds each plot's growth-*rate*
shape fixed (borrowed from its yield class) and only fits the ceiling. So if a plot's real growth
rate doesn't match what its yield class assumed, that mismatch doesn't disappear — it leaks into
the fitted ceiling number instead. $\Delta y_{max}$ is therefore not a clean, isolated measurement
of "how much higher/lower does this stand's ceiling sit" — it's better described as a stand-in for
the whole trajectory's deviation from the yield-class-expected shape, rate and ceiling tangled
together.

**Why this can't be fully resolved with this data**: separating "genuine correlated site-quality
effect" (better conditions raise both growth rate and ceiling together, a real biological story)
from "estimator artifact" (the fixed-rate assumption forces a rate mismatch into the ceiling
number) would need either more survey points per plot than 4–6, or a simulation with a known
ground truth. Both explanations are plausible and not mutually exclusive — this section should say
so plainly rather than pick one. This is a case where, per your framework, the honest answer is
"we can't fully separate these two, here's why," not a forced pick.

**Consequence for the rest of Q2**: every attribution claim in this chapter (CanopyCover, terrain,
soil rankings) is really attributing *some blend* of ceiling and growth-rate differences, not a
pure ceiling effect. This doesn't invalidate the attribution work, but it changes what the reader
should be told it means.

---

## 5. The outlier/artifact story (recap — already solid, no new concerns found)

This is the best-evidenced part of the chapter and I didn't find anything wrong with it:
- ~10 plots recur as the worst residuals across every model/set (real, not random — flag rate ~36x chance).
- Traces cleanly to the fixed-$k$ curve-fitting mechanism: growth rate is pinned to yield class, so slow/unstable growth forces an inflated asymptote.
- Scales to 266 (4-survey, 0.47%) / 709 (6-survey, 5.32%) plots via Tukey fence, clustered by compartment.
- Removing flagged plots does not improve $R^2$ — they carry real signal, not just noise.
- Full-population split test (250-plot recomputation): 71.6% fit the artifact signature (exposed terrain), 28.4% look like genuinely fast, real growth (sheltered terrain) — tested with actual p-values, not eyeballed.
- Already-disclosed open items: the 266-vs-250 count discrepancy (unreconciled, both individually verified), and the fact that the artifact mechanism is confounded with a real wind-exposure effect (already flagged in an AUDIT comment) — both remain fair to leave open, nothing new found here.

One thing worth connecting explicitly: finding #4 above (growth-rate mismatch leaking into
$\Delta y_{max}$ for the *whole* population) is the same mechanism as this section's fixed-$k$
artifact, just at a much smaller, more extreme scale for the 266 flagged plots. They're not two
separate findings — the flagged-plot story is the tail of the population-wide pattern in #4. Worth
saying so directly rather than presenting them as unrelated.

---

## 6. Interesting graph ideas (ranked by how much they'd actually move the story)

1. **The growth-mismatch scatter (already exists as a loose PNG)** — genuinely the single most
   important unbuilt figure in this chapter. It's the visual evidence for point 4 above, which
   currently isn't represented anywhere in the actual draft. High priority to promote to a real
   chapter figure.
2. **A before/after SHAP bar chart for the CanopyCover ablation** (same style as Q1's Figure 1) —
   makes "80% of the model is one variable" visceral rather than a sentence with two numbers in it.
3. **GNNWR CanopyCover local-coefficient map**, already planned — worth keeping, now backed by the
   CV=0.24 finding (genuinely varies, doesn't just look impressive).
4. **A "ceiling decomposition" figure**: EN's real $R^2$ (0.240) vs. the compartment-average
   diagnostic ceiling (0.538) vs. GNNWR's real $R^2$ (0.294), as three bars — visually shows "here's
   how much structure is really there, and here's how little of it GNNWR actually captures." This
   is a more honest, more interesting figure than just showing GNNWR "winning."
5. **windward_topex or slope_degrees coefficient map** instead of / alongside topex, given topex is
   the least-stable of the terrain variables by GNNWR's own numbers.

---

## 7. What would actually move the ceiling (future work material)

Two different levers, worth keeping separate in the dissertation's future-work section — one is
cheap and already queued, the other is the more honest "real" answer.

**Cheap, queued 2026-08-22 (not back yet)**: feed GNNWR pre-saturated inputs instead of raw ones.
GNNWR is structurally local-*linear* — it cannot represent a curve, even one whose strength varies
smoothly across space (see section 1's two-different-linearities discussion). The SHAP-dependence
check found `slope_degrees` and `windward_topex` both have genuine saturating (steep-then-flat)
relationships to the target, not noise. So: `slope_degrees` was capped at 15° and `windward_topex`
clipped to $[-12, 6]$ (knots chosen by eye from the SHAP plot, not a separate validation split —
disclosed as a real shortcut, fine for a quick robustness probe, not for a headline claim), and
GNNWR refit on Set4 with those two columns swapped in, everything else unchanged. No new
architecture, no new package — same GNNWR, transformed inputs. This directly tests whether
linearity (not just CanopyCover-dominance) was costing GNNWR real signal. Results not back yet.

**The more honest, higher-leverage answer**: even if the transform above helps, the entangled-
target finding (section 4) suggests the real ceiling on this whole chapter is the *target*, not the
model. $\Delta y_{max}$ bakes in growth-rate-mismatch noise because the curve-fitting procedure
holds each plot's growth rate fixed to its yield class rather than fitting it freely.

**Would simply freeing $k$ per plot fix this? Checked (08-22) -- not necessarily, and here's a real
number why not.** Every plot in the 4-survey cohort is observed across the exact same 15-year
window (age span has zero variance across all 71,766 rows), and **42% of plots are never observed
past age 40** -- their entire observed record sits in the steep, still-rising part of the growth
curve, nowhere near where it flattens toward its ceiling. For those plots, freely fitting *both*
$k$ and $y_{max}$ from 4 points on a still-climbing curve is a genuinely poorly-determined problem:
a slower rate with a higher ceiling and a faster rate with a lower ceiling can produce almost the
same rising trajectory over a young, non-asymptotic window -- the data can't tell them apart. This
is almost certainly *why* the original method fixed $k$ to the yield class in the first place, not
an oversight. So freeing $k$ trades one problem (rate-mismatch smearing into the ceiling) for
another (unstable, poorly-identified fits for a large share of the population) -- not a strict
improvement, a genuine trade-off.

**Tested directly (08-22), not just argued theoretically -- the trade-off is real and decisive.**
Built a cheap pilot: fit each plot's own $(y_{max}, k)$ freely via per-plot non-linear least
squares, NO shrinkage (the crude, fast version of the idea -- `rq3_free_k_pilot_check.py`, ~3
minutes to run, no cluster needed). Result:

| Method | Plots with an implausible ceiling ($<5$m or $>60$m) |
|---|---|
| Current (fixed-$k$) | 2,596 / 56,526 = **4.6%** |
| Free-$k$, no shrinkage | 23,971 / 56,526 = **42.4%** |

Freeing $k$ without shrinkage produces roughly **9x more** physically implausible ceilings than the
current method -- confirming the identifiability argument above with real data, more severely than
even the 42%-never-observed-past-40 estimate suggested (only 25% of the extreme cases were
young/short-window plots -- the instability is broader than that one specific risk factor). Worse,
it doesn't pay for itself: XGBoost on the free-$k$ target (CanopyCover included, restricted to the
32,350 plots with a non-implausible fit) scored $R^2=0.219\pm0.037$ -- not an improvement on the
current method's $R^2=0.250$, if anything slightly lower (population and column-count differences
mean this comparison isn't fully clean, so don't lean hard on the direction, just note it's not a
win either). **Conclusion: naive free-$k$ is not worth pursuing on its own merits** -- it trades a
well-characterised 4.6% artifact problem for a much worse 42.4% implausibility problem, for no R²
gain. This is real evidence *for* the shrinkage approach specifically (not "freeing $k$" in
general) being the only version of this idea worth the multi-day cost below, if pursued at all.

**The better middle path**: partial pooling / shrinkage, not a binary fixed-vs-free choice. A
nonlinear mixed-effects model where $k$ has a population-level distribution and each plot's own $k$
is a shrunk deviation from it -- pulled hard toward the yield-class value for plots with little
identifying information (the 42% that never approach their ceiling), allowed to deviate more freely
for plots with better curve coverage. This project already has mixed-effects (LMM) tooling built
for other targets; this would extend the same idea to the growth-curve-fitting stage itself, not
introduce an unrelated technique. **Cost**: a full, properly-validated nonlinear mixed-effects
implementation is real new statistical infrastructure -- nothing in this project's toolkit does
this today, and getting it to converge sensibly across ~56,000 plots / 231 compartments, handling
the 42%-identifiability problem cleanly, is realistically days of work, same order as the non-
linear-GNNWR estimate above -- not recommended given the timeline. A cheaper approximation (fit
each plot's own $k$/$y_{max}$ independently via ordinary nonlinear least squares, then shrink $k$
toward its yield-class value with a simple precision-weighted formula, not a full mixed-model
solver) is closer to a day of focused work -- still non-trivial, not a "quick cluster job" like
what's already queued in this chapter.

**A trap worth naming if this is ever pursued**: fixing the target and then re-running the *same*
attribution models (EN/LMM/GNNWR/XGBoost) doesn't cleanly tell you what was wrong if results are
still weak. Two separate limitations sit at two different pipeline stages -- the target's own noise
(this section), and the attribution models' structural linearity (the linearity discussion above) --
and fixing one does not isolate the other. The clean way to tell them apart: a 2x2 comparison, old
target vs. new (shrinkage) target, crossed with linear (EN) vs. non-linear (XGBoost) attribution.
If XGBoost's edge over EN *widens* on the new target, real non-linear signal was being masked by
target noise before, and linearity is now the live bottleneck. If both stay flat and CanopyCover
still dominates the same way, that points to genuinely low environmental signal, not a modelling
artifact. Worth stating this as the honest diagnostic path, not attempting it now.

**Does XGBoost have the target-noise problem too? Yes, directly -- already evidenced, not just
inferred.** The growth-rate/ceiling entanglement lives in the target itself, built before any
attribution model sees the data. Whether EN, LMM, GNNWR, or XGBoost predicts $\Delta y_{max}$ from
environmental features, all four are predicting the same noisy quantity -- XGBoost's non-linear
flexibility helps it fit non-linear *environment-to-target* relationships, but does nothing to fix
noise baked in upstream. This is exactly why XGBoost without CanopyCover only reached $R^2=0.061$,
barely above linear EN's 0.042 -- if XGBoost's flexibility could rescue signal a noisy target was
hiding, that gap should have been much bigger. It wasn't. The target-noise problem is shared
equally by every attribution model in this chapter, linear or not.

A purpose-built architecture for the *attribution* half of this problem, if pursued anyway
(separate from the target-noise fix above): a shared non-linear "backbone" (small MLP or GBM) for
the feature-target shape, plus a low-dimensional per-compartment embedding for spatial variation in
strength -- separates the two problems (shape vs. place) that GNNWR's single local-linear layer
currently has to solve at once. Not attempted here; flagged as a real direction, not a promise it
would work, given the target-noise ceiling described above.

---

## 8. Open worries — genuinely unresolved, said plainly

- **6-survey collapse**: sample size ruled out, true cause unknown. Fewer compartments (47 vs 231)
  is the leading candidate; deliberately not being pursued further (your call — 6-survey isn't
  carrying narrative weight either way, its R² is ~0 across all three methods already). Leave as an
  explicitly acknowledged, un-investigated limitation, not a resolved one.
- **266 vs 250 flagged-plot count**: disclosed discrepancy, never traced to the exact ~16 plots
  that differ.
- **Growth-rate vs. ceiling entanglement (#4)**: cannot be cleanly separated with 4–6 survey points
  per plot. Real limitation of the data, not a fixable bug.
- **GNNWR's edge — RESOLVED (08-22)**: entirely dependent on CanopyCover, confirmed directly by
  ablation (0.294 → 0.049 without it, same floor as EN/XGBoost). No longer open.
- **Is CanopyCover's ~24% coefficient swing genuine environmental heterogeneity, or GNNWR's
  nearest-neighbour smoothing?**: still open — the ablation confirms CanopyCover is *where* the
  signal is, not *why* it varies spatially. Would need a comparison to a plain compartment-fixed-
  effects model to separate; not attempted.
- **Non-linearity capping GNNWR's environmental signal**: live and evidenced (SHAP-dependence
  check, 08-22), not fully resolved — real for windward_topex/slope_degrees, but how much of the
  CanopyCover ablation result it explains vs. genuine spatial-strength variation isn't separable
  with this project's toolkit. Stated as a real, disclosed limitation, not chased further (would
  need new non-linear+spatial methodology, days of work).
- **SHAP redistribution after the CanopyCover ablation**: real but partial — some variables rise
  40–60%, but overall $R^2$ still collapses. Doesn't cleanly resolve to "real independent signal"
  or "pure artifact" — said that way above, not forced to a verdict.
- **GNNWR CanopyCover-only**: partially back (3/5 folds, 08-22) — $R^2\approx0.246$, directionally
  confirms CanopyCover alone gets most of the way to the full 0.294 (see section 1). Not final;
  revisit once folds 3–4 land.
- **Still queued, not yet run**: a 2-seed reseed check of the 4-survey coefficient map, and the
  saturation-transform check (does capping slope/windward_topex help GNNWR at all). Come back to
  this file once rsynced.

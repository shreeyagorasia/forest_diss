# Q2 brain dump — what's actually true, what's shaky, what's new

Written 2026-08-21. Plain language, for you to read and decide what's convincing before any of
this goes in the dissertation. Marked **NEW** where I ran a real test this pass that wasn't in the
draft before. Everything else is re-checked against saved files, not re-derived from scratch.

## The story Q2 is supposed to tell

Q1 found: global models (same formula everywhere) leave real spatial clustering in their errors.
Q2 asks: does letting the model change *by location* (GNNWR) pick up what the global models miss?
And separately: of all the environmental variables, which ones actually drive the plot-to-plot
gap between a stand's own growth ceiling and what its yield class assumes ($\Delta y_{max}$)?

Two sub-stories, and they're not equally solid. The "does location-aware help" story is weak but
honest (small edge, overlapping confidence intervals, corroborated two ways). The "what drives it"
story turns out to hinge overwhelmingly on one variable — more than anyone had written down yet.

---

## Verdict map

| Claim | Verdict | Confidence after this pass |
|---|---|---|
| GNNWR beats EN/XGBoost on every 4-survey set (point estimate) | **True** | Confirmed, unchanged |
| ...but the edge is statistically solid (CIs don't overlap) | **False** | Confirmed false — CIs overlap everywhere |
| Moran's I drop backs up the R² edge | **True, but weak corroboration** | Still just "suggestive," see below — NEW test adds a reason to stay cautious |
| GNNWR's edge = genuine spatially-varying environmental relationships | **Partly true, mostly unresolved** | **NEW**: GNNWR only closes ~22% of the available compartment-level gap — see "why GNNWR wins" below |
| CanopyCover is the top variable, not circular (not just the target restated) | **True** | Confirmed, correlation 0.35–0.47 range (Q1); **NEW for Q2**: 0.47 |
| CanopyCover "matters a lot" for Q2 | **Understatement** | **NEW**: it's ~80% of the model's entire explanatory power, worse than Q1 |
| Terrain matters moderately, soil is noise | **True but needs a huge caveat** | Only true within the ~5% of R² left after CanopyCover — see below |
| $\Delta y_{max}$ cleanly measures "ceiling difference" | **False** | Confirmed: entangled with growth-rate mismatch across the whole population, not just outliers |
| 6-survey collapse = sample size | **False, ruled out** | Confirmed ruled out; true cause still unknown |
| The ~10/266/709-plot outlier story (fixed-$k$ artifact) | **True, well-tested** | Solid — this is the best-evidenced part of the chapter |
| 71.6%/28.4% split (artifact vs. real fast growth) | **True, tested on full flagged population** | Solid |

---

## 1. Why does GNNWR actually win? (NEW investigation)

The draft says GNNWR's R² edge is "corroborated but not confirmed" by its bigger Moran's I drop,
with an honest caveat that this doesn't prove genuine spatial non-stationarity. I wanted a sharper
answer than "maybe." Here's a real test.

**The idea**: if GNNWR is mostly just learning "this compartment tends to run high/low" — a
location-level *average* effect, similar to what a compartment random intercept would do — then
most of its edge should be recoverable just by shifting each compartment's predictions by that
compartment's own average error. If GNNWR is doing something richer (the *relationship* between
CanopyCover/terrain and the ceiling genuinely differs by place, not just the average level), the
edge should be harder to explain by a simple shift.

**What I did**: took Elastic Net's actual Set4 predictions, and asked: what would $R^2$ be if you
"cheated" and corrected each plot's prediction using its own compartment's average leftover error
(computed from the same test data — this is a diagnostic ceiling, not a real model; a real model
can't see test-set answers, so treat this as an upper bound, not an achievable score).

**Result**:
- EN's real Set4 $R^2$: 0.240
- **That diagnostic ceiling (perfect compartment-average correction): 0.538** — more than double
- GNNWR's real Set4 $R^2$: 0.294

So there's a *lot* of real, compartment-level structure sitting in the data that a global model
completely misses (0.240 → 0.538 is a huge gap). GNNWR only recovers about a fifth of that gap
(0.240 → 0.294, against a ceiling of 0.538). That's not nothing — it means GNNWR is doing
*something* real — but it's a much more modest achievement than "GNNWR captures the spatial
structure." It captures a small slice of a much bigger pool of structure that's sitting right
there, unexplained, by any of this chapter's models.

**Does GNNWR's top coefficient actually vary, or is it flat?** Checked directly (GNNWR's own
saved per-plot `coef_CanopyCover` values, Set4, pooled 5 folds): mean 23.6, SD 5.7, always
positive, coefficient of variation ≈ 0.24 (i.e. it swings by roughly ±24% of its average value
across plots). So it's not a frozen constant dressed up as a "local" model — the strength of the
CanopyCover-to-ceiling relationship genuinely differs from place to place, it just never flips
sign. This is reassuring for "GNNWR is learning something," but doesn't resolve how much of that
24% swing is genuine environmental heterogeneity vs. GNNWR's nearest-neighbour weighting just
smoothing over locally similar plots (which would also produce a varying-but-never-flipping
coefficient, without needing a real causal explanation).

**Honest summary for the dissertation**: GNNWR's edge is real but small, and demonstrably captures
only a modest fraction of the spatial structure that's actually there. This is a *stronger*,
more specific caveat than the current draft's "suggestive not confirming" line — worth swapping in.

---

## 2. Is CanopyCover really holding the model together? (NEW — and yes, more than Q1)

This exact question was asked and answered for Q1's target. It was never asked for Q2's target,
even though Q2's target uses the same variable. It should have been — the answer is more extreme.

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
stated (within the model that includes CanopyCover), but the fair caveat is that this whole
ranking exists inside a very small remaining budget — CanopyCover is carrying ~80–90% of the
model's total signal, and the "terrain vs. soil" ordering is being read off what's left in the
other ~10–20%. Worth saying explicitly, the same way Q1 now does for its own CanopyCover finding.

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

## 7. Open worries — genuinely unresolved, said plainly

- **6-survey collapse**: sample size ruled out, true cause unknown. Fewer compartments (47 vs 231)
  is the leading unexplored candidate but untested here.
- **266 vs 250 flagged-plot count**: disclosed discrepancy, never traced to the exact ~16 plots
  that differ.
- **Growth-rate vs. ceiling entanglement (#4)**: cannot be cleanly separated with 4–6 survey points
  per plot. Real limitation of the data, not a fixable bug.
- **GNNWR's edge — genuine spatial heterogeneity vs. smoothing artifact**: narrowed by the new
  compartment-ceiling test (only ~22% of available structure captured) and the coefficient-CV
  check (real variation, never flips sign) — but not fully resolved. A cleaner test would need
  something like comparing GNNWR to a plain compartment-fixed-effects model, not attempted here.
- **SHAP redistribution after the CanopyCover ablation**: real but partial — some variables rise
  40–60%, but overall $R^2$ still collapses. Doesn't cleanly resolve to "real independent signal"
  or "pure artifact" — said that way above, not forced to a verdict.

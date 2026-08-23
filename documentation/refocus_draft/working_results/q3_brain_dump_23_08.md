# Q3 brain dump — plan for the new results, in plain language

Written 2026-08-23. This is a planning document for us to read together, not text to paste into
the dissertation. Written in plain language on purpose — simple words, short sentences, jargon
explained the first time it's used. Same facts and numbers as before, just easier to read.

Quick reminder: $R^2$ is a score from 0 to 1 for how well a model predicts tree height. 0 means
the model is useless, 1 means it predicts perfectly. Higher is better.

Each finding below is labelled with how solid it is:
- **READY** = solid evidence, can write into the dissertation as-is
- **READY, but reframe** = the evidence is solid, but how we describe it needs a small change
- **NEEDS NEW WORK** = we don't have this data yet

---

## The story Q3 is telling, in order (reordered 2026-08-23)

Changed from the original 1-2-3-4-5 listing. Reasoning: a reader who's just learned (in part 1)
that XGBoost and DNN both beat PINN/PINN-$k$ on accuracy will immediately want to know *why this
chapter keeps talking about PINN at all*. Answering that right away (part 2 below) keeps the
reader motivated before drilling into which PINN variant to prefer (part 3, a refinement of part
2's answer, not a new topic). The "why you can trust these numbers" material (part 4) reads
better once the reader already cares about the result, not before.

1. Compare the models — why do some predict height better than others? (DNN vs.\ XGBoost)
2. Compare PINN vs.\ DNN — PINN can draw a growth curve for each plot, DNN can't. What does
   that actually buy us, and where does it go wrong? *(was part 3)*
3. Compare PINN vs.\ PINN-$k$ — now that we know PINN's value is interpretability, which
   variant should we actually use? *(was part 2)*
4. Other new findings — explaining *why* things came out the way they did, and why the
   reported numbers can be trusted
5. **NEW** — do the models make bigger mistakes at certain tree ages or heights? Does this
   differ between DNN, PINN, and PINN-$k$?

---

## The findings, one at a time

**1. XGBoost beats DNN. We tested four possible reasons why, and ruled all four out.** —
**READY.** Already written up (current draft, lines 53–64), and today's work doesn't change it.
The numbers: XGBoost scores 0.674, DNN scores 0.655 (both on Set3, the main feature list, on the
4-survey data). We double-checked today that these numbers are still correct and not out of
date. One extra bit of good news: today we retested whether better training settings
(learning rate, batch size, etc.) would help DNN close the gap — using the *correct* batch size
this time (256, matching what's actually used everywhere else — an earlier test back on Aug 19
used a different batch size, 512, by accident, which made it *look* like better settings might
help. They don't, once tested properly). So the "training settings aren't the problem" claim now
has stronger evidence behind it than before — worth one added sentence, not a rewrite. Full
numbers: `temp_results_pinn/RESULTS_TABLE.md`, section 3.

**2. PINN's real advantage over DNN is that you can draw and check its predictions plot by
plot — DNN is a black box. But PINN sometimes gets carried away and predicts an impossibly
tall tree.** *(was finding 3)* — **READY**, mostly already written (current draft, lines 78–94,
includes the example-plot figure). Two things to add:
- We already checked how often this "impossibly tall" problem happens (see
  `temp_results_pinn/RESULTS_TABLE.md`, section 2): on average, PINN's predicted final height
  is about 2.9m taller than the population-wide average curve, and this varies a lot from plot
  to plot (spread of about 5.3m). 77% of plots get pushed taller than average, 23% shorter. Out
  of 11,508 plots, only 18 (0.16%) land on a genuinely unbelievable number (under 5m or over
  70m tall). Small in count, but real, and worth stating plainly rather than glossing over.
- Frame this as a trade-off, not a clean win: DNN can't be checked this way at all — it just
  gives you a number with no story behind it. PINN gives you a story you can actually look at
  and sanity-check, but sometimes that story is wrong. Both things are true at once.
- **NEW (2026-08-23) — why does this happen?** Checked whether the too-tall predictions are a
  terrain-data problem (`temp_results_pinn/RESULTS_TABLE.md`, section 8). Two different answers
  for two different groups: the *general* tendency to predict too tall (the top 10% most-inflated
  plots) has **no link to unusual terrain** — these plots sit at completely ordinary terrain
  values. So the broad pattern isn't a data-quality issue, it's just how the model behaves. But
  the much smaller group of genuinely implausible plots (72 of 46,032 test rows, 0.16%) *does*
  show a real, if modest, ~2x higher "unusual terrain" score than average — small sample, not
  proof, but a sensible signal that the worst individual failures may be partly linked to terrain
  values the model saw less of during training.

**3. Plain PINN beats PINN-$k$ on accuracy, every single time we tested it.** *(was finding 2)*
— **READY, but reframe.** Here's where I want to push back gently on how you first described
this ("which is more practical"). The numbers:

| Test | PINN (predicts only final height) | PINN-$k$ (predicts final height AND growth speed) |
|---|---:|---:|
| Main result | **0.631** | 0.618 |
| Learning-rate test | **0.630** | 0.614 |
| Bigger-batch test | **0.641** (not fully finished) | 0.619 |
| Physics-weight test (8 settings tried) | 0.623–0.632 | 0.612–0.620 |

Plain PINN wins every time, by a small but consistent amount (about 0.01–0.02). So "PINN-$k$ is
more practical" isn't really true on accuracy grounds — we need a different, honest reason to
prefer it. We tried one candidate reason earlier this session (that PINN-$k$'s growth-speed
number could be converted into a "yield class" foresters already use) and it didn't hold up —
turned out to be an unrelated quirk in the yield-class formula, not something PINN-$k$ actually
explains. That thread is closed. The honest reason to still care about PINN-$k$ is simpler: it
draws a *more complete* picture of each plot — both how tall the trees will get AND how fast
they'll get there, not just the final height. That's a real thing PINN-$k$ does that plain PINN
doesn't, even if it's not more accurate. Worth saying it that way, not as "more practical."

**NEW (2026-08-23) — why does adding k make things worse?** Checked directly
(`temp_results_pinn/RESULTS_TABLE.md`, section 7): compared how much extra correction the shared
"trunk" network has to do in each version. PINN-k's trunk does almost **twice** as much
compensating work as plain PINN's (ratio 1.98). In plain terms: personalizing growth speed (k)
doesn't produce a better-fitting curve by itself — it makes the curve fit slightly *worse*, and
the rest of the network has to work harder to clean up after it. This isn't just an observed
pattern anymore, it's a real, tested reason why plain PINN wins on accuracy.

**4. Environment information genuinely helps PINN — but only up to a point, and physics-weight
and training settings are already close to as good as they can be.** — **READY**, everything
below is from this session's testing (`temp_results_pinn/RESULTS_TABLE.md`, sections 3, 4, 5):
- We tested PINN with no environment info at all, then with three different-sized lists of
  environment features. Going from zero environment info to *any* environment info gave a real,
  solid jump in accuracy (about +0.05 $R^2$ — much bigger than the normal wobble between runs,
  which is about ±0.02). But going from a medium, carefully-picked list of features to a much
  bigger list didn't help any further, and slightly hurt plain PINN. This is the main new
  finding worth headlining: environment matters a lot, but a well-chosen list beats a big one.
  It's also the fixed version of an earlier, broken test — the old version (before we fixed a
  bug in the code) showed no difference at all between having environment info or not, which
  was wrong.
- We also tested how strongly the model should be told to "obey" the tree-growth rule (called
  physics-weight, or $\lambda$) — tried 8 different strengths. None beat the setting we were
  already using. So the setting we picked originally, without knowing this, turns out to already
  be a good one — not by luck, we checked.
- Same story for the learning speed, the "keep it simple" penalty, and the training group size —
  tried different values for all three, none beat what we already had. This means the footnote
  in the dissertation currently saying "PINN received no hyperparameter tuning" can be upgraded
  to something stronger: "we checked, and untuned settings are already close to as good as
  they can be" — a tested, closed point instead of an open worry.

**5. Do prediction mistakes cluster at certain tree ages or heights, and does this differ
between DNN, PINN, and PINN-$k$?** — **NEEDS NEW WORK.** Nobody's checked this yet. What we have
and don't have:
- DNN: ready to go, the file already has everything needed (age, actual height, predicted
  height, error) — `outputs/spatial_block_kfold/rq1_dnn_env_terrain_nested_set3_gated_terrain_wind_vif_seed42/4survey/fold_0/predictions.csv`.
- PINN-$k$: ready to go, same story — `temp_results_pinn/outputs/example_curve/test_set_predictions.csv`.
- Plain PINN: **not ready** — the file we saved only has age and predicted final height, not the
  actual predicted tree height or the error. Cheap to fix (the model's already trained, we'd
  just export a different, small piece of information from it), just hasn't been done.
- The actual analysis (grouping errors by age band or height band, comparing the three models)
  hasn't started. This is a genuinely new question, not a repeat of anything above.

---

## A related idea that came up in chat — now done, and it's a real, positive finding

Q2 found something important about a different model (GNNWR): its whole advantage over simpler
models came down to one single input variable (CanopyCover), carrying about 80–90% of its total
signal, not "understanding location" in general. We checked whether PINN has the same problem —
is its personalized final height also secretly driven by just one terrain feature, or does it
genuinely use several? (Method: a "shuffle test" — shuffle one input at a time, see how much the
prediction moves. Full numbers: `temp_results_pinn/RESULTS_TABLE.md`, section 6.)

**Answer: no, PINN doesn't have GNNWR's problem.** The single most important terrain feature
(wind exposure) only accounts for 15.2% of the total importance — nowhere near GNNWR's 80–90%.
All eleven terrain features contribute meaningfully, spread fairly evenly between about 5% and
15% each. This is a genuine, positive point of difference worth stating in the dissertation:
PINN's personalization draws on the whole terrain picture, not one variable secretly doing
everything. And regardless of this result, one thing was worth keeping either way: PINN still
produces an actual, checkable growth curve for each plot, which GNNWR never does — GNNWR only
tells you "this variable matters more around here," not "this specific plot is expected to grow
like this."

---

## Which findings deserve the spotlight

Not a work-priority list — a list of which findings deserve the best figure and the most
explanation, versus which are just supporting evidence.

1. **Environment helps, but a well-chosen list beats a big one (finding 4).** The strongest, most
   new result from this whole session. A real effect, and a non-obvious twist (more isn't
   always better) that makes for a more interesting story than a flat "yes it helps."
2. **PINN sometimes predicts impossible tree heights (finding 3).** Good because it cuts both
   ways: evidence PINN is doing something real (a consistent 77% upward shift, not random noise)
   *and* evidence it can be checked and shown to be wrong sometimes. A model story with an
   honest flaw is more believable, not less.
3. **Plain PINN beats PINN-$k$ on accuracy (finding 2).** Surprising — you'd expect more detail
   (predicting two things instead of one) to help, not hurt. Cheap to support, we already have
   the numbers.
4. **Do errors cluster by age/height (finding 5)?** Could be the second-best finding in the whole
   chapter, or could turn out to be nothing — we don't know until we run it.
5. **Physics-weight and training-setting checks (part of finding 4).** Necessary to show we
   didn't just guess the settings, but not exciting on its own — defensive evidence, not a
   discovery. Best as a small table, not a big figure.
6. **XGBoost beats DNN (finding 1).** Already fully explained before this session, nothing new
   to add visually beyond the existing table.

---

## Plot ideas

Matching the look of figures we already have (`q3_pinn_example_plot_curve.png`,
`q3_pinn_param_distribution.png`) and the map style already used in Q1/Q2.

**For finding 1 (environment helps, then plateaus):**
- A simple line chart: accuracy ($R^2$) on the y-axis, feature-list size on the x-axis (none,
  small, medium, large), one line for each PINN version. Shows the "helps, then flattens out"
  story at a glance instead of buried in a table. Probably the single most useful new figure to
  add.

**For finding 3 (impossible tree heights):**
- Check whether the existing figure `q3_pinn_param_distribution.png` already shows the
  right (corrected) numbers, or whether it's from before this session's bug fix and needs
  redoing.
- **New idea — a map:** colour each forest compartment by how much PINN's predicted final
  height differs from the population average there. Same map style as existing Q1/Q2 figures.
  Interesting either way it turns out: if the "too-tall" predictions cluster in specific
  compartments, that's either a real environmental pattern (good sign) or a sign the model is
  reacting badly to something about the terrain data in those spots (a real weakness worth
  flagging). We don't know which until we look.

**For finding 2 (plain PINN vs.\ PINN-$k$):**
- A small chart comparing accuracy for both versions across all the tests we've run — shows the
  gap is consistent, not a fluke from one test.
- Optional: a scatter plot of predicted final height vs.\ predicted growth speed, one dot per
  plot — shows the actual shape of how these two numbers relate, more informative than a single
  summary number.

**For finding 5 (errors by age/height, once the data exists):**
- A line chart: average error on the y-axis, tree age (or height) band on the x-axis, one line
  per model (DNN, PINN, PINN-$k$). Obvious question to answer: does DNN get worse at very
  young/old ages (where there's less training data) compared to the two PINN versions, or is
  there no real difference?

**A second map idea (not tied to one specific finding):** colour each compartment by how wrong
DNN or XGBoost's predictions are there. If the mistakes cluster in the same places PINN's
too-tall predictions do, that's a strong, visual link back to the dissertation's bigger point
about environment affecting predictions (Q1/Q2). If they don't overlap, that's useful to know
too — it would mean these are two separate issues, not one.

**Before building any of this:** every plot idea above needs either (a) plot coordinates
(`data/interim/plot_coordinates.csv.gz`) or (b) forest-compartment shapes
(`models/common/geo.py`'s `load_compartment_boundaries()`). Checked already — both connect
cleanly to the prediction files we already have, so no new data prep needed, just the plotting
code itself.

---

## Questions for you before I start writing any of this into the dissertation

1. **Finding 2's reframe** — do you agree "PINN-$k$ gives a more complete picture" is the right,
   honest way to describe its value, replacing "more practical"? Or is there a real, demonstrated
   usefulness argument I'm missing?
2. **Finding 5's scope** — should I fix the missing plain-PINN data now, so all three models can
   be compared, or is DNN vs.\ PINN-$k$ only enough for this part?
3. **Ordering** — keep your original five-part order, or fold finding 4's second half (the
   physics-weight/training-setting checks, which are mostly "here's why you can trust our
   numbers" rather than a new discovery) into a smaller side-note instead of a full headline?

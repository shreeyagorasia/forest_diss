# Results-chapter review framework

Working process for going through the Results chapter (and Methodology, where it feeds Results)
section by section. Purpose: nothing goes into the dissertation until the user has actually
understood it well enough to explain it back — not until it "sounds right."

## The single test behind every claim

For every headline claim, answer explicitly:

**Does the evidence give a clear answer, a partial/suggestive answer, a clear "no," or no answer
at all?**

| Verdict | What it means | How to word it |
|---|---|---|
| Clear answer | Strong, direct evidence, checked | State it plainly, no hedge needed |
| Suggestive, not confident | Real evidence, but not decisive (e.g. overlapping CIs, small effect, single check) | "consistent with," "points toward," not "shows" or "confirms" |
| Ruled out / answered no | Directly tested and did not hold up | State the negative result plainly — a "no" is still a finding |
| No answer yet | Genuinely untested, or evidence doesn't distinguish explanations | Say so directly (`% AUDIT:` or a stated limitation), don't paper over it |

A claim's *wording strength* must match its column, not the column we'd prefer it to be in.
Never let "confirms"/"proves"/"shows" describe something that's actually suggestive-not-confident.

## Per-section checklist

For each section/paragraph, before it's considered done:

1. **Method** — does the described method match what the code actually does? Trace to the real
   script/function, don't assume the prose is right.
2. **Numbers** — every headline number either recomputed live from saved outputs, or traced to a
   specific dated source file. No number goes in on trust. If two sources disagree, say so rather
   than picking one silently.
3. **Wording strength** — check against the table above. Look specifically for "confirms",
   "shows", "proves", "explains" used where the evidence only supports "suggests"/"consistent
   with". Look for causal language describing an association.
4. **Jargon and readability** — written for someone with little ML background and who isn't a
   native English reader. Concretely:
   - Short phrases over long compound sentences.
   - One idea per sentence.
   - Define every technical term (statistical, ML, or forestry) the first time it's used, in
     plain words, not just a citation.
   - No unexplained acronyms.
   - Read it back and ask: would this survive being read aloud to someone outside the field?
5. **Story flow** — does this section connect to the one before/after it? Where possible, reuse
   the *same* example plot(s)/compartment(s) across Q1 → Q2 → Q3 rather than introducing a new
   one each time — a reader following one concrete example through the whole chapter understands
   the story better than three different, disconnected examples.

## Discussion-paragraph structure (per finding)

Each bolded finding in Results follows this shape, in this order:

1. **Headline result** (bolded sentence) — the one-sentence takeaway, in plain language, before
   any supporting detail.
2. **Candidate explanations / checks** — what was actually tested to explain or stress-test the
   headline (numbered if there are several).
3. **Alternatives ruled out** — what was considered and rejected, stated plainly as a "no." A
   ruled-out alternative is evidence *for* the headline, so it belongs here, not buried in a
   subordinate clause.
4. **Caveats** — what this finding does NOT establish; anything still open (`% AUDIT:` etc.).
5. **Conclusion phrase** — one sentence closing the loop back to the headline.
6. **So what** — one explicit sentence linking back to the research question itself (Q1/Q2/Q3):
   what this means for actually answering "does environment explain growth deviation" (Q1),
   "does spatial variation matter" (Q2), "does this improve prediction" (Q3) — not just "here is
   a fact," but "here is what this fact means for the question."

### Ordering principle: headline signal before caveat-analysis

Lead with whatever answers the research question for the *majority* of the data — the actual
headline signal (e.g. Q2's modest-but-real result across the ~55,850-plot majority) — before any
outlier or robustness analysis (e.g. Q2's 266-plot Tukey-fence investigation). The outlier
analysis is a caveat *on* the headline, not a second headline competing with it: it explains why
the headline signal can be trusted despite a few implausible points, it doesn't replace the
headline. Order paragraphs so a reader gets the actual answer to the research question first, the
robustness story second — even when the robustness story is more narratively striking. (Q2's
current draft order does this backwards in places — worth fixing paragraph order, not just
wording, when a section is revisited.)

## What to cut vs. keep (space-constrained editing)

Triage every paragraph/figure against argument-centrality (does it change the answer to the
research question?), not how interesting it is on its own:

- **Keep, main text**: anything in the headline-signal chain for a research question (the actual
  "so what" answer), or a caveat that materially changes how confidently that answer should be
  stated (e.g. CI overlap meaning "directional, not confirmed").
- **Compress to one sentence, or move to appendix (still referenced in the main text)**:
  robustness/sensitivity checks that corroborate a conclusion without changing it (a second
  Tukey-fence recomputation, a reference-density ablation). State the result in one sentence in
  the main text ("a robustness check confirmed X"), full detail in the appendix.
- **Cut entirely, or fold into one clause**: numbers already stated elsewhere, candidate
  explanations that were quickly ruled out and don't need their own paragraph, narrative detail
  that doesn't change the verdict on a claim.
- **Never cut**: an unresolved `% AUDIT:` flag, a genuine caveat that changes how confidently a
  headline claim should be stated, or the one worked example that carries the reader through the
  section (see story-flow above) — cutting the example plot loses the thing that makes the
  section understandable, even though it "looks" cuttable as a single data point.

## Figure review checklist

For every figure (existing or newly built):

- Does it actually look finished (clean layout, readable labels, no overlap) — or does it need a
  formatting pass first?
- Main text or appendix? Space is limited — headline figures that carry a claim's main evidence
  go in the main text; figures that are useful but not load-bearing (a secondary robustness check,
  a "typical" companion example, a diagnostic that supports but doesn't drive a conclusion) can be
  appendix items, *still discussed in the main text*, just not displayed there.
- If a plot doesn't add anything beyond what a sentence already says, it can just be text — not
  every claim needs a figure.

## Response structure to use when reviewing a section together

1. State which claim/paragraph is being checked.
2. What the evidence actually shows (the real, verified number/result, with its source).
3. Verdict from the table above (clear / suggestive / ruled out / unanswered).
4. A plain-language rewrite, if the current wording doesn't pass the jargon/readability check.
5. Anything still open, flagged plainly, not hidden.

## Ground rule

The user has to actually understand and be convinced by a section — able to explain it back in
their own words — before it goes into the dissertation. That's the actual finish line, not
"the prose reads fine."

# Next: isolate state drift before changing supervision

Planning decision after phase7; not an executed or frozen GPU protocol.

**First reuse joint128 without training to locate the remaining gap.** It retains
4/5 local and3/5 full-generation repairs; both complete-visibility targets work
in both modes. Universal inability to fit or dominant cross-sample conflict is
not supported. Apple/bowl is the clearest local-success/full-failure case. Keep
all five cases, flag the old bicycle-label issue, and report the clean two-case
subset separately instead of selecting only the observed failure.

Freeze the same checkpoint, behavior, seeds, flow10/512/CFG4.5 and query indices
(2,0,0,1,3). Compare these execution rules:

1. Old prefix, one updated action, old suffix: replay phase7 local reference.
2. Old prefix, updated student from query to end: isolate later action changes.
3. Updated prefix, one updated action, old suffix: isolate the changed query state.
4. Full updated student: replay phase7 whole reference.
5. Old prefix, updated student for the next2 or4 steps, then old suffix: two
   predeclared interval controls, not per-image selection of a better length.

For index0, several modes are identical; check duplicates rather than treating
them as independent evidence. Preserve old targets/noops. Record actual sigma
endpoints and NFE. This is a flow10 diagnostic, not a robustness test.

**Expected information, not a promised gain:**

- If old-prefix updated suffix retains repair but updated-prefix old suffix
  loses it, prioritize on-policy state coverage and re-query experts on actual
  student states.
- If old-prefix updated suffix already loses repair while the single action
  succeeds, extra unsupervised time-step changes are implicated. Short intervals
  may localize them; they need not improve the image.
- If both mixed cases work but full generation fails, investigate interaction.
- If short intervals lack credible net value, do not implement an interval selector.

**Then, if supported, train on complete short-interval expert sequences.** Propose
a2–4-step intervention at its own branch states, finish with common behavior,
verify terminal repair, and supervise every intervened state/action. Do not credit
a multi-step result only to its first velocity. Compare with one-step supervision
and a fixed teacher/interval strategy under declared compute, including verification.
Do not add a critic, learned gate, VLM reward or gated-LoRA architecture yet.

Before a method claim, match ordinary OPD and selection-arm initialization,
data/loss/optimization coverage; include continued OPD if starting from a trained
student. Establish fixed-strategy headroom on all queries, then fresh on-policy
training and locked held-out evaluation. Only after the main deployment works
should flow20/40 and DPM2 robustness be tested. Five selected training queries
cannot replace those comparisons.

# Accepted-pair review rubric

Fixed before viewing phase4 review sheets or their identity/score key.

Review all 22 predetermined accepted training pairs. Show A/B images with prompt;
hide noop/target identity and scores. The observer implemented the experiments and
has seen some phase3 images, so this is not independent ground truth or perfect
blinding. Preserve original observations and label any post-reveal caveats separately.

For each A/B image record:

- Every constraint ID as true/false/null. `text` means the requested word is legible
  once without extra letters/words (ignore case; letter spacing alone does not fail
  visual readability). Record a literal `visible_text` when discernible.
- `count` includes the requested object identity and exact count. `color` includes
  object identity, exact count AND color, matching the metadata rather than ignoring
  count. `relation` includes both required objects, exact counts and the stated relation.
- `full_visibility` is separate: requested objects complete within the frame and
  without substantial foreground occlusion. Uncertain missing/merged object identity
  or peeled-versus-multiple-fruit interpretation gets null; do not guess hidden objects.
- Notes include partial progress even when the exact clause still fails, extra text,
  clipping, non-black text, absent white sign, or ambiguous identity/count.

Then give A/B/tie/uncertain for better satisfaction of the prompt, with a reason.
Use uncertain for conflicting improvements/regressions or ambiguous comparison.
Do not rank by aesthetics alone. A relative improvement can still have major failures;
binary unchanged clauses do not prove that a positive progress score was erroneous.

Freeze all observations, timestamp and SHA before opening key.json. After reveal,
report descriptive target-better/noop-better/tie/uncertain counts and clause transitions,
including remaining defects. Never use the review to remove selected queries from
the GPU matrix or to rewrite any original v1 label. Independent adjudication remains
necessary for scientific claims about true constraint satisfaction.

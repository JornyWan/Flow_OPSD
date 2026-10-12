# Calibration observation rubric

Written before viewing the 16 new calibration images. All cases are predetermined;
show prompts/images, hide automatic scores, retain every case and uncertainty.

- `text`: one legible occurrence of the requested word, ignoring case and letter
  spacing. Extra letters, words, numbers or registration symbols fail; ambiguous
  tiny strokes get null. Record literal visible text without guessing hidden letters.
- `count`: required object identity and exact count. `color`: required identity,
  exact count and color. `relation`: exact counts of both required objects and the
  stated relative position. Unclear merged/peeled fruit or object identity gets null.
- `full_visibility`: separate from the encoded clauses. False for clear cropping,
  absent required objects or substantial occlusion; null for uncertain boundaries
  or identities. Do not guess hidden objects.
- Notes record malformed objects, ambiguity, visible text, missing white sign,
  non-black lettering and other prompt failures. A readable spaced word may pass
  `text` while failing style or placement requirements in notes.

Freeze all observations with SHA/UTC time before reading the score files/key.
Original observations stay immutable; label post-reveal caveats separately.
These implementing-assistant observations are provisional, not independent human
ground truth. Prompts appeared in prior training; only these noise realizations
are held out from developing/checking the fixed measurement rules.

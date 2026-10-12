# Final update128 image review

Read `runs/le7_review_001/items.json` and the five sheet images only before
freezing `observations.json`. Hide key, numerical scores, fitting-arm and mode.
This is an arm-blinded implementing-assistant review; familiar training images
and prompt wording mean it is not an independent human study.

For EVERY one of20 images, record each requested encoded clause as true/false/
null, full_visibility as true/false/null, literally visible text, and concrete
notes. OCR correctness requires exactly the specified word, case-insensitive,
with no duplicated/missing characters or extra visible text. A stylized but
identifiable letter is acceptable; an unclear glyph is null, not auto-corrected.
Count requires exactly the specified recognizable objects. Relation is explicitly
subject relative to object in the image plane. Do not infer an absent object.
Visibility is separate: cropping/occlusion must not be silently dropped.

Freeze observations and their hash before joining the key. Retained repair means
all previously repaired AND initially correct encoded clauses are true, and no
new full-visibility failure when the old target was fully visible. Unknown is
not success. Report the two full-visibility target cases separately from the
three targets retaining old visibility failures. Review does not retroactively
change any training target, scorer or phase4 label.

# Phase11: model/score-hidden review of every transfer image

Record text, each encoded semantic clause, full_visibility, literal visible text
and notes for each A/B/C/D image. Values are true/false/null. Freeze all42 sheets
before key join. Do not infer a model from picture quality or change previous labels.
The reviewer is the implementing assistant, not independent human ground truth.

Text: requested word case-insensitively, no missing/extra character or other
visible letters/words. License plates and object labels count. Identifiable
stylized letters pass; unclear glyphs/possible extra letters remain unknown.

Count: exactly the requested recognizable objects; partial but identifiable
objects contribute to count while cropping separately fails full_visibility.
Relation: requested subject clearly left/right of requested reference object,
with the specified object multiplicities. A subject mainly below another is
not clearly right merely because its center is slightly to the right. Ambiguous
additional objects/signs remain unknown, as in phase10's extra-bowl uncertainty.

Color: the specified number of recognizable target objects must have the
requested color, matching the encoded color clause's include/count requirement.
Thus a count mismatch can also fail a color clause that requires that count;
also record the observed colors in notes so this coupling remains inspectable.
Ordinary shading/specular highlights are allowed; unclear/pale color is unknown.
Do not silently reinterpret a blue-apple/red-banana prompt as its natural color.

Full visibility: all requested objects are complete within the image; cut limbs,
wheels, rims, book edges etc fail. If object identity/multiplicity makes this
uncertain, use null. This is separately recorded from encoded semantics.
Note scene defects not captured by these booleans: extra unrelated objects,
text on an object instead of a separate sign, deformed geometry, low contrast.

Report per-clause and complete-scene results, known harms/repairs and unknown
coverage. Strict success requires every clause=true and full_visibility=true.
Unknown is not strict success, and is never recoded as definite harm/failure.

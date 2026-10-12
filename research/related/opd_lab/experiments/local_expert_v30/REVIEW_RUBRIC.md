# Complete-scene observation v2 — frozen before Phase21 images

Inspect all images at native512px, with candidate identity/time/strength/seed hidden.
For each image record exactly: clauses (text plus the supplied count/color/relation),
full_visibility, visible_text (literal transcription), notes. Labels are true/false/null.
This is provisional implementing-assistant review; independent human confirmation
is a separate source, never silently substituted into this frozen version.

Text: requested word only, case-insensitive. Clearly extra/missing/repeated letters
or extra words anywhere fail. Read visible glyphs, not the intended prompt. Normal
typographic letter spacing does not itself create a different word; a gap alone is
not an OCR failure. If identity/word boundaries or tiny extra lettering are unclear,
record null rather than inventing characters. Record literal text including gaps.
This spacing criterion is explicitly new for Phase21 and is not retroactively
applied to previous phases. Font preference, boldness and aesthetic style are not
semantic failures by themselves. Completely unreadable requested text is false;
ambiguous but plausibly correct text is null.

Count: exact required count of identifiable objects. Distinct cropped objects can
count, while visibility fails separately. Do not invent objects behind occlusion.
Color: specified class and exact count must also be correct. Relation: both classes
and exact counts, then subject on the indicated side from the viewer's perspective.
For every condition, uncertain identity/count/position is null, not a forced error.
Required objects should be separate as stated; substantial overlap that prevents
checking full objects makes visibility false or null if unclear.

Full visibility: all required objects are present, complete inside the frame,
without substantial occlusion. Missing objects and obvious cropped rims/legs/body
fail. Touching the image boundary is not alone proof of missing geometry; if unsure
about whether an actual piece is cut, record null. Reflections and shadows are not
extra objects unless clearly depicted as distinct objects.

Complete scene: every clause AND full_visibility is true. The same set is used for
selection: a determinate false→true in ANY of these fields, including visibility,
is a repair; any true→false is harm. Any unknown in before/after forbids a safe
acceptance but does not erase a known harm. Preserve unknowns in denominators.

Record visible technical defects (washed-out detail, malformed geometry, unreadable
secondary print) in notes. These are not a calibrated aesthetic score or diversity
metric; complete_scene must never be described as comprehensive image quality.
Do not drop images or revise the rubric after observing outputs. Freeze the entire
cohort before joining the key; identical pixels and requirements use equal labels.

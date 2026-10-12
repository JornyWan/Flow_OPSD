# Phase17 original-task review, unchanged semantic criteria

Inspect all views at native512px. Model/arm/step/score are hidden. This is provisional
implementing-assistant review, not independent human annotation. Every item needs
an explicit true/false/null for each encoded clause AND full_visibility, literal
visible_text and concise notes; do not default unclear clauses to true.

Text: only the requested word anywhere in the entire image, case-insensitive;
extra letters, repeated letters, other words, labels, licence plates and small
printed sublines fail. Copy actual visible glyphs; do not complete an intended word.
Punctuation and within-word gaps are not silently deleted. If legibility or the
presence of additional printed letters is uncertain, use null and describe why.
Case-insensitive, NFKC and whitespace normalization do not excuse extra characters.

Each GenEval clause includes EXACT counts in its metadata. Count distinct objects,
including clearly identifiable cropped ones; do not invent hidden objects. A color
clause also fails if its required class/count is wrong. Relation position belongs
to its containing subject and references include[index]; require both identities
and exact counts, then the stated direction from the viewer's perspective.

full_visibility: all required objects present, complete within the frame, without
substantial occlusion. Missing objects or obvious cropped legs/rims/body fail. If
occlusion, multiplicity or completion is unclear, null. This is separate from
individual text/count/color/relation clauses. Do not infer full objects from a crop.

Complete scene means ALL encoded clauses AND full_visibility true. A safe repair
requires at least one known false-to-true encoded clause, no true-to-false clause
or visibility, and all baseline/candidate labels determinate. Unknown stays unknown;
known harm still counts even when another field is uncertain. Baseline false
visibility may remain false while another clause is safely repaired.

Use the same labels for identical pixels and requirements. Retain every generated
view and every ambiguity; freeze the entire draft before joining the model key.
Do not select images, drop nulls, improve labels after learning model identities,
or call an unencoded change (e.g. typeface/style) a repair of an encoded constraint.

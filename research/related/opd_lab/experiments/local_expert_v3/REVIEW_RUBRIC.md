# Training-image coverage audit rubric

Written before viewing phase-3 review images or their score/identity key. Use the
eight prompt IDs already frozen in `probe_plan.json`, with all five policies for
each ID. The implementing assistant is the observer, not an independent evaluator.

Each observation is true, false, or null (uncertain); never force an unclear image
into a pass/fail label. Record reasons for false/null and additional prompt failures.

- `text_exact_no_extra`: the requested word is legible and spelled correctly,
  ignoring case, and appears once with no other visible words, letters or registration
  symbols. Clearly duplicated words, misspellings or extra text fail. Ambiguous tiny
  marks/glyphs get null with a note. This is descriptive visual inspection, not OCR.
- `scene_explicit`: the explicitly encoded GenEval objects/count/color/relation are
  satisfied visually. Use the prompt to identify the intended entities. Uncertain
  object identity, color or count gets null. Keep framing/occlusion separate here.
- `full_visibility`: all requested objects are fully inside the frame and visibly
  complete. Clear clipping or substantive occlusion fails. Unclear boundaries get null.
- `notes`: mention text content, ambiguity, clipping/occlusion, object separation,
  sign placement, or other unencoded prompt requirements when visible. Do not infer
  hidden objects or score an image using its aesthetic appeal.

Save and hash all 40 observations before reading the key. Then report descriptive
agreement/disagreement with the original v1 clauses and instances where a v1 pass
misses visibility or extra-text defects. Do not change old v1 labels, use this
small convenience audit as a new training reward, or claim independent truth.

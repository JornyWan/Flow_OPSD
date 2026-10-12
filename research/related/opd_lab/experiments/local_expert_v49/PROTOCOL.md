# Phase49: local action transfer, prespecified for both learning rates

Paper claim C2: does a learned one-step correction retain terminal value with an
unchanged student prefix/suffix? No training or new selector. Use all32 Phase47
prompts/noise states, already exposed development data. Restore the verified
Phase46 student as frozen behavior. Reuse its saved prefix through index1.
Compare three single-action actors: published GenEvalTeacher, original-LR Phase47
verified student, and low-LR Phase48 verified student. Frozen old behavior
recomputes every remaining suffix action.96 new images, all task evidence and
all five quality scores. Reuse old full before/high/low endpoints unchanged.

Primary diagnostic low_local minus before B joint OCR exact AND GenEval strict;
high_local and teacher_local are prespecified controls, plus local/full gaps.
Report all cases, no outcome-based selection. Family-stratified prompt bootstrap
20000 seed49049, descriptive intervals, no training-seed inference. Positive
local point difference with negative full-minus-local difference identifies a
route gap worth controlled confirmation; not proof that a specific time step or
optimizer moment caused damage. Mixed-model routes are diagnostic only and are
not presented as deployment by a single student. No tuning from this result.

All prefix states and sigmas verified; first case replays all routes exactly,
behavior/expert weights frozen.1200-second singleL40 cap, zero optimization,
96 new images, zero agent image reads, no automatic retry/time extension.

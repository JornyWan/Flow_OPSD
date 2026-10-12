# Phase16: local-action transfer and full-trajectory interference

Freeze before submission. No training, optimizer, EMA, checkpoint choice, learning-rate
search, sampler change or automatic-verifier acceptance gate. All 18 phase15 positive
prompt states: six original fitting seeds and twelve previously exposed fresh seeds.
These are development data, not untouched confirmation. Same SD3.5-M/LoRA raw
joint128 and label_selected128, flow10, query index3, alpha1, 512px, fp16, student
CFG4.5, OCR CFG4.5, GenEval CFG1.0, original complete task and clause definitions.

Replay full joint128 and full label_selected for all18, plus the six previously
selected expert target endpoints. Fail immediately if any of42 saved PNG hashes
differs. Capture z3 during the full baseline replay; on six old seeds require exact
state, context, timestep and saved-noop equality. Replay all18 behavior noops.
Freeze behavior as joint128 throughout; only the default adapter is swapped.

After every reference passes: predict label_selected at each shared z3 (18 new
images), and query both original OCR and GenEval at each of12 fresh states (24 new
images). Every branch applies exactly one substituted action at index3, then uses
joint128 at indices4..9. Save state/context/actions/CPU-noise and provenance hashes.
Reuse the six old selected teacher endpoints and existing full-generation labels.
No cherry-picking cases or averaging selected successful seeds.

Budget: one L40, 20min cap. Exactly42 references +42 new images, 24 new teacher
queries, 0 updates. Base full replay180 student forwards; learned full180; learned
local18. Behavior18 noop checks +36 old reference suffix +252 new suffix =306.
OCR12, GenEval12; CFG examples1404 (GenEval uses1, others2); decode/verifier84;
text encode12 (six prompts, negative and positive). These counts are checked by
the worker against its frozen plan. Replay overhead is included in GPU-hours.

Review all42 new images at native512px with randomized path-hidden identities,
before joining the key or automatic scores. Full image text, exact object counts,
color/relation and visibility retain true/false/null. Copy old full/selected-target
labels as inherited, provisional assistant observations with source hashes.
The implementing assistant has seen phase15 images: partial blinding, not an
independent human assessment. Identical pixels retain identical semantic labels.

Analysis reports all18 per-state rows, separated old/fresh, complete scene,
safe repair, known harm and unknown, using the frozen phase15 compare criterion.
Fresh teachers are reported separately, plus an explicitly oracle/diagnostic
ANY-teacher opportunity count; it is not a deployable selector result. Pair each
single-student and full-student endpoint with the same baseline. Compare learned
action MSE/direction against both expert actions descriptively, not for selection.
Show prompt-cluster uncertainty (six prompts), do not count seeds as independent
prompts. No pass/fail claim of a new method from this small diagnostic.

Interpretations may coexist: scarce teacher safe repairs implicate candidate
coverage; teacher-safe/student-single-not-safe cases implicate local learning or
state generalization; student-single-safe/full-not-safe or full-more-harm cases
implicate changing the rest of the trajectory; failure on old targets despite low
MSE motivates terminal sensitivity/residual action error checks. These observations
do not identify prefix versus suffix separately or establish sampler robustness.

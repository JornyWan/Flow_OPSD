# Phase48: single-factor low-LR ablation

Claim C2 deployment transfer. Reuse Phase47's exact per-arm48 states and teacher
targets, all48/family-matched3/verified3 masks. Restore each arm's own Phase46
step8 student, nonempty Adam, scaler and RNG; explicitly override only LR from
3e-4 to1e-4 after restoration. Eight full-batch attempts, denominator48, original
loss, teachers, Flow10 shift3 CFG4.5, fp16,512. No new collection, optimizer reset,
new retention term, target normalization or checkpoint selection.

Generate three final students on all32 Phase47 prompts with unchanged noise:
96 new images. Reuse original224 before/after/noop endpoint scores verbatim under
their own original fingerprints. Previously exposed development data, not an
independent confirmation. Primary diagnostic: low-LR verified minus original-LR
verified, B OCR exact AND GenEval strict. All three corresponding contrasts,
before comparisons, fixed/random, components, harm, all five quality metrics,
family/template groups, target MSE and same-state step0 drift reported.

Family-stratified prompt bootstrap20000 seed48048; intervals descriptive, no
training-seed uncertainty or multiplicity adjustment. A joint point improvement
and OCR point improvement versus original LR is a signal to design independent
confirmation, not proof of efficacy. Do not scan more LRs on these32 prompts.
Regardless of outcome, next stage compares both fitted students' one-step actions
under the same old student prefix/suffix. No winner chosen for that test.

Preflight: old default restore behavior retained, LR override preserves Adam
history and source state, disk resume with new LR exact, own-state targets,
selection only A, original upstream loss kernel and source identity contracts.
GPU audit3 single-query updates reset before formal24 updates. New image cap96;
singleL40 cap1200 seconds including failures/loading/scoring. No automatic retry
or time extension. Original archives and old session budget unchanged.

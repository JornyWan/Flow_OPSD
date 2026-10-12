# Phase9: do experts still repair the current student's states?

Frozen after phase8 analysis and before this GPU job,2026-09-13. A separate
campaign; no alteration of phase8's no-teacher-query protocol. No training.

Use the same five training prompts/seeds/query indices and joint128 student as
phase8, flow10/512/CFG4.5/fp16. Generate its actual full trajectory and use the
state reached at the old query index. Recompute both original experts there:
OCR CFG4.5 and GenEval CFG1, with pinned unchanged assets. Propose four actions
v_student + alpha*(v_teacher-v_student), alpha=.5/1. This is a fresh teacher
query on the new state, not transplanting the old state's target vector.

For every candidate, use the same frozen joint128 student to finish sampling.
Baseline is its ordinary full generation. Five baseline images must match phase8
full-image hashes exactly; all teacher proposals and suffixes are inference only.
Save all25 images and unchanged-v1 scores, actual state/teacher/candidate actions,
and complete branch state/action sequences for later reproducible analysis.
Expected216 model forwards:206 student,5 OCR,5 GenEval;427 CFG examples.
One L40-class GPU, ≤30 minutes. Include failure/startup costs. No LR or sampler sweep.

Review ALL20 new candidates without candidate identity or scores. Pixel-exact
matches may inherit existing phase7/8 frozen observations, but never old phase4
labels with the known bicycle-only-word inconsistency. Freeze observations before
joining keys. Same strict phase8 rubric, uncertainty retained; reviewer is the
implementing assistant, not independent human ground truth. Current baseline's
labels are inherited by exact image hash.

Primary question: does any candidate repair an encoded false clause of CURRENT
baseline, preserve all currently true encoded clauses, and avoid losing full
visibility when baseline is fully visible? Unknown does not pass. Also report
visibility-only gains, harms, all candidate rows and both teacher/alpha settings;
do not hide changes on the already-correct bowl/banana controls. The candidate
max over these five selected training cases is only a diagnostic headroom count,
not a policy gain, dynamic-selection benefit or generalization estimate.

If fresh teacher actions repair the new apple/bowl state under the current
suffix, test bounded on-policy state refresh with matched optimization before
inventing a temporal gate. If not, inspect intervention timing/duration before
expecting more fitting at the old time to solve it. If teacher gains are mostly
proxy false positives, retain that limitation rather than declaring success.

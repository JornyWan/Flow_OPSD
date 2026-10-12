# Phase13: verifier development on immutable phase12 endpoints

Freeze endpoint_evidence_v2_dev1 before its first image replay. This is development
on all192 previously reviewed phase12 images, not held-out validation. No new
diffusion generation, action queries, training, candidate curation or label edits.
Original joint_constraints_v1/select and upstream source remain unchanged.

OCR evidence: complete original512 and2x LANCZOS images; group detected boxes
by vertical overlap >=.5 and order left-to-right. Join adjacent boxes on a row,
retain all recognized characters, punctuation, spaces inside each recognized
string and all extra rows. Reread every detected row from BOTH full-image passes,
with8px padding and2x scaling using the same recognizer (det=False). Every read
must be nonempty and confidence >=.8. Agreement across all four read sets on
exact expected text gives true/false; disagreement or low confidence gives null.
Distinct wrong transcriptions may agree on false. No target-only crop selection,
best-read reward, global space deletion or confidence-as-ground-truth claim.
More than32 rows abstains, never drops tail rows silently. These are correlated
views of one OCR model, not four independent verifiers.

Object evidence: use the pinned upstream evaluate_image closure to retain its
strict result, progress, failure reason and detected boxes for each GenEval
clause. Original image and16px white-padded image must agree, else null. No
changes to upstream thresholds, NMS, color classification or relation rules.
Record original-view parity with phase12 OCR text and all GenEval scores/labels.

Visibility support is an explicitly provisional bbox heuristic: all required
objects must be detected and all target-class boxes must lie >3px inside the
ORIGINAL image boundary, also after removing16px from padded-view coordinates.
Both views agree =>true/false, else null. Audit against frozen visibility labels;
do not call bbox support a verified perceptual completeness model.

v2 selection: all baseline AND candidate clause/visibility evidence known;
at least one false->true encoded clause; no true->false clause or visibility.
Among eligible candidates prefer more repaired clauses, then smaller saved
action distance, then canonical order. No fitting and no manual repair pool.
For mechanism attribution also report v1_discrete_event: existing v1 binary
clauses with the same discrete-repair/protection rule, without visibility evidence.
This is not an original baseline; unchanged v1 automatic choices are retained.

One L40 GPU/four CPUs/96GB,30-minute cap. Existing images read-only; no student
model loaded. Expected192 evidence images,384 OCR detection views,480 GenEval
clause views (80 GenEval clauses across64 prompts,three candidates,two views),
variable recorded OCR crops. Record all times, failures and source/input hashes.

Summarize every query using frozen phase12 implementing-assistant labels: safe
precision /ALL accepted, known harms, unresolved, acceptance, headroom recall,
strict endpoint success, per-clause/visibility confusion, abstention reasons,
plus all accepted/rejected cases. Use2000 whole-prompt resamples/seed0. This
development set cannot support claims of generalization or new-seed gains.

Development readiness for a separate new-seed confirmation: retain at least4
of the6 available safe-repair states, safe/all-accepted >=80%, no known harms,
unresolved accepted <=10%, and zero original-view replay mismatches. These are
engineering checks, not statistical proof. If unmet, finish the diagnosis and
do not spend a confirmation cohort on a clearly nonfunctional prototype.
Any implementation fix uses a fresh numbered attempt, preserving failure/source.
Any algorithm change is a separately named development version, not a retry.

If readiness passes, freeze a NEW protocol and128 new states (all64 train prompts
x2 seeds), compare v1/v2 on identical candidate pool, hide identities/decisions
for all384 reviews before label freeze, and apply the phase12 learning gate.
No development labels, scores or phase13 revisions may tune that confirmation
set. Keep validation/test unused here. No claim that a safer selector alone
solves student fitting, behavior drift or sampler/step robustness.

# Phase 4: action direction/amplitude and accepted-target audit

Frozen before phase-4 GPU outcomes, 2026-09-11. No optimization, new query
selection, validation sweep or test use in this phase. External-expert OPD remains
the main method. This is a conditional mechanism diagnostic on selected TRAIN states.

## Fixed inputs and interventions

Use every one of the 22 accepted queries in phase3 `le3_probe_001`, including the
saved latent, conditioning, detached selected target, time index, seed and record.
Retain all four predefined raw checkpoints (fit2, fit2_keep, fit4, fit4_keep).
Reuse the same pinned SD3.5-M, fp16/512, flow10 schedule and CFG settings. Every
intervention finishes with the ORIGINAL frozen behavior; never refresh it when
loading a student checkpoint. No teacher queries or optimizer updates are needed.

Let b be the frozen behavior action, delta=a_target-b and e=a_student-b. Compute
gamma=<e,delta>/||delta||² and e_perp=e-gamma*delta in FP64 reductions. Preserve the
unclamped gamma, including negative values and values above one. The projected
action is b+gamma*delta. Log norms, cosine, gamma and orthogonality error.
Linear combinations use FP64 arithmetic before rounding the final action to FP32;
the unchanged suffix/backend runs with the original fp16 inference settings.

For each query generate/save/score these 13 actions:

- ray beta=0, .25, .5, .75, 1, with action b+beta*delta. Use b and the original
  saved target directly for beta=0 and 1 to avoid cancellation changing endpoints.
- Actual student action and projected student action for each of the four raw
  checkpoints. The only changed component within a pair is e_perp.

Total: 22*13=286 new images, all scored with unchanged joint_constraints_v1.
Save the actual action tensors as well as PNG hashes and scores. Pin hashes of
the parent frozen queries, behavior, data/config, logs and raw checkpoint files.
Check query identities, sigma increments, frozen behavior and checkpoint hashes.

## Reproduction and statistics

Regenerate all 22 noop, 22 target and 88 actual-student images, and compare their
pixels/file hashes and verdict fields with phase3. Reuse NO old score in phase4
contrasts: all comparisons use newly generated images within this phase. Report
every mismatch; do not silently substitute old outcomes. Parameter/input mismatch
or invalid verifier output is an error; nondeterministic pixel/score mismatch is
reported as a reproducibility limitation, without hiding the full new matrix.

Report every beta gain versus regenerated noop; adjacent beta changes; target
gain; and, for each checkpoint, actual/noop, projected/noop, projected/actual and
target/projected. Use paired prompt bootstrap (2000 resamples, seed0), retaining
all 22 selected prompts. This is conditional on phase3 selection and one training
seed, not an independent efficacy estimate. No multiple-comparison adjustment.

Also count per-query nonmonotonic ray values and how often removing e_perp changes
the score or loses an initially satisfied clause. Do not treat positive means or
the best beta as proof of a deployable method. Projection uses an expert target
at inference and is diagnostic only, not a proposed standalone sampler.

## Measurement audit

Before inspecting new ray results, review the existing noop/selected-target pairs
for ALL 22 accepted queries (44 images). Order cases and A/B identities by SHA256;
show prompts, hide scores, action identity and checkpoint. Record each explicit
OCR/GenEval clause separately, full object visibility and other prompt defects.
Use true/false/null and preserve ambiguity, including peeled-fruit interpretations.
Also record visible text and an overall A/B/tie/uncertain comparison with reasons;
an improvement need not yet pass a binary clause, so do not label every unchanged
binary outcome as a false-positive gain.
Freeze observations before reading the key. Derive only descriptive improvement,
regression and uncertain counts after reveal; never rewrite v1 or filter the GPU
query set based on this audit. This observer is the implementing assistant, not
independent human truth; prior phase3 exposure is a limitation of blinding.

## Interpretation and budget

If projected > actual, orthogonal movement is a candidate bottleneck on these
states. If small pure-ray actions fail but larger ones improve, amplitude or
nonlinear image/verifier thresholds remain candidates. If selected targets are
visually questionable, resolve measurement before using proxy gains as evidence.
These explanations can coexist and must not be declared before results.

One L40-class GPU, maximum 35 minutes, plus CPU image audit/reporting. Log actual
operations, wall time and allocated GPU-hours, including failed attempts. Finish
the full matrix and report before proposing further fitting or broader sampling.

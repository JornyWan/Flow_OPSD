# Next experiment after the complete Phase48–50 evidence

Status: candidate plan and CPU prototype, not submitted. Phase50 completion and
verification must precede a finalized Phase51 submission. No further GPU time
is reserved by this document. Completed Phase48/49 archives remain immutable.

## Scientific decision

Current shared-state results suggest that fitted local actions and whole-policy
behavior need separate tests. The completed Phase50 table additionally shows that
the Phase46 anchor already lost general quality relative to the published student.
The next candidate anchor is therefore the published student, not the Phase46
checkpoint. This changes the earlier candidate plan before any new development
images or tests; it does not change Phase48–50 designs or results. Low learning rate partly helps but is not a win
against the matched random control. Do not make further claims from repeatedly
searching the old32 prompts. The next single intended change is the temporal
support of the fitted adapter, with its tensors held fixed.

Candidate: published OPD Student as the frozen anchor and Phase48 low-LR fitted weights;
use the fitted state only for sigma in [0.9367391467094421,0.9800646603107452),
and the anchor outside. The boundaries are midpoints of the fixed training
query's neighboring sigma values, not fitted to new images and not recentered
when the grid changes. The previous Phase49 endpoint results do not transfer automatically to this new
anchor; only the noise-time route identity is reused. The old fitted checkpoint
is a development diagnostic, not fresh common-start online training.
All steps have one denoiser query; there is no external
teacher, verifier or search during deployment. This is one backbone with TWO
student adapter states, not an ordinary single-LoRA checkpoint. Adapter storage,
latency and NFE must all be reported. It remains an Euler-only prototype.

## Bounded development test to finalize before launch

- Cohort: all128 paper_joint_v2 development prompts, one already frozen noise per
  prompt. Do not touch512 confirmation or768 final-test prompts yet.
- Policies: published OPD Student anchor; low-LR full deployment; low-LR temporal
  deployment with the published anchor outside the window.
  Exactly384 images, same prompts/noise/Flow10/CFG4.5/resolution512.
- Primary endpoint: paired joint success, temporal minus anchor. Full versus
  temporal diagnoses localization. Report text exact, strict structure, repairs,
  harms, unknowns, errors, every family/style and all five quality metrics.
- No optimization, mask changes, teacher changes, new LR search or best-of-N.
  No prospective claim versus random/native OPD from these three deployment arms.
- Resource proposal: one L40, maximum3600 wall seconds; must fit the actual
  remaining current-session budget when launched. No automatic retry or extension.
- Decision: a positive development joint estimate with a positive descriptive
  95% lower bound would justify freezing a common-start training comparison;
  otherwise retain an inconclusive/negative diagnosis instead of claiming efficacy.
  Quality regressions must be reported even if joint success improves. Absence of
  significant quality harm is not evidence of noninferiority. If automatic endpoint
  errors make the primary score invalid, archive the error and stop that inference.
- This is development, not paper confirmation. The full prespecified readout and
  failure rules must be committed to the stage protocol before any images are made.

## What turns a positive development result into a paper result

1. Implement the corresponding online training distribution, including actual
   states visited with this time-dependent student. Do not claim a posthoc route
   retrofit of an old checkpoint is already that complete learning algorithm.
   Keep one common initial anchor fixed outside the window for the entire run;
   inside use a frozen snapshot of the current fitted adapter during collection
   and both terminal suffixes. Optimize only the trainable fitted adapter, then
   refresh that frozen snapshot. Training therefore needs anchor / behavior /
   trainable adapter ownership; deployment exports only anchor / fitted. Never
   accumulate a new anchor adapter for every round or silently sample with the
   full-time learner when the intended deployed policy is time-dependent.
2. Common published initialization, same train/dev data and task definition for
   adapted author OPD and verified corrections. Fresh optimizer history for both;
   do not mix Phase46 arm-specific histories into the main comparison.
3. Keep fixed/random selection controls under the SAME temporal deployment rule,
   plus a fixed-teacher temporal OPD control, to separate localization from
   terminal selection. An ordinary full-time baseline alone cannot do that.
   Add the public DanceOPD implementation
   with its actual low-noise sampling and velocity loss. NFT uses ordinary online
   generation, an explicit joint reward adapter, and all sampling costs.
4. At least3 independently collected training runs, with noise/order/selection
   randomness recorded. Use the paired run×prompt analysis, show each run, and
   report the limited precision with only3 runs. Equal steps alone are not equal
   compute; include teacher branches, verifiers and baseline extra training.
5. Freeze the main comparison, sample size, seeds and stopping rule before using
   confirmation/final outcomes. The prepared512/768 pools are capacity, not proof
   of power; training variability still needs estimation. Do not keep extending a
   negative confirmation or silently treat it as new development data.
6. Evaluate public GenEval553 prompts, OCR1018 and DrawBench200 separately from the
   custom joint task. State repetitions and native versus unified sampler protocol.
   Flow10 findings do not establish Flow20/40 or DPM2 robustness; fix weights and
   sigma-window before that matrix, preserve solver history, and report actual NFE.

## Implemented so far

- `opdlab.temporal_student`: explicit sigma-based route, exact original-grid CPU
  replay, one model query per step, input/solver validation.6 relevant tests passed.
- `opdlab.paper_statistics`: balanced paired run×prompt inference; averages fixed
  noises inside a cell and rejects missing/duplicate/nonfinite observations.
  4 tests passed. It does not fix evaluator bias, exposure or compute mismatches.
- Exact512 train /128 dev data-reader parity for pinned OPD, NFT and DanceOPD.
  Native OPD joint-data configuration prepared, not trained. The NFT text reader's
  empty metadata still needs an explicit joint-reward sidecar integration.
- New data literal targets survive all three tokenizers; no prompt was truncated.

No new SD3 efficacy claim or DPM integration is implied by CPU tests.

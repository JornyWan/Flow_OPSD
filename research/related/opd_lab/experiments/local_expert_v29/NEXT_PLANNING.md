# Planning notes before Phase29 visual outcomes

These are hypotheses for interpreting the fixed second cycle, not extra criteria, changed training settings, or authorization for an automatic third cycle. The Phase29 protocol and primary contrast remain unchanged.

The original idea asks whether an expert that is strong globally supplies useful corrections at this particular student state. Phase28 gave a first positive learning signal: verified21/36 versus random19/36 and initial13/36, but selection superiority is uncertain and the unknown-adverse sensitivity crosses zero. Improvements were concentrated in the color-plus-text family. Five known harms remained.

The saved-state audit shows that all three learners change their full trajectories before the teacher-supervised window. This is expected for a shared LoRA and is a measured property, not proof that drift caused the harms. Hard time-window deployment would change the method and remain tied to a schedule; do not call it a sampler-robust solution without testing.

If refreshed OCR-only learning keeps improving from S1, prioritize independent visual confirmation and fixed-checkpoint inference-grid checks before more optimization. A Flow10-trained checkpoint should be compared with its exact initial checkpoint and matched random counterpart at multiple inference grids on paired noise. One positive training-grid result cannot establish sampler/step robustness.

If OCR headroom shrinks or another cycle harms geometry, do not simply increase OCR fitting. Return to the original multi-expert question: can the unchanged GenEval expert supply a locally useful repair even when its overall image quality is lower than the current student? A bounded future probe can compare noop, two OCR actions and two GenEval actions from the same refreshed state, using the same frozen behavior suffix and unchanged verifier. Count extra candidates and suffix calls. First establish available safe repairs; only then train on them. This would test reciprocal expert corrections rather than assume another OCR-only cycle helps.

If a teacher branch is safe but learning its actions is repeatedly harmful, distinguish failure to fit actual local states from changes outside the supervised window. A preservation loss on current-student states outside the intervention region is one candidate, with a strength-matched baseline and total GPU-cost comparison. It must not be added post hoc to the ongoing controlled comparison. A smaller trust region may be useful, but cannot be claimed to solve the issue without an explicit test.

All claims still depend on provisional implementing-assistant observations and a small, repeatedly exposed development prompt family. Reserved final prompts remain untouched while choices are made.

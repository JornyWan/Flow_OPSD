# 0002 — Constrained-retention OPD experiment

Date: 2026-08-28  
Status: invalid implementation attempt; superseded by 0003

## Research question

Can direct PickScore optimization improve the released multi-task Student while a
one-sided external-teacher constraint prevents degradation of OCR and GenEval?

## Smoke-test method

- Initialization: released `DiffusionOPD/Student/lora`.
- Task slot 0 (`pickscore`): RTC raw reward-gradient target at the configured low-noise
  query only.
- Task slots 1 and 2 (`ocr`, `geneval`): ordinary routed teacher target at all training
  timesteps, converted to a hinge constraint against the frozen initial Student on the
  same current on-policy state.
- Constraint:

  ```text
  current_kl = KL(current student, routed teacher)
  reference_kl = KL(frozen initial Student, routed teacher)
  retention_loss = mean(relu(current_kl - (1 + rel_tol) * reference_kl - abs_tol))
  ```

- The direct and retention branches are disjoint by task mask. No raw KL is compared
  across teachers.
- First smoke runs only long enough to validate control flow and measure per-branch
  gradient norms. It is not a performance result.

## Runtime invariants

The smoke is invalid unless the log shows:

1. three routed task slots and released Student initialization;
2. frozen-reference means computed on the same sampled states;
3. direct mask contains only `teacher_idx == 0`;
4. retention mask contains only `teacher_idx in {1, 2}`;
5. finite direct loss, current/reference KL, violation, active fraction, and both
   branch gradient norms;
6. at least one optimizer update, incremented global step, and a checkpoint;
7. no use of the legacy RTC `continue` path that skips the ordinary teacher branch.

## Non-claims

Passing the smoke does not show that constrained retention improves quality. It only
establishes that the intended two-source objective was actually executed and provides
the calibration needed for the four-arm experiment.

## Submitted implementation

- Overlay: `code/0002_constrained_retention.patch`
  (`SHA256 65a05f8f8cbe2770340cbf598194aa66a2f5787a1752b0f3f8380507ad9a3374`)
- CPU equation/mask test: `code/0002_test_constraint_math.py`
- Exact batch script: `slurm/0002_cr_smoke.slurm`
- Run record/output: `runs/0002_cr_smoke/`
- Submission command: `sbatch codex_research/slurm/0002_cr_smoke.slurm`
- Job ID: `1035889`
- Output: `runs/0002_cr_smoke/slurm_1035889.out`

The job applies the overlay only to a temporary source copy after verifying the two
audited base-file hashes. It does not modify the shared DiffusionOPD tree.

Job `1035889` failed before sampling because the new validation referenced
`snr_weight` before its existing assignment. No optimizer update occurred. The exact
failure and single correction are recorded in 0003; the valid retry result is 0004.

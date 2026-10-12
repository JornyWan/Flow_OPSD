# 0005 — Next action: validate the direct engine before extending constraints

Date: 2026-08-28  
Status: decision gate; no new long job submitted

## Decision

The immediate next step is **not** another multi-task method arm. Let job `1035879`
finish its 100-step official-DiffusionOPSD top-k PickScore run, then evaluate its saved
checkpoints on PickScore, OCR, and GenEval with the existing unified offline evaluator.

Only after that result identifies a direct optimizer that actually improves PickScore
should constrained retention be integrated with it.

## Why this gate comes first

1. CODEX-0004 proved only that the direct/constraint control flow differentiates and
   that the hinge reacts to degradation. Its direct branch is the repository's RTC-raw
   approximation, not a validated best reward optimizer.
2. The sampled epoch-1 direct parameter gradient was about 105 times the retention
   gradient. Calibrating the constraint around an engine that may later be replaced
   would make the calibration obsolete.
3. `1035870` cannot validate top-k: its three epochs logged `TOPK skipped` because it
   constructed 368/376 rather than the expected 384 samples.
4. The full job `1035879` does enter the intended path. Its first diagnostic reports
   `k=6/24`, 48 local groups, 2 retained samples per local group, 96/384 unique samples,
   and batch conservation `OK`. At this decision point it is around epoch 23/100 and
   has checkpoints 10 and 20.
5. The existing `retain` job is a moving self-snapshot anchor. It is not the frozen
   external-teacher constraint from 0002–0004, so its result cannot answer the new
   research question.

## Phase A — finish and evaluate `1035879`

Wait for checkpoints 10, 20, ..., 100. Run the same offline protocol on every selected
checkpoint and on the released Student initialization:

- PickScore: direct objective;
- OCR: first protected capability;
- GenEval: second protected capability;
- identical prompt files, seeds, sample count, guidance scales, and evaluator versions.

Also compare with canonical official DiffusionOPSD under the same update budget. The
top-k arm is worth retaining only if it improves the held-out PickScore frontier, not
merely its on-policy training reward.

Go/no-go:

- **Go:** a checkpoint has a reproducible held-out PickScore improvement over the
  released Student/canonical direct baseline.
- **No-go:** training reward rises but held-out PickScore does not, or the best direct
  checkpoint is not better than canonical OPSD. In that case use canonical OPSD as the
  direct engine and drop top-k; do not tune constraints to rescue it.

OCR/GenEval degradation at this phase is expected diagnostic information. It measures
the retention problem that the next phase must solve.

## Phase B — integrate the constraint with the winning direct engine

Do this in the official DiffusionOPSD training path rather than continuing to build on
the RTC-raw smoke branch. Preserve the selected direct loss unchanged and add two
separate external-teacher constraints:

```text
c_ocr = KL(current, OCR teacher) - (1 + eps_ocr) * KL(initial Student, OCR teacher) - delta_ocr
c_ge  = KL(current, GE teacher)  - (1 + eps_ge)  * KL(initial Student, GE teacher)  - delta_ge

loss = L_direct + lambda_ocr * relu(c_ocr) + lambda_ge * relu(c_ge)
```

Required corrections relative to the three-step smoke:

- use real PickScore, OCR, and GenEval prompt distributions;
- keep OCR and GenEval metrics, active fractions, and multipliers separate;
- log epoch-averaged branch parameter-gradient norms, not only the first local batch;
- export raw LoRA weights as well as EMA weights;
- run at least 20 optimizer steps so EMA has updated (current interval is 8);
- use a primal-dual/adaptive multiplier per constraint rather than guessing one shared
  fixed coefficient from the smoke's 105x gradient mismatch.

Suggested multiplier update, detached from the model graph:

```text
lambda_m <- clip(lambda_m + eta_lambda * mean(c_m), 0, lambda_max)
```

Initialize each multiplier from measured branch-gradient scale, then let violation
feedback adapt it. Log cap-hit rate; a multiplier stuck at `lambda_max` invalidates the
calibration.

## Phase C — 20-step calibration gate

Run only two matched arms from the released Student:

1. selected direct engine alone;
2. the same direct engine plus adaptive OCR/GenEval constraints.

This is a mechanism/calibration run, not the final claim. Pass only if:

- direct, OCR-constraint, and GE-constraint gradients are finite;
- both constraint multipliers respond and do not remain at their cap;
- active fractions decrease after an initial violation rather than remaining near 1;
- global steps and raw/EMA checkpoints progress correctly;
- no constrained task is silently absent from a rank or batch;
- offline checkpoint evaluation is executable and uses updated, not stale EMA, LoRA.

## Phase D — performance experiment

Only after Phase C passes, run the preregistered comparison with matched seeds and
budgets:

1. released Student (no update);
2. direct optimizer alone;
3. direct optimizer plus adaptive external-teacher constraints;
4. fixed-weight constraint as an ablation, not as the primary method.

Use at least three seeds for the two trained primary arms. Report every metric
separately. The main success criterion is higher PickScore without a statistically
credible OCR or GenEval drop relative to the released Student. A weighted average
cannot establish success.

## Immediate operational actions

1. Do not submit another long training job yet.
2. Monitor `1035879` to completion and verify `TOPK` did not later skip or violate batch
   conservation.
3. Evaluate its checkpoints and record the Pareto trajectory.
4. Let Claude audit 0002–0004 in parallel, especially mask routing, frozen-reference
   timing, branch normalization, and EMA/raw checkpoint behavior.
5. Choose canonical OPSD or top-k OPSD from held-out evidence; then create sequence
   0006 for the official-path constraint implementation.

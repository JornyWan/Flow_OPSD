# 0001 — OPD project audit and next direction

Date: 2026-08-28  
Status: complete  
Scope: read-only audit of project documentation, source, results, current manuscript,
and the local sources for Flow-GRPO, DiffusionNFT, Flow-OPD, DiffusionOPD, and
DiffusionOPSD.

## Executive conclusion

The clearest next step is not another OPSD target construction, top-k variant, or
teacher-loss weighting rule. The project should test a routed hybrid in which direct
differentiable-reward optimization is responsible for improvement and external
teachers are used only as one-sided retention constraints for capabilities that direct
reward optimization cannot cover, especially OCR and GenEval.

The three previously mixed research questions must be separated:

| Line | Actual question | Current evidence |
|---|---|---|
| DiffusionOPD / MOPD | How to integrate OCR, GenEval, and preference specialists | Faithfully reproduced and currently the strongest foundation |
| CA-OPD / POE / CAGrad | How to weight routed teacher losses | No final-quality improvement; several conflict methods are harmful |
| RTC / OPSD / Best-of-N / top-k | Can reward-derived supervision replace external teachers | It raises its differentiable training reward but forgets OCR/GenEval |

The actual project-level target is to exceed the released multi-task Student without
trading away capabilities, not merely to improve one reward or move a trade-off point.
No existing arm has yet done that: the best multi-reward checkpoints are within the
released Student's noise band, while the single-reward RTC/OPSD family is roughly
17--23% worse on the six-axis aggregate.

## What is already reliable

The reproduction is unusually strong. There are 96 comparisons (16 targets, three
datasets, six metrics), 92/96 fall inside the estimated sampling-noise band, and the
released Student's largest six-metric deviation is 0.22%. The evaluation pipeline is
therefore not a plausible generic explanation for the major negative results.

The following experimental conclusions are reasonably stable:

1. CA temperature values 0.2, 0.5, and 1.0 are effectively tied. Temperature should
   not be advertised as an effective final-quality control knob.
2. Conflict/CAGrad variants are significantly harmful. Teacher-gradient conflict is
   not the demonstrated bottleneck in this benchmark.
3. POE lambda mainly moves the GenEval versus aesthetic/OCR/HPS exchange point. It has
   not produced a Pareto improvement over the released Student.
4. Reward-gradient direction is necessary, but the tested line search, certification,
   k-step, multi-query, and normalizer modifications have not yet yielded attributable
   end-to-end improvements. The weighted multi-query run was terminated around epoch
   12 and remains untested rather than disproved.
5. Moving other samples toward a best-of-N sibling is strongly harmful because group
   siblings have different initial noise and are different images. Latent interpolation
   toward the sibling is not an image correction direction.
6. A multi-task initialization slows forgetting, but sequential single-reward training
   continues to erode other capabilities.

## Required narrative corrections

### CA-OPD shared-state theory does not match hard routing

The manuscript describes per-teacher losses evaluated at shared on-policy states and
uses this to claim that uniform MOPD drives the student to a pointwise teacher-mean
velocity. The implementation instead selects one teacher slot, samples prompts from
that slot's task-specific dataset, and queries the corresponding routed teacher. The
actual objective is a sum of expectations under different task/prompt/state
distributions. It is not generally a pointwise mean of all teacher fields at one
`(x_t, c)`.

The current pointwise teacher-mean theorem therefore cannot carry the empirical story.
The CA-OPD manuscript should be archived as a negative/ablation study or theoretically
rewritten, not incrementally polished around the existing central claim.

### DiffusionOPSD radius criticism is overstated

DiffusionOPSD explicitly derives the adaptive branch normalizer's cancellation of a
positive scalar residual scale and presents this as scale robustness. The project's
rho sweeps and frozen-normalizer null result are best interpreted as evidence that the
tested moderate radius region lies on the intended robust plateau. They do not by
themselves show that trust-region geometry is meaningless in every regime.

Likewise, the experiments show that the tested target refinements did not improve
finite realization. They do not show that reward-gradient target construction is
unnecessary: both the official ablation and the local random/no-op controls show that
the reward-gradient direction is essential.

### Current top-k is not RAFT and does not reach nondifferentiable rewards

The running `topksmk` job printed `TOPK skipped` because it constructed 368/384 and
376/384 samples. That run therefore did not apply the intended filter. The current
source changes the algorithm to rank-local proportional selection; that can preserve
local batch size but is not exact global top-k.

More importantly, top-k only changes which samples enter training. The subsequent
target remains canonical reward-gradient OPSD, and OPA initialization still requires a
differentiable scorer. It is therefore top-k-filtered OPSD, not RAFT/RFT and not a
solution for OCR or GenEval. DiffusionNFT also reports that positive-only rejection
fine-tuning collapses without its negative-aware branch.

## Proposed method: constrained-retention OPD

Start from the released DiffusionOPD Student or a faithfully reproduced MOPD
checkpoint. Optimize a differentiable preference reward while imposing capability
retention constraints:

```text
maximize    R_pick(theta)
subject to  L_ocr_teacher(theta) <= baseline_ocr + epsilon_ocr
            L_geneval_teacher(theta) <= baseline_geneval + epsilon_geneval
```

The direct branch operates only on PickScore prompts. OCR and GenEval prompts retain
hard-routed teacher supervision. Crucially, their teacher loss is not an objective that
must always be driven lower. It is a guard that activates only when the current student
has moved farther from the corresponding teacher than the initial strong Student has.

The cleanest same-state constraint is:

```text
violation_m = KL(current_student, teacher_m)
              - KL(frozen_initial_student, teacher_m)
              - tolerance_m
retention_m = relu(violation_m)
```

Both KL terms are evaluated on the same current on-policy state. This avoids comparing
raw losses across tasks and avoids conflating the changing state distribution with the
reference-model baseline. The teacher is a preservation boundary rather than a target
that the student must chase forever.

## Decisive four-arm experiment

All arms must start from the identical released Student checkpoint and use identical
prompt/sample/update budgets:

1. MOPD continuation / no-new-objective compute control.
2. Direct reward optimization only.
3. Direct reward plus a fixed unconditional routed teacher loss.
4. Direct reward plus one-sided constrained retention.

Primary success criteria should be vector-valued rather than an aggregate ratio:

- PickScore improves beyond evaluation noise relative to the initial Student;
- OCR and GenEval remain within a preregistered one-sigma tolerance of the initial
  Student;
- at a matched-PickScore checkpoint, constrained retention Pareto-dominates the
  direct-only arm;
- an apparent mid-run lead is not extrapolated to convergence, and any positive final
  result is replicated.

Before these four full arms, a short smoke must establish loss and gradient scales,
masking, one-sided gate activity, global-step progression, and checkpoint creation.
Existing logs suggest RTC loss and routed-teacher loss differ by roughly three orders
of magnitude, so raw loss addition is not acceptable. The first smoke will measure
per-branch gradient norms; fixed-weight controls should be gradient-calibrated, while
the constrained arm should use explicit per-capability multipliers or dual variables.

## Engineering containment

The current mutable `reference_papers/DiffusionOPSD-main` tree is no longer a pristine
reference because experimental mq/bestof/top-k/frozen changes have been added in place.
The main trainers are large multi-branch files and there is no project-level Git
history; `.orig` and `.pre_*` backups are not an adequate experimental provenance
system.

All new Codex experiments are therefore recorded in this directory. Source files are
hashed, method changes are stored as patches, patches are applied to a job-local source
copy, submission commands and job IDs are logged, and corrections receive new ordered
entries instead of silently rewriting prior conclusions.


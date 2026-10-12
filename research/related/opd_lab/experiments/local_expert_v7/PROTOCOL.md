# Phase 7: bounded positive-learning and fixed-target controls

Frozen before GPU execution, 2026-09-13. Implements A+B of
`docs/research_reassessment_20260912.md`. This is a training diagnostic, not a
method efficacy experiment or a reproduction of the paper's reported numbers.
Old runs, labels, scorers, datasets, and the pinned upstream remain unchanged.

## A. Teacher capability and ordinary OPD

On all 64 phase3 training prompts, with their original noise seeds, generate
flow10/512 images using backbone CFG4.5, backbone CFG1, OCR teacher CFG4.5,
GenEval teacher CFG1 (original recipe), and GenEval teacher CFG4.5. Report paired
v1 component/proxy scores, with the known measurement limitations from phases4–6.
No validation/test prompts, oracle opportunity estimates, or score-based selection.
Separately copy each teacher's LoRA into the student and compare predictions at
matched CFG on the selected fixed queries. Equality checks representability, not
optimizer learning or superiority of every teacher action. Restore the initial
student and frozen behavior afterwards.

Run pristine-upstream SOPD-OCR with the existing managed lifecycle wrapper:
one L40-class GPU, first16 upstream rows in each OCR train/test split, 32 outer
epochs, train/sample batch1, three sampling batches per epoch, nine supervised
times per trajectory, upstream LR3e-4/AdamW/EMA/CFG4.5 unchanged. Eval flow10
every8 epochs, checkpoint every8 epochs. This deliberately small native subset
control uses the existing `smoke-examples` mechanism and its honest subset label.
The upstream OCR test prefix is distinct from the locked joint-v1 validation/test.
Native online training losses/rewards and saved evaluation images are diagnostic;
non-paired evaluation noise must not be interpreted as a paired improvement.
Finite-budget failure is not proof that ordinary OPD cannot learn.

## B. Selected fixed-target learnability

Selection is deterministic from the already frozen phase4 implementing-assistant
review: target preferred, at least one encoded false→true clause, no true→false,
no unknown encoded clauses or visibility labels, and no visibility regression.
This selects cases9/11/12/17/19 (five, not sixteen). Cases9/17 also have complete
visibility and all encoded clauses correct at the target; report them separately.
The other three improve encoded constraints but retain pre-existing visibility
failures. These labels are provisional implementing-assistant observations,
not independent human ground truth. This is a deliberately selected positive
set, not a sample for generalization or selection headroom.

Use the exact phase3 saved state/context/target and initial behavior LoRA. Each
of five independent single-query fits and one joint five-query fit starts from
the same initial student, optimizer, scaler and RNG. Never refresh behavior.
LR3e-5 only; original deterministic transition loss, AdamW and clipping unchanged.
No rejected queries, target rescaling, new scorer, EMA, on-policy refresh or extra
optimizer steps. Joint loss is the mean over five queries. This compares fit
ability at equal optimizer counts, NOT equal compute: report backward counts.
Audit at updates0/1/4/16/64/128, stop at128 without automatic budget extension.
Save target MSE relative to initial, target-direction projection gamma, prediction
residual, exact identities, checkpoints, single-updated-action/old-behavior-suffix
images and whole-updated-student images. The latter are secondary diagnostics at
this stage; no per-image best checkpoint selection. Main comparison is update128.
Replay the original noop and target endpoints before fitting, preserving their
old image hashes as references and reporting any numerical/image mismatch.

Engineering target: at128 at least4/5 queries have MSE ≤5% of initial and at
least4/5 preserve the annotated target repairs without losing initially correct
clauses. Report complete-visibility two-case subset separately. This threshold
is not a significance test and does not guarantee discrete correctness. Review
new images without proxy scores or arm labels; archive observations before
joining the key. Existing images may be familiar to the implementing assistant;
call this an arm-blinded provisional review, not an independent blind study.

## Decisions and resource ceiling

- If single fits fail, inspect optimization/CFG/target implementation before gate changes.
- If singles fit and joint fails, diagnose supervision interference before scaling up.
- If joint local repair survives but full generation loses it, next isolate a
  fixed short sigma interval and supervise all intervention states.
- Even success on all five is only learnability evidence; later require strong
  fixed-policy controls, fresh on-policy training and locked held-out evaluation.

At most three initial jobs, each one L40-class GPU and ≤1 hour: native32,
capability, fixed fitting. Accounting includes failed attempts. No automatic
new LR, training seed sweep or sampler sweep. A failure of execution requires
a fresh numbered attempt with the reason recorded, never patch a frozen run.

# Phase 3: fixed-query fitting and rejected-state preservation probe

Frozen before phase-3 GPU outcomes, 2026-09-11. This is a mechanism diagnostic,
not a new on-policy efficacy benchmark or native OPD reproduction.

## Question and fixed design

Can further fitting realize selected external-expert corrections, and can a
rejected-noop penalty reduce collateral movement while the behavior stays fixed?

Use the same SD3.5-M, external OCR/GenEval files, fp16/512, rank-32 alpha-64 LoRA,
seed 42, LR 3e-5, optimizer and flow10/CFG settings as phase 2. Start fresh.
Collect exactly one query for each of the 64 training prompts using a balanced
frozen schedule, new phase-3 noise/time keys, uniform time index 0..8. All queries
are collected under the initial frozen behavior. Compute/save all five candidates
(noop, OCR/GenEval alpha .5/1) once, then select with the unchanged protected v1
terminal rule. Every branch receives exactly the same states, targets and flags.

One full fitting update accumulates all 64 queries and divides by 64. This differs
from phase 2's 16-query on-policy batches: no direct causal comparison with old
checkpoints or their scores. Targets and behavior stay frozen for the whole probe.

| Predeclared checkpoint | Fitting updates | Rejected-noop coefficient |
|---|---:|---:|
| fit2 | 2 | 0 |
| fit2_keep | 2 | 1 |
| fit4 | 4 | 0 |
| fit4_keep | 4 | 1 |

Accepted queries retain .5*dt^2*velocity-MSE to the selected detached action;
rejected queries regress the original frozen behavior action when coefficient=1.
No accepted-count normalization, new reward, time weighting, EMA selection or LR
sweep. The initial rejected penalty has zero gradient in exact arithmetic.
Compute the common first update once, preserving its student/Adam/GradScaler/RNG
state; both branches restore that identical checkpoint. Numerically audit the
initial rejected prediction error, never silently assert fp16 arithmetic is exact.
Within each branch, fit2 is the saved prefix of fit4. Seven unique updates are
executed (one shared, three per branch); all four predefined checkpoints reported.
This shares computations, not additional evidence. Branch differences beyond the
shared prefix may still include nondeterministic CUDA arithmetic.

## Measurements and integrity

At initialization and after every actual update, log accepted target MSE and
rejected noop drift separately. At all four checkpoints, save/score the same-query
single-step intervention with the original behavior suffix and full flow10 rollout
using the same initial query noise. Preserve all 512 post-fit images, all candidate
images, decisions, hashes and actual optimizer steps. Verify behavior parameters
never change through student restore, fork audit or checkpoint preparation.
Snapshot/export operations must not accidentally refresh behavior.

Then generate base plus four raw checkpoints on all 32 validation prompts, seed 42,
flow10 and flow40: 320 ordinary images, no teachers/search at inference. This is
a small diagnostic probe, not a claim of sampler robustness; no DPM sweep here.
Use the established paired protocol, unchanged joint_constraints_v1 and prompt
bootstrap. Predeclared contrasts: fit2_keep-fit2, fit4_keep-fit4, fit4-fit2,
fit4_keep-fit2_keep, plus each vs base. Intervals do not include training-seed
variation and are not corrected for multiple testing. Test remains unused.

Do not use Tesseract as a new target judge: phase 2 established substantial false
negatives. Instead preselect two training prompt IDs per task family by SHA256
without outcomes, and inspect all five full-generation images per ID (40 images).
Hide method identities and scores; show prompt. Before revealing the key, record
text exactness/extra text, explicit scene constraints, full object visibility and
uncertainty separately. This extends coverage of the implementing-assistant audit,
not the automatic v1 metric, and is not independent human ground truth. Do not
retroactively rewrite any training/evaluation labels. Calibration/measurement
coverage work stays on training images; previously used validation stays development.

## Interpretation and next action

First compare keep/no-keep at the same fitting count. Check actual accepted fit
and rejected drift, rather than only the scalar training objective. If the penalty
reduces latent drift but not endpoint regressions, it is insufficient as an image
constraint guarantee. If more fitting improves MSE but not ordinary generation,
do not equate fit with efficacy. A positive local result still requires on-policy
tests, reliable independent measurement and multiple training seeds.

Report the entire predefined matrix even if negative. Do not select a preferred
fit count/weight using these outcomes and claim a locked-test gain. No longer
training or additional sweeps in this phase before inspecting the probe.

## Execution and budget

One L40-class GPU, up to 75 minutes for collection/fitting/audits; a dependent
one-GPU evaluation job, up to 40 minutes. Frozen source/config/data/plan and a
fresh run per attempt. Both branch marginal operation counts and shared costs
logged; within-count keep/no-keep is the main compute comparison. Wall time
includes diagnostics, validation, loading and failures; not an optimized native
training-efficiency claim. Source repos and old campaigns stay unchanged.

# Phase74 official and local OPD numerical acceptance

Frozen engineering protocol, not a student performance comparison. One L40S,
1800 requested seconds maximum, no retry/requeue/timer. CPU prerequisite: actual
pinned loss reduction and Accelerate 27-backward accumulation on 81 synthetic
states, matching gradients and one Adam update. Preserve the native EMA call
placement; disclose its difference from once-per-update local EMA, do not fix
archived native behavior or infer it caused RAW evaluation failures.

Actual SD3.5M starts from fresh rank32/alpha64 LoRA. First three TRAIN prompts from
the frozen paper_joint_v2_native text view, CPU noise seeds74010..74012, text128,
512px, FP16, Flow10, studentCFG4.5, GenEvalCFG1. Same initial adapter, states,
contexts, targets, LR3e-4, Adam betas(.9,.999), eps1e-8, wd1e-4, clip1.

Each trajectory provides all nine supervised states (j0..8): 27 state examples.
This intentionally uses microbatch1, not official microbatch3; it is hardware
acceptance, not a reproduction of the paper's 81 states/card or 8-card batch.
Official arm executes AST-extracted pinned student/teacher transition functions
and actual reduction. Local arm executes existing backend predict and transition
loss. Reset to identical initial weights and fresh Adam/scaler before each arm.
One averaged update per branch, all 27 states retained, no selector/reward/extra
loss. CPU checks cover real Accelerate accumulation; GPU runner explicitly
divides by27 and does not claim to execute the whole upstream training main.

Predeclared acceptance: teacher mean maxabs<=2e-6; aggregate preclip gradient
relativeL2<=1e-3; post-Adam weights maxabs<=3e-6; per-state loss maxabs<=2e-6;
finite values and one successful Adam update per branch. Export local RAW,
reload exact adapter payload, physically remove all other adapters and require
exact ten-step latent replay from a new fixed seed74100. No image decoding or
quality/task efficacy inference. No long training, no confirmation/final use.

CPU EMA audit exercises16 synthetic update values including actual repeated
calls at global steps7/15, distinct from GPU's single RAW update. Previous native
GPU save/load acceptance remains evidence only for its historical configuration.

Sources/framework/teacher adapter/config and prompt bytes are hashed. Backbone
inventory is recorded, not full-content hashed; this is an explicitly limited
engineering check, not a final publication asset audit. Failure: preserve outputs,
stop and diagnose, no automatic GPU retry or threshold relaxation. Success:
prepare sufficient-training task-adapted SOPD baseline with fresh LoRA and native
all-time supervision; do not submit that larger study from this one-shot runner.

Probe minilab-gpu/sablab-gpu/scu-gpu/preempt_gpu immediately before submission.
Keep sablab allocated+pending across projects<=6. Prefer ordinary queues; permit
preemption only with stop-on-interruption and no retry. Submit only with credible
estimated start within one hour. No other jobs are moved or cancelled.

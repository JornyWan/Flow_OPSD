# Phase19: teacher opportunity on the exact Phase18 primary states

Frozen before generating new teacher images. This is a post-result mechanism diagnosis,
not another independent evaluation: all12 primary phase18 noise states are exposed
and now development data. No filtering by old outcomes, confidence, harm or uncertainty.
Six prompt identities, two seeds each; exclude added fitting diagnostics and other cohorts.

Keep SD3.5-M,512px,fp16,flow10,index3,alpha1,student/behavior CFG4.5,OCR CFG4.5,
GenEval CFG1. Use raw joint128 for prefix and suffix; use the frozen phase18 multi
step128 model only for its one predicted action. No optimization, model selection,
behavior refresh after initial load, time/strength/sampler sweep or new network.
The only new scientific outputs are two teacher actions/endpoints per state (24total).
Each alpha1 action is the direct teacher prediction, matching phase16 branch_actions.

Before ANY teacher prediction, reproduce all12 baseline endpoints and their saved
state/context/noise/noop/step sizes exactly, then all12 multi single-action endpoints
exactly. Keep every reference and gate counts. Same L40S/Torch/numerical evaluation
mode as phase18, CUBLAS_WORKSPACE_CONFIG=:4096:8, no deterministic fitting context.
A failed replay ends the attempt; no tolerance relaxation or failed-image replacement.

After the gate, query OCR and GenEval once each on every same saved index3 state and
finish with the frozen joint128 suffix. Save tensors, predictions, image hashes,
noise/context hashes, student/behavior hashes and relative student-to-teacher action MSE.
The residual MSE denominator is that teacher's distance from the initial/noop action.
Report both teachers across all12 states before any descriptive safe-target subset.

Expected measured counts: student132,behavior228,OCR12,GenEval12,
CFG forward examples756,decode/verifier48,text_encode_prompts12,backward0.
24 exact references +24 new images,24 teacher queries,0 updates. One L40 allocation,
15-minute cap. Count all failed allocations and distinguish inherited phase18 costs.

Review12 four-view native512px sheets. Each includes baseline, multi single action,
OCR and GenEval in randomized slots. Previous24 image labels are inherited unchanged;
new teacher identity/metrics/automatic scores remain hidden until all48 observations
are frozen. Any further identical pixels inherit existing labels. Use the same original
rubric including exact object counts, full visibility, extra letters and unknowns.
These are provisional implementing-assistant labels, not independent human truth.

Predeclared outputs: per-mode complete scene, safe encoded repair, known harm,
candidate/comparison uncertainty; teacher-safe availability and teacher-safe complete
availability, separately. Report partial-only teacher repairs; oracle completeness
including noop is an upper-bound diagnostic, never learned performance. Count a known
student-learning gap only when a teacher safely completes and the learned single action
is determinate and incomplete. Unknown baseline/teacher/student is not evidence of failure.
Summarize by prompt-cluster bootstrap2000/seed0 and retain all12 per-state comparisons.

Interpretation: safe teacher completion with determinate student failure motivates
state-generalization work; no safe teacher completion motivates candidate/constraint
composition before more fitting. Mixed or uncertain cases remain mixed/uncertain.
This small diagnostic cannot prove a universal cause, deployable gating, or sampler robustness.
Phase18 results/labels stay frozen. No reusing these seeds as untouched test data later.

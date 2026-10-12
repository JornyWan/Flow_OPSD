# Phase80: fixed-state gradient mechanism, no learning/evaluation claim

Frozen after Phase79 full audit: EMA joint50vs49,16wins/15losses,paired interval
crosses0; RAW43vs45; all five EMA quality intervals cross0, RAW HPS negative.
Native and block training are not compute/information matched. This diagnostic
asks whether endpoint residuals can conceal first-transition mismatch, and how
the second-step input Jacobian contributes to the gradient. It does not test
student improvement, and cannot prove this mechanism caused Phase79 outcomes.

16 TRAIN prompts: first4 sorted IDs in each color/count/multi/relation family,
not selected on dev results. CPU FP32 seeds80080..80095, initial actual saved
S0 native LoRA generates zero-noise Flow10 anchors j0/3/6/8.64 common anchors.
Compare actual saved S0, Phase77 finalEMA, Phase79 finalEMA on exactly those
states, with fixed detached external GenEval teacher first/second targets from
its own successive states. Same512px/text128/studentCFG4.5/teacherCFG1.0/native
AMP outputFP32 before CFG. Shared pure base model and rank32alpha64 student.
Common anchors are prospective S0 states, NOT historical sampled/accepted
states or complete own-policy distributions of either final model.

Re-use pinned original native mean functions and qualified block wrapper for
GPU acceptance. Disable backend layer checkpointing; first transition uses
nonreentrant activation recomputation, as qualified block. Core calls
autograd.grad, never optimizer.step or backward accumulation. Fixed loss scale
65536 prevents avoidable FP16 underflow; report unscaled per-state gradients,
not gradients after81-state aggregation/clipping/Adam. No parameter.grad mutation;
loaded and final adapters equal exact source arrays. Targets and anchors detached.

For each state report local one-step and horizon-normalized two-step loss;
full/first-direct/second-direct/input-Jacobian/no-input-Jacobian gradient norms;
full-vs-local,full-vs-no-Jacobian,direct-first-vs-second,direct-vs-Jacobian cosines.
Undefined zero-gradient cosines remain null and counts reported. Entire-middle
detach removes first direct AND input-Jacobian terms; do not conflate it with
removing just the Jacobian term. All192records required. Aggregate by training
prompt (16 independent prompts, four anchors each), with all times reported.
No significance gate or new claim based on choosing favourable diagnostic states.

First state of EACH weight: core full gradient must match existing canonical
two-step wrapper relativeL2<=1e-5 and normalized loss abs<=1e-7; values finite.
Recomputed second call on detached middle must have exact same forward value.
Original qualified checks plus new CPU affine analytic decomposition, loss-scale,
nonreentrant recomputation, cancellation and cohort tests precede submission.
No images/decode/task scoring/quality inference; Phase79 already reports full
tasks/five quality and remains unmodified. confirmation/final sealed.

If repeatable endpoint/local discrepancy or Jacobian opposition is observed,
it motivates a separately frozen single-variable learning comparison, not an
automatic parameter sweep or proof of improved quality. Absence of such evidence
argues against changing gradient flow merely because a toy predicts benefit.

One L40S<=4500requested seconds; no requeue/retry, no training or newtimer/agent.
New reservation must be made atomically from autonomy_20261011_2h before sbatch;
count request even failure, no refund. Immediately probe four partitions and
sablab allocated+pending<=6. Require estimated completion withincampaign expiry,
don't move others. Sources,weights,teacher files,prompts hashed; backbone uses
metadata inventory for large files, explicitly not full-content fingerprint.
Incomplete/technical failure stops and preserves evidence; no automatic retry.

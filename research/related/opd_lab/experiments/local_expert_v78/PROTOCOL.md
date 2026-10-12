# Phase78: native-integrated two-step closed-loop learning pilot

This is the first GPU implementation check of the proposed short-block learning
component. It is NOT the complete terminal-verification idea, an unchanged OPD
baseline, or an efficacy test. Phase77 joint17->49/128 is the native comparator.
User2026-10-11 explicitly accepts a small CLIP drop as a tradeoff; future decisions
report all metrics rather than requiring every mean to rise. Original Phase77
criteria/results remain immutable. Do not tune a new tolerance to that dev result.

Start fresh SD3.5M/LoRA seed42, original official SOPD-GenEval backbone recipe:
512 training prompts, batch3,3 batches/epoch, Flow10, studentCFG4.5/teacherCFG1,
512px/text128, FP16 output conversion, LR3e-4, native Adam/clip/EMA. No Phase77
continuation.8 actual updates/648 anchor examples,216 gradient-accumulated blocks.
No time-index filtering, verifier, reward change, added expert or learning-rate scan.

Both branches start at detached behavior z[j], j0..8. Teacher takes its first
original native transition, then queries the SAME frozen teacher at its own new
state for j+1; final target detached. Student uses original native transition at
z[j], then at its OWN differentiable intermediate state; backward traverses both
calls. Do not reuse recorded behavior z[j+1] for the second prediction. Supported
only noise_level0/Flow. Endpoint difference scaled by dt[j]/(dt[j]+dt[j+1]) before
original0.5 mean-square reduction: preserves original first-step time weighting
for constant velocity errors. One-step limit returns original computations exactly.
Nonlinear rollout changes are the hypothesis, not a guarantee of terminal gain.

All9 anchor indices retained (including j8 whose second step uses j9). Original
single-step student trained only j0..8; terminal j9 inside two-step blocks is an
explicit change in state coverage. Teacher10 original plus9 additional queries
per trajectory=19, versus10 native.81 anchor examples/update,162 logical student
forward examples and81 first-step recomputations; at batch3 this is54+27 calls,
versus27 native. CFG doubles transformer internal student examples. Sampling
cost remains original. Same optimizer-update count is NOT equal compute/query cost.
No attempt to attribute gains uniquely to graph coupling without later matched
information/compute and stop-gradient controls.

Checkpoint first forward with non-reentrant PyTorch activation recomputation to
avoid retaining two full transformer activation sets on48GB L40S. No per-layer
native model gradient-checkpointing change. Before first update run real microbatch1
checks: original versus one-step wrapper output/gradient exact; uncheckpointed vs
checkpointed full two-step output<=1e-6 and gradient relL2<=1e-5, nonzero finite
gradient, teacher detached. Fork/restore CUDA+CPU RNG, no optimizer update in test.
CPU analytic checks verify chain derivative, no behavior/teacher gradient,
constant-velocity normalization, off-trajectory inputs and native loss reduction.

The actual main then executes8 batch3 updates, original27 backwards/update and
original repeated EMA cadence. Read-only native audit checks actual Adam1..8,
LR3e-4, finite losses and EMA replay/export<=1e-6, changed RAW/EMA; savefinal8 only.
Record actual peak memory/time to budget a subsequent complete256-update study.
No decode/score for quality or dev here;72 original training previews remain,
agent reads0. No inference about joint/OCR/five-quality or superiority from a
passing engineering pilot. Confirmation512/final768 remain sealed.

One new L40S job <=1800 requested seconds, no retry/requeue/timer or automatic
follow-on GPU. Probe minilab-gpu/sablab-gpu/scu-gpu/preempt_gpu before submission;
sablab allocated+pending<=6, prefer ordinary if similar, compatible start<=24h.
Preemption/OOM/nonfinite/skippedupdate/contract failure => stop without retry,
lowering batch, changing LR or detaching the middle. Completed inputs immutable;
CPU analysis repair independent. On pass, freeze next matched256 learning protocol
using measured compute; never assume matching256 updates matches GPU-hours.

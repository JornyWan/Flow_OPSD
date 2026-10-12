# Phase77: adequately exposed native OPD task-adaptation baseline

Question: after256 actual updates with dense original OPD coverage, does the
GenEval expert produce a better single student on joint text/structure constraints?
This establishes the comparator needed before fixed two-step closed-loop learning.
It is not the proposed method, not a convergence guarantee, not a paper-scale
8-GPU reproduction, and not continuation from the published OPD student.

Fresh SD3.5M, seed42, native rank32/alpha64 Gaussian-A zero-B LoRA; unchanged pinned
SOPD-GenEval main commit0c85bf0193a6da24bb1f72152b612d353a154999, one L40S.
Original batch3,3 sampling batches/epoch,27 backward calls and81 states/update;
256 updates,2304 draws,20736 states,6912 backwards from512 training prompts.
LR3e-4, original optimizer/FP16/clip/time weighting, studentCFG4.5/teacherCFG1,
Flow10/text128/512px, original repeated-step EMA. Native lifecycle/local assets
and observation hooks only; stripped computation AST must equal original.
Save final256 only; final EMA primary, RAW secondary. No best-checkpoint choice.
Training decodes2304 images automatically; agent reads zero. Monitoring proxy
reward is empty; dev not used for updates, target selection or early stopping.

Evaluate actual saved fresh S0/final256 EMA/final256 RAW,128 identical existing dev
prompts/noise per arm:384 new images. Shared explicit FP32 CFG, FP16 model,
Flow40/CFG4.5/text128/512px, one default adapter physically deployed; this common
evaluation path is not the literal complete native paper evaluation pipeline.
Old Flow10/OPD-student outcomes cannot substitute for the new S0.
Report joint/OCR exact/structure/CER/A and B unknown plus all PickScore/CLIP/
aesthetic/HPSv2.1/ImageReward values, paired family-stratified prompt intervals.
Intervals are exploratory: same128 prompts repeatedly used, one training seed.
Two fixed historical task references require exact A/B/evidence parity.
Eight fixed historical reference images/40 quality scores must replay within1e-4;
include full per-image provenance and no dropped/error rows. No evaluator changes.

Primary contrast EMA256-S0 joint; positive mean alone does not establish robust
improvement. A fully positive paired interval with no mean decreases in five
quality metrics is a promising exploratory baseline, not independent validation.
A cross-zero interval is unresolved; a wholly negative interval is adverse.
Always report RAW and all endpoints even if unfavorable. Stronger structure with
worse OCR/joint distinguishes expert task tradeoff from a successful joint model.
If task/quality decline, audit coverage/loss and expert tradeoff before claiming
more training or closed-loop fitting solves it. Do not alter thresholds or swap
primary to RAW. Next new-method comparison needs same initial weights, prompts,
actualLR, update/data budget, full evaluation and explicit extra-compute accounting.

One new GPU job <=10800 L40S seconds, no retry/requeue/timer/fallback. Pilot measured
training6054s extrapolated; remaining4746s covers load/generation/scoring with a
hard stop. Stop on OOM, AMP skipped update, source/identity mismatch, scoring error,
reference parity failure, preemption or timeout; retain incomplete evidence and
no efficacy conclusion. CPU report fixes only in a new independent record.
Probe all four partitions before submission, expected start<=24h, sablab cross-
project allocated+pending<=6; ordinary queue preferred, preemption stops/no retry.
Do not cancel/move other jobs. Source/teacher/data hashes frozen; backbone large-
file byte/mtime inventory limitation remains explicit. Confirmation512/final768
sealed. No visual reads, new agents, timers or remote evaluation.

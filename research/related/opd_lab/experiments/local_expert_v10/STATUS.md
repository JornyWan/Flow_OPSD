# Phase10 complete: fresh sparse supervision retains the repair in full generation

Job1049122 completed successfully:1140 allocated seconds on one L40S,
0.316667 GPU-hours, no failed attempt. Dense32 and sparse320 each used1600
backward queries from the same joint128 weights and reset AdamW/scaler/RNG.
Sparse32 is the predeclared equal-update control.66 CPU tests passed.

**The newly verified apple correction survives ordinary full student generation
after sparse fitting at BOTH32 and320 updates.** CLOUD and complete visibility
remain correct. No expert intervention, selection or inference mask is used in
those full generations. All four controls preserve their previously true clauses
and protected visibility. These are the five selected training prompts/seeds.

| State | Apple local endpoint | Apple ordinary full generation | Protected controls in full /4 | Fully correct full scenes /5 |
|---|---|---|---:|---:|
|joint128 baseline|not a new fit|fails relation|4|2|
|sparse32|fails relation|passes|4|3|
|sparse320|unknown: possible extra bowl above|passes|4|4|
|dense32|passes|fails: two bowls|4|3|

The sparse320 and dense32 full cup scenes also improve visibility; the bicycle's
extra plate lettering and cropped car persist. All30 endpoint images were new
unique images and reviewed without arm/update/mode/score. Labels were frozen
before joining keys; the uncertain local sparse320 scene stays unknown. Review
is by the implementing assistant, not independent human annotation.

## Fitting and drift explain the limits of the previous hypothesis

| Arm/checkpoint | Apple query residual MSE / initial | Mean preservation-point MSE | Query-state drift MSE | All50 transition loss / initial |
|---|---:|---:|---:|---:|
|sparse32|12.0965%|0.00928478|0.000924833|2.34371|
|sparse320|1.04323%|0.0126091|0.000901124|3.46365|
|dense32|29.7435%|0.000829386|0.0000278251|0.514110|

Dense supervision reduces fixed-trajectory error and query-state drift, but its
full generation still fails. Sparse supervision changes the approach and suffix
more, yet full generation succeeds. Thus preserving a previous path, minimizing
local action residual, and improving terminal semantics are distinct objectives.
The sparse local/full mismatch also shows that a frozen-state, frozen-suffix
endpoint cannot replace evaluation of the updated student's ordinary generation.

This budget does NOT establish that dense supervision is intrinsically worse:
its expert action still has29.7% residual, and averaging over50 points changes
relative preservation pressure while giving only32 optimizer updates. Do not
silently increase its budget or relabel these confounds after observing outcomes.
The supported next direction is fresh verified external-expert corrections with
sparse fitting and explicit controls, followed by new-seed/development evaluation.

## Recorded checks and cost

Five phase9 full trajectories and six reference images replayed exactly. All50
stored transitions and49 preservation targets were checked at their own states.
Both initial checkpoints match; full terminal Adam/scaler/RNG and raw LoRA
exports are verified.352 updates,3200 backward queries,3600 student forwards,
223 frozen-behavior forwards,7646 CFG examples,36 decoded/verifier images
(six references plus30 endpoints),250 fixed-point metric rows. No new teacher
queries; the one reused expert action's acquisition remains recorded in phase9.

Per-arm worker time including its audits/exports: dense521.52s, sparse567.85s.
The allocated total also includes model loading/reference checks. GPU-hours
and backward-query counts are both reported; the arms are not exactly FLOP matched.
Historical phase6/7/8/9 indexed artifacts and upstream source remain unchanged.

See results.json, verification.json, ledger.jsonl, fitting.png,
apple_endpoints.png and ../../docs/phase10_runbook.md. No phase11 job is pending.
This is a positive selected-case learning result, not online-selector superiority,
held-out efficacy or cross-sampler robustness. See NEXT.md for the next gate.

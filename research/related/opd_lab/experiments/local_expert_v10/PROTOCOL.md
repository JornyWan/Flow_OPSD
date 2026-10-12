# Phase10: can a freshly verified repair survive fitting?

Freeze before GPU execution. This follows phase9/NEXT: one selected apple/bowl
repair and four preservation controls, not an online method comparison.
All inputs, phase9 labels, source snapshots and checkpoint hashes are retained.
Do not edit previous campaigns or infer held-out/sampler improvement from this test.

Initial weights for BOTH arms: phase7 joint step128. Reset AdamW, scaler and RNG
to the same initial state for each arm. This is a weight warm start, not a full
optimizer resume. LR3e-5, betas(.9,.999), epsilon1e-8, weight_decay1e-4,
max_grad_norm1, fp16, raw LoRA rank32/alpha64, seed42; no EMA or schedule change.
Same flow10,512px,CFG4.5 and CPU FP32 initial noise as phases8/9.

Use phase9 branch19 geneval@1, which has a protected relation repair, and base
branches09/11/12/17. Preserve prompts/seeds/query indices2/0/0/1/3. The apple
target contains current-student prefix, ONE external GenEval action at index3,
and current-student continuation on the corrected states. Never use baseline
suffix actions on those states. Controls use their entire unchanged baseline.
Validate every stored transition and all non-expert actions against frozen
joint128 predictions. Reproduce five baseline images and the apple target exactly.

| Arm | States per update | Updates | Backward queries | Audits/endpoints |
|---|---:|---:|---:|---|
|sparse|5 original query indices on phase9 states|320|1600|0(metrics only),32,320|
|dense|all10 steps of all5 chosen trajectories|32|1600|0(metrics only),32|

One optimizer update accumulates the mean over all listed points. Use original
deterministic OPD transition loss .5*mean((delta_sigma*(v-target))^2) per point,
then average over5 or50 points. Preservation weight1, expert weight1; no accepted-
count normalization. Dense thus changes temporal coverage and relative loss mass;
equal backward queries do not imply equal optimization difficulty or exact FLOPs.
Sparse32 is an equal-update control, not an alternative selected per image.
No automatic extra LR/updates, best-checkpoint search or inference mask.

At each audit evaluate all50 fixed state/action residuals, the apple query's
relative target MSE/direction, full generated trajectories and query-state drift.
At three nonzero audits save five full and five single-action endpoints (30 total).
Single-action evaluation uses the phase9 query state, learned action, then frozen
joint128 continuation. Full generation uses the learned student at every step.
Save full optimizer/scaler/RNG at each arm's terminal checkpoint and raw weights
at all five audits. Keep complete endpoint trajectories and exact counters.

Primary comparison: sparse320 versus dense32 full apple repair plus preservation
of all previously true clauses and full visibility on four controls. Report
sparse32 and local endpoints separately. All30 endpoints get arm/step/mode-hidden
review under REVIEW_RUBRIC.md, frozen before key join; unknown is not a success.
Exact pixel matches may inherit frozen phase9 labels. Other prompt details and
existing bicycle text/cup cropping failures must remain visible in the report.

If both underfit, this does not settle sparse-versus-trajectory coverage. If local
repair is learned but full repair fails, inspect saved approach-state drift. If
dense alone preserves the full repair, it supports this coverage mechanism only
on this selected seed/case. A new independent cohort is needed before efficacy claims.

One L40-class GPU, four CPUs,96GB, one-hour maximum; both arms sequentially in a
single run. Account startup, audits, exports and any failed attempt.3200 total
backward queries,352 optimizer updates, no new teacher calls (one reused expert
action, whose phase9 acquisition cost remains recorded separately).

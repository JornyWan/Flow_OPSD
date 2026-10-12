# Phase9 complete: a fresh expert query repairs the drifted apple state

One L40S job1048968 completed in66 allocated seconds (.018333 GPU-hours).
No training or failed attempt.25 images,216 model forwards/427 CFG examples;
all5 current-student baselines exactly match phase8. The20 unseen candidates
were reviewed with candidate identity and scores hidden, then observations
were frozen before joining the key. Full CPU suite:64 passed.

**At the current joint128 student's apple/bowl state, GenEval alpha1 repairs the
relation and preserves CLOUD plus complete visibility under the CURRENT student
suffix.** OCR alpha.5/1 and GenEval alpha.5 leave the apple below the bowl.
Unchanged-v1 proxy utility agrees on this case: baseline .333333, GenEval.5
.4375, GenEval1 1.0. Proxy scores alone did not define the visual conclusion.

This follows phase8's causal pattern: keeping the old prefix preserves repair,
whereas the new prefix loses it even with the old suffix. Together the probes
support checking fresh teacher targets on actual student states instead of
assuming the original frozen-state target remains sufficient. On-policy refresh
already exists in the framework; this is mechanism evidence, not a new algorithm.

| Fresh candidate, applied to all5 cases | Protected encoded repairs | Safe visibility gains | Known-harm cases | Unknown cases |
|---|---:|---:|---:|---:|
|OCR .5|0|0|1|1|
|OCR1|0|2|2|0|
|GenEval .5|0|0|1|0|
|GenEval1|1|0|1|0|

GenEval1 repairs apple/bowl but introduces extra text in cup/bottle. OCR1 improves
visibility for cup/bottle and bicycle/car, yet changes BLOOM to BILOOM on the
bowl example and reduces two bananas to one. OCR.5 also loses a complete banana.
Safe visibility gains may retain pre-existing encoded failures, such as plate
text. The bicycle's strict only-word problem is still not clearly solved.

These are candidate interventions, not a newly trained model. The frozen
joint128 student still has its old whole-generation behavior until new training
is performed. Five selected training cases and implementing-assistant visual
labels do not establish selector headroom on the full distribution, generalization
or sampler robustness. Unknown has not been converted into success or definite harm.

**Next:** compare fresh single-state supervision against supervision of the
complete verified branch trajectory, from the SAME joint128 weights. The latter
also teaches the prefix and the continuation at its corrected states, addressing
the observed drift. Use compute-matched controls; see NEXT.md. Do not introduce
a global2/4-step inference mask: phase8 found no clear aggregate advantage.

This turn completed phases8+9:60 generated images total, .044444 allocated GPU-hours
(160 seconds), no parameter updates. See results.json, observations.json,
PROTOCOL.md, ledger.jsonl and ../../docs/phase8_9_runbook.md.

# Phase8: frozen-student prefix/suffix and interval diagnosis

Frozen before GPU execution, 2026-09-13, following phase7/NEXT.md. No training,
new teacher queries, reward changes or new prompt selection. Use exactly joint128
from le7_fit_001 and the phase3 initial behavior, same five queries/seeds/contexts,
flow10/512/CFG4.5/fp16. Queries have indices2/0/0/1/3. All old artifacts immutable.

For each query k, run all seven rules from the same initial noise:

| Rule | Indices using updated student | Others |
|---|---|---|
|base|none|old behavior|
|local|k only|old behavior|
|suffix|k through final step|old behavior|
|prefix|0 through k, INCLUDING the query|old behavior|
|full|all steps|old behavior|
|window2|k,k+1|old behavior|
|window4|k through k+3|old behavior|

Base/local/full must exactly replay the15 corresponding phase7 image hashes.
At k=0, prefix=local and suffix=full: four exact-duplicate checks. Execute both
duplicates to check implementation, but do not count them as independent evidence.
No parameter updates: verify student and behavior hashes before/after. Save actual
sigma intervals/NFE, state/action trajectories, and state displacement at the
query. Distances to the base trajectory at different states are descriptive
drift measures, not supervised target error or evidence of causal harm per step.

Thirty experimental images plus five base replays,350 model forwards total
(700 CFG examples);35 unchanged-v1 verifier calls. One L40-class GPU, ≤30 minutes.
Include startup/failure accounting; an execution failure requires a fresh attempt.

Primary readout: strict repair retention under the phase7 review rubric for ALL
five cases, with the two complete-visibility targets separately reported. The
bicycle target's old phase4 text-label defect is retained as a limitation, not
silently relabelled or removed. No statistical efficacy or generalization claim.

Reuse frozen phase7 visual labels only for bit-identical images. Shuffle all
other unique images, hiding execution mode/proxy values; review and hash the
observations before joining the key. Exact duplicates share one observation.
Report mode rows separately but do not inflate sample count. This remains a
provisional implementing-assistant review, not independent human ground truth.

Interpret each case using the four local/suffix/prefix/full outcomes. A successful
old-prefix suffix with a failing new-prefix old-suffix implicates changed query
states; a failing old-prefix suffix already shows later updates can lose repair.
If both mixed rules work but full fails, interaction matters. Other patterns
must be reported without forcing these diagnoses. Inspect window2/window4 on the
same whole set, never choose interval length per image. A short interval that
rescues one case but damages others is not a global remedy.

This is a mechanism experiment on selected training cases, not a new deployment
method. Short-interval expert sequence training is contingent on the evidence;
no critic/learned selector/new VLM or sampler sweep is included.

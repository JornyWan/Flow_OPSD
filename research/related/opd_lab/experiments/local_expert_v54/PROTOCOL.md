# Phase54: fixed fitting-budget continuation, 4 -> 16

Claim C2 / mechanism: does better fitting realize accepted terminal repairs, and
does verified selection beat count-matched random after equal extra fitting?
Only intended change is fitting length. Both arms preserve their own Phase52
student, Adam, scaler and RNG at step4; continue exactly12 updates to step16.
Same64 training queries, original seven-target masks matched by family/domain,
same target actions, LR1e-4, transition loss /64 slots, clipping, FP16 precision,
anchor and time window. No retention term, target selection change, pool refresh,
EMA, checkpoint search, LR search, reward or verifier tuning. Rejected slots
remain unsupervised. This is a reused-pool mechanism experiment, not native OPD.

One backbone/two student adapters; learner only Flow10 query1, published anchor
elsewhere. Flow shift3 CFG4.5 512. Teacher and verifier absent at inference.
Two arms x (all64 original training prompts + all128 previously exposed dev
prompts), one original noise each =384 new images. Training anchor prefix may
be cached; new learner action and all subsequent anchor queries recomputed.
Independent exported-LoRA reload, exact saved query/prefix checks, full route
replay, checkpoint action metrics and optimizer/RNG identity audited. Repeat
one resumed update twice then restore before actual fitting;2 audit updates
and14 backward queries count as cost, not scientific training steps.

Primary: verified16 minus random16 B joint on128 dev prompts, family-stratified
paired prompt bootstrap20000,seed54055, descriptive95% CI. The development gate
requires positive delta and lower>0; does not authorize paper efficacy claims.
Training analysis seed54054: full64 endpoints, accepted7/rejected57 breakdown,
retention of the original four verified direct joint repairs and two random
direct repairs, action MSE and drift. Do not select a new subset after fitting.
Secondary: each16 minus own4, each16 minus anchor, verified16 minus fixed4 and
teacher-local references; original4-step controls explicitly have less fitting
compute. Report all five quality metrics, OCR/structure, A lower/upper/unknowns,
repairs and harms with conditional denominators, family/template subgroups.
All secondary intervals descriptive, no multiplicity adjustment. Any quality
CI entirely below zero is reported as harm; absence of significance does not
establish quality noninferiority. A/B share evaluator components.

Reuse frozen Phase51/52/53 endpoints, scores and metrics with ORIGINAL hashes
and evaluator identities; never relabel old cached predictions as new scores.
No confirmation512/final768 prompts, second noise, human labels or direct agent
image reads. Zero task errors required; no dropping. All384 new images receive
all five local quality scores, using the existing frozen assets/parity checks.

Decision: MSE improvement with accepted terminal repairs still missing argues
against blindly increasing fitting. Training realization without dev advantage
points next to coverage/new training pools, not a success claim. A positive dev
signal motivates a separate multi-run/compute-matched protocol. No automatic
next GPU job, checkpoint selection, failed-job retry or budget extension.

One L40 job, hard2400 GPU seconds, including loading/scoring/failure. Separate
Phase54 budget under the user's continuing-work authorization. The prior
overnight timer expired before approval completed and performed zero periodic
wakes; it is NOT restarted or extended. No claim of overnight experiments.
Phase53 stays independently accounted at772/1200 seconds and immutable.

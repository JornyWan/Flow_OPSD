# Phase7 completed: sufficient fitting recovers useful repairs

2026-09-13. Three jobs completed, **0.458333 L40S GPU-hours** including startup;
no failed GPU attempts. Full CPU suite:60 passed. Exact commands are in this
campaign's ledger; source/config/targets/images/checkpoints are separately hashed.

**Sufficiently fitted local corrections can survive actual generation on these
selected training cases.** This does not establish policy generalization,
selection headroom over a strong fixed strategy, or sampler robustness.

| Final update128 | Five independent single-query fits | One joint five-query fit |
|---|---:|---:|
|Mean target MSE / initial MSE|1.52%|3.38%|
|Queries with MSE ≤5% of initial|5/5|4/5|
|Mean target-direction gamma|.9863|.9667|
|Local repair retained, provisional arm-blinded review|3/5|4/5|
|Full generation repair retained|2/5|3/5|
|Two complete-visibility target cases, local / full|1/2 / 1/2|2/2 / 2/2|
|Same query meets both MSE and local-repair checks|3/5|3/5|

The protocol's separate ≥4/5 MSE and ≥4/5 local-repair checks are met by the
joint fit. They are NOT the same four queries: bowl case9 has5.75% residual MSE
but repairs successfully; bicycle case12 fits within4.04% but fails strict text
review. This aggregate engineering gate is not a method efficacy result.

Native SOPD-OCR showed a positive diagnostic trend: EMA OCR reward on16 upstream
test-prefix prompts was .6270/.7518/.8353/.8670 at updates0/8/16/24. Training
finished at32; no native eval at32. Noise was not paired. This is a small
single-task subset control, not a paper reproduction or a matched comparison
against the local five-query fit.

All10 matched-CFG teacher-copy predictions were exactly equal; all10 old
noop/target endpoint images replayed with identical hashes. Thus the frozen
inputs and adapter use were faithfully recovered.

**Next:** isolate changed prefix states from changed suffix/time steps using the
same joint128 checkpoint. Its apple/bowl relation survives a single updated
action but becomes apple-below-bowl under full generation. Do not interpret all
remaining loss as cross-sample conflict: joint fitting retains more repairs.

See [report](report.md), [results](results.json), [curve](learnability.png),
[next experiment](NEXT.md), and [execution instructions](../../docs/phase7_runbook.md).
Five positives were deliberately selected from old training observations. The
image review is by the implementing assistant, not independent human truth.

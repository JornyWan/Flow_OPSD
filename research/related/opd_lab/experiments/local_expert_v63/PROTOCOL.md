# Phase63: real SD3 semantic online interface acceptance

Frozen before submission. One allocation, one L40, maximum1800 GPU seconds;
sequential verified/random training then common evaluation, no retries/timers.
This is an engineering pilot, not a paper efficacy test or promotion of the old
CFG1/query1 configuration. Its goal is the versioned Phase62 interface on real
models: A evidence, online behavior refresh, exact supervision-count control,
full-student deployment, B task scoring and all five quality metrics.

Data chosen without outcomes: from the existing512train, lexicographically first
two ids in each of4families x product-photo/flat-vector =16train. Two passes,16
queries per epoch, one uniform-hash query index in0..8 per trajectory,32query
states per arm. First one id per4families x archive/colored-ink =8existingdev;
no confirmation/final or human rating. Seed63063 and paired hash-derived dev seeds.

Both arms: same published Student, fresh Adam/scaler/RNG,512px,Flow10,studentCFG4.5,
GenEvalCFG1,text128,lr3e-4,accumulation16,2epochs,one fitting pass. Native full-
amplitude detached transition loss/all16, current local EMA schedule retained;
evaluate final epoch2 RAW only. No best-checkpoint choice. This new online/full-
student platform differs from old query1 trials; differences cannot be attributed
solely to verification, nor treated as a fair comparison to the 128dev reports.

Verified uses existing semantic_accept A rule. Random uses the complete verified
arm's immutable per-batch family/template count schedule; same prompt/noise/time
identities. Each arm refreshes its own behavior, so states may diverge after the
first update. Same supervision and update count do not imply identical state
distributions. Family/template matching also matches photo/illustration domain.
Random selects fixed teacher actions without using its own A acceptance labels.

Accept/reject errors stop training, unknown abstains; no forced positive examples.
Save all128candidate endpoints and A evidence. If no accepted update exists,
report inconclusive_no_accepted_updates, do not switch prompts or force targets.
Frozen live source fingerprint checked before preparing random; both immutable
snapshots must match. Two model/scorer processes run sequentially in the allocation.

After training: physically remove behavior; only default adapter remains for all
10steps. Evaluate initial/verified/random on the same8dev,24images and120quality
values. Bjoint/OCR exact/CER, A unknown, PickScore/26,CLIP,Aesthetic,HPS,ImageReward
all reported. Re-score first saved training noop/teacher with the established
TaskScorers to require exact A evidence parity (2reads by local scorer,0agent reads).
Training counts/optimizer steps/exported raw equality and finite values are gates.
No significance test or reward-improvement gate on8dev. Latency is diagnostic.

Success means executable interface and measured cost, not stable superiority.
Do not submit a larger trial automatically. Next freeze proper common-data,
common-start native OPD/single-query/random/verified comparisons including all
128dev and total collection+verification+fit+evaluation cost. Large asset full
fingerprints remain necessary for formal results; this engineering pilot retains
the established inventory checks and exact initial/teacher adapter hashes.

Immediately probe minilab-gpu/sablab-gpu/scu-gpu/preempt_gpu; L40 only, no automatic
requeue. Include other projects and pending in sablab<=6 cap. No start estimate
within one hour => leave prepared. Any GPU failure stops without retry; completed
archives immutable, CPU report repair in an independent record.

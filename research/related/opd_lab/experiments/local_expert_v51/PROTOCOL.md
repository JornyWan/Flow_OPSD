# Phase51: temporal support of fixed cumulative corrections

Paper claim C2 development diagnosis. Published OPD Student is the common anchor,
not the quality-degraded Phase46 model. Fitted tensors are Phase48 verified low-LR
step16, unchanged. Three policies: published_anchor all10 anchor queries; low_full
all10 fitted queries; low_temporal fitted only for sigma in
[0.9367391467094421,0.9800646603107452), anchor elsewhere. This is index1 on the
frozen Flow10 grid. Boundaries from the reference training grid are never tuned
per prompt or recentered per deployment grid. One backbone/two student adapters,
one denoiser query per step, no teachers/rewards/search at generation time.

Use ALL128 frozen paper_joint_v2 development prompts, first prespecified noise
per prompt, four equally sized task families and both style domains. Exactly384
new images. Flow/Euler10 shift3, CFG4.5, resolution512, fp16 autocast with FP32
backbone, latents and VAE. Both weights remain unchanged. No optimization, new
acceptance masks, best-of-N, learning-rate selection, or outcome-based subset.
Confirmation512 and final_test768 remain ungenerated. This is development, not
an independent paper confirmation or common-start online-training result.

Primary: low_temporal minus published_anchor, terminal B joint OCR exact AND
strict structure. All128 paired cases, family-stratified prompt bootstrap20000,
seed51051. Gate: delta>0 AND descriptive95% lower>0. A pass justifies freezing a
common-start training comparison; a failure gives no temporal efficacy claim.
low_full-anchor and temporal-full are declared secondary contrasts. Report all
OCR/structure scores, A lower/upper/unknowns, clause repairs/harms, family/template/
domain subgroups, PickScore/CLIP/Aesthetic/HPSv2.1/ImageReward, storage and measured
latency. No multiplicity-adjusted success claim for secondary outcomes. Quality
harm must be reported even if the task gate passes; failure to detect harm is
not noninferiority. A/B share components and are not independent human truth.

Exact first-case GPU replay of all three routes; all384 image/model/cache hashes,
3840 Euler transitions and128 shared prefixes checked on CPU. Scoring uses the
unchanged Phase42 path and8 historical parity cases, no evaluator tuning. Errors
invalidate inference; preserve them and do not silently drop examples.

OneL40, maximum3600 wall seconds including loading/scoring/failures. Spend from
the prior paper session's5867-second unused balance; do not reset its10800-second
cumulative ceiling. A separate continuation ledger preserves that completed
archive. Zero agent image reads; no automatic retry, extension or next GPU run.

# Phase52: fresh-anchor temporal target-selection mechanism experiment

Claim C2. Compare fixed_all, random_matched and verified target acceptance under
the SAME fixed temporal deployment rule, common published OPD Student and fresh
Adam/scaler/RNG. Only acceptance varies across trained arms. This is one shared
frozen pool and one training seed, not an independent online method comparison,
native OPD reproduction, multiple refreshes, or paper confirmation.

Training:64 previously unused paper_joint_v2 TRAIN prompts, eight per family x
domain. Select by hash phase52-train-v1 BEFORE generation, use first frozen noise.
Collect original temporal behavior and one full-amplitude GenEval teacher action
at index1. Outside sigma [0.9367391467094421,0.9800646603107452), always use the
permanently frozen published anchor. Inside use frozen behavior during collection,
learner during deployment. In this first round behavior equals the anchor.
All candidate suffix actions are recomputed on their own reached states.

Selector: unchanged Phase45 ordered semantic_accept on A, known repair, no known
protected harm, abstain on unknown; no labels from development or humans. Random
matches accepted counts in EACH family x domain stratum using fixed hashes and
no individual outcomes. Fixed_all retains64 targets. All three arms start from
identical published tensors and NEW empty Adam history; four full-pool fitting
attempts, LR1e-4, Adam(0.9,0.999), eps1e-8, weight decay1e-4, clip1. Original
transition-mean loss, sum over accepted targets /64; no noop/retention gradients,
no EMA, no optimizer update if the mask is empty. These are equal attempts, NOT
equal compute. No Phase46/48 weights or optimizer history are continued.

Development: ALL128 already-exposed Phase51 development prompts, first frozen
noise, no new subset. Reuse published-anchor images and scores with original
hashes/fingerprints verified separately. Generate teacher_local plus all three
fitted temporal policies:512 new development images, plus128 collection images,
640 new images total. Flow10 shift3,CFG4.5,512,FP16 autocast and FP32 backbone/
latents/VAE, GenEval teacher CFG4.5. Same sigma window for all arms; no window,
checkpoint, LR, seed or amplitude search. Confirmation512/final768 untouched.

Three-level accounting: direct teacher intervention versus anchor; fitted action
versus direct teacher; complete deployed temporal policy versus anchor. Here
there is exactly ONE learner query, so fitted local action with anchor suffix
and full temporal deployment are IDENTICAL policies. Do not generate duplicate
images or present them as independent evidence. Deployment is one backbone with
two student adapters, ten denoiser queries, no teacher, verifier or search.

Primary: verified minus random_matched B joint OCR exact AND strict structure.
Family-stratified prompt bootstrap20000,seed52052, descriptive95% interval.
Selection development gate: delta>0 AND lower>0. Zero acceptance is a negative
coverage diagnosis; keep all arms and report skipped fitting, never resample.
Also report verified-fixed, each fitted-anchor, teacher-anchor and verified-
teacher, all OCR/structure, A lower/upper/unknowns, protected harm, all family/
domain/template groups, all five quality metrics and actual NFE/time/storage.
Every secondary interval is descriptive, no multiplicity-adjusted efficacy claim.

Decision: if selection gate fails, selection advantage is not established; do not
automatically enlarge or tune. If it passes but verified-anchor lower<=0, no net
gain is established. If both pass and ANY of five verified-anchor quality95%
upper<0, report task signal WITH quality harm; otherwise quality noninferiority
is still unproven. No quality tolerance or noninferiority claim is invented from
these data. A positive development result only motivates a separately frozen
multi-run / compute-matched training design; it is not a paper success gate.

Fresh-initial GPU action/teacher replay on all64 training queries; first-route
replays; independent exported-LoRA reload; disk-resume versus uninterrupted Adam
audit; all new Euler traces,512 development prefixes, all score/image/model/cache
identities and selection decisions replayed. Frozen Phase42 scorers, eight
historical parity cases per scoring phase. Errors invalidate inference; no drops.

One L40 job, hard3600-second limit, including loading/scoring/failure. Debit the
prior4177-second balance (cumulative ceiling10800, already6623 used). Separate
continuation ledger; completed archives immutable. No automatic retry, extension
or subsequent GPU job. Zero agent direct image reads, no remote vision API.

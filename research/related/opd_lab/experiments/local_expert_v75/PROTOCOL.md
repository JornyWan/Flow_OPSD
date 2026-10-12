# Phase75: same-state CFG precision ablation

Frozen 2026-10-09 in response to the user's request to continue without conflating
engineering mismatch and research effectiveness. This is a new three-arm study,
not a retry of Phase74. Its completed archive and thresholds remain untouched.

Question: on the SAME 27 cached real SD3 states and teacher targets, how much of
the official/local gradient and Adam-update discrepancy is removed solely by
combining CFG in FP32? Arms: official extracted transition/reduction with actual
Accelerate autocast/output conversion; historical local; local with explicit
cfg_combine_precision=float32. Official vs local also differs in loss arithmetic;
legacy vs fixed changes only CFG arithmetic precision.

Fresh SD3.5M rank32 alpha64 LoRA, same initial state per arm verified exact,
seeds74001 initialization,74010..12 collection; first3 frozen TRAIN prompts,
512px,text128,fp16,Flow10,j0..8 supervised,studentCFG4.5,GenEvalCFG1. Collection
ALWAYS legacy precision, so the candidate cannot change queried states. Each arm
resets RNG, initial weights, AdamW and GradScaler; LR3e-4, betas(.9,.999),eps1e-8,
wd1e-4,clip1,loss divided by27, exactly1 successful update. No EMA or selector.
Microbatch1 and27states are a diagnostic, not native81states/card training.

Retain original engineering limits: target maxabs2e-6, gradient global relative
L2 .001, weights maxabs3e-6, loss maxabs2e-6; report each gate. Successful execution
returns completed even if parity is false; technical errors return nonzero. A
numerical gate failure cannot be called method ineffective; success cannot be
called superior image generation. Do not relax thresholds after results.

Save shared initial/queries, full unscaled gradients and updated weights/losses
per arm. Report reference-denominated global gradient error, parameter error
relative to initial-to-updated displacement, absolute-error quantiles and sign/
near-zero counts. Near-zero cutoff1e-8 is descriptive, not an efficacy threshold.

Deploy initial and each updated weight with SAME FP32 CFG, first prompt,new fixed
noise seed74100,10steps; compare final latents. Export/reload fixed RAW, physically
remove all other adapters, require exact latent replay. One deployment prompt
is an engineering diagnostic, not a quality/generalization evaluation. No VAE
image decoding, no task/OCR/five-quality inference; those remain required in the
subsequent matched efficacy study. Confirmation512/final768 untouched.

New independent budget:1 L40S,1800 requested seconds, one job, no requeue/retry,
no timer. Previous failed/unused reservations are not reused. Probe all four
partitions immediately before submit, sablab allocated+pending<=6 across all
projects. Ordinary partitions preferred for similar estimates; preemption means
stop with no retry. Expected-start window<=24hours. Do not move other jobs.

Framework/upstream snapshots and data/adapter bytes frozen by SHA. Large backbone
inventory uses bytes/mtime rather than full-content hashes, a disclosed limitation.

Decision: if fixed improves parity, use the explicitly documented configuration
in prospective local matched arms. If residual mismatch remains, quantify its
scale and functional drift, retain native baseline for attribution, and decide
whether another narrowly scoped fix is necessary; do not spend indefinite runs
chasing bitwise equality or reject trajectory learning without efficacy data.
Next scientific comparison needs sufficient supervised coverage and matched
actual LR/teacher/data/state-time weighting/compute, full deployed student joint,
OCR/structure/CER/unknown and all five qualities. No such GPU training is submitted
by this diagnostic launcher. Short-trajectory target learning remains a hypothesis.

# Phase50: full DrawBench coverage with paired generation

Supporting paper evidence on general quality and two-noise feature dispersion.
Freeze five checkpoints regardless of Phase48 outcome: unadapted SD3.5-M,
published OPD Student, Phase46 verified step8, Phase47 verified step16 at3e-4,
Phase48 verified step16 at1e-4. The latter three are mechanism checkpoints, not
native OPD/OPSD algorithm baselines or independently trained replicas.

Use all200 unique prompts in the pinned upstream DrawBench test.txt, which has
five identical copies of every prompt. Preserve source hash and original row
indices, deduplicate only for defining prompt clusters, then allocate exactly
two new fixed noise seeds per prompt.200×2×5=2000 new images. Same prompt/noise,
Flow/Euler10 shift3 CFG4.5,512,fp16,FP32 backbone/latents/VAE. This unified
protocol is distinct from upstream default Flow40 and five repeated rows.
Public benchmark coverage is not a claim these prompts have never appeared in
any earlier upstream training or evaluation. No changes based on scores.

All PickScore/CLIPScore/Aesthetic/HPSv2.1/ImageReward values reported. Main quality
family: all five low_verified minus before_verified differences; prompt-cluster
bootstrap20000 seed50050 preserves both noise samples. Report95% descriptive
intervals and marginal99% intervals for that five-metric Bonferroni family.
Other contrasts descriptive. No aggregate quality winner, no task-success or
noninferiority claim. Token truncation audited; no literal target-word test is
applicable to DrawBench. Same-image feature extraction checked against standard
CLIP scoring; per-prompt two-noise cosine distance and exact duplicate rates are
supporting diversity proxies, not standalone image quality or collapse verdicts.

Bare backbone represented exactly by zero LoRA B matrices with bias=none, while
all backbone weights remain frozen. Only student/behavior adapters loaded, no
experts/verifier during generation. Image/feature/cache/model hashes checked.
7200-second singleL40 cap includes loading, all generation/scoring/failures.
No automatic extension/retry or partial-sample success claim. Zero agent reads.

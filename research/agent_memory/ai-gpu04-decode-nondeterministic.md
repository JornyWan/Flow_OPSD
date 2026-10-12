---
name: ai-gpu04-decode-nondeterministic
description: On ai-gpu04 (A40) the SD3 VAE decode of identical latents is intermittently not bit-identical across two calls; transformer sampling is. Exclude it for bit-exact probes
metadata:
  type: reference
---

2026-10-04, P11 (§4.850): selftest `decode_eq_pipeline_image` failed on **3 of the 5 shards that landed on ai-gpu04**
(1068379_4/10/11 failed, _5/_6 passed — intermittent, maybe per-GPU) and passed everywhere else; the other three checks (segment resample == pipeline, tail from i*,
δ=0 work == pre) passed on ai-gpu04 too ⇒ only the VAE decode path is non-reproducible there.
Paired-contrast probes that need bit-exact cells: `--exclude=ai-gpu01,ai-gpu03,ai-gpu04` (01/03 already excluded for
leaked GPU memory). Standard cross_eval runs are unaffected in practice (noise ≪ MDE), but don't use ai-gpu04 for
selftest-gated work. Related: [[autocast-cache-stale-weights]].

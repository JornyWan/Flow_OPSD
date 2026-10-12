# 0003 — Smoke retry: initialize `snr_weight` before validation

Date: 2026-08-28  
Status: completed; validated in 0004

## Why 0002 is invalid

SLURM job `1035889` passed the source-hash, patch, CPU equation, configuration,
hardware, model-loading, and three-teacher-routing checks, then failed before the
first sample with:

```text
UnboundLocalError: local variable 'snr_weight' referenced before assignment
```

This is an implementation-order bug in the new fail-closed validation, not an
experimental result. No optimizer update or checkpoint occurred. Job `1035889` is
therefore invalid and cannot support any claim about the method.

## Single correction

`code/0003_fix_snr_init_order.patch` moves only the `cr and snr_weight` validation
from the early constrained-retention flag block to immediately after the existing
`snr_weight` assignment. The objective, masks, tolerances, data, initialization,
hardware request, and acceptance criteria remain unchanged.

The retry must apply `0002_constrained_retention.patch` and then
`0003_fix_snr_init_order.patch` to the same audited source hashes. It receives a new
job ID and output directory; 0002 artifacts are preserved.

## Submission

- correction patch SHA256: `bcdfbc306d29ec56f743cae932c964d72b9c9ad95ac914f6e4f3c75adbae4992`
- corrected trainer SHA256: `bedcbc8fbf4a6fa21705876b93db01aad0b551696630cda775505f88b53956a7`
- exact batch script: `slurm/0003_cr_smoke_retry.slurm`
- command: `sbatch codex_research/slurm/0003_cr_smoke_retry.slurm`
- job ID: `1035891`
- output: `runs/0003_cr_smoke_retry/slurm_1035891.out`

The retry completed with exit `0:0`, three optimizer steps, and three checkpoints.
Full metrics, invariant checks, limitations, and the next experimental gate are in
0004.

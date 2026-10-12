# Phase32: deterministic retention experiment

Work in `opd_lab`, using `OMP_NUM_THREADS=4 PYTHONPATH=python:scripts` and `../../conda_envs/envs/diffusionopd/bin/python`. Previous completed campaigns and the three-hour-session archive are immutable.

The authoritative experiment protocol is `experiments/local_expert_v32/PROTOCOL_v2.md`. The original protocol remains alongside it as history. Both coefficient0 controls and coefficient1 retention arms are freshly trained from S1, with identical masks, targets, schedules and optimizer settings. The historical all-target model is only a contextual reference. Never relabel its score as a freshly trained deterministic control.

## Reproducibility correction

Attempt001 omitted the historical `CUBLAS_WORKSPACE_CONFIG=:4096:8` binding and failed the eight-update checkpoint comparison. Attempt002 added diagnostics and was cancelled when the missing binding was identified. Attempt003 restored the binding, but still failed: identical first losses had different gradients and later weights. No proposed retention model or development images were produced by these attempts.

`runs/le32_repro_001` isolates this issue using the unchanged original `mask_step`: reset weights, optimizer, scaler and RNG, then repeat the same first update twice with default algorithms and twice with `torch.use_deterministic_algorithms(True)`. Default repeats differ; deterministic repeats match student/optimizer/scaler exactly. This establishes update nondeterminism in this runtime, without identifying a particular CUDA operator or establishing its effect on old image scores.

Attempt004 enables deterministic algorithms and explicitly requires the cuBLAS environment binding before CUDA initialization. It checks one original update against the independent probe checkpoint, then lambda0 and lambda1 first updates against that exact state. All164 used anchor targets must also replay exactly at initialization. This keeps an exact current implementation check without pretending to reproduce an old nondeterministic checkpoint.

## Current run and review

- Worker: `python -m opdlab.retention_learning --run <absolute runs/le32_retention_004>`, launched with `env CUBLAS_WORKSPACE_CONFIG=:4096:8`. The Slurm command, source snapshot and job ID are recorded in `submission.json` and the campaign ledger. Do not rerun into an existing directory.
- Four arms each train8 updates. No-retention controls use160 backwards each; retention arms use320 each. The successful attempt has960 training backwards plus80 audit backwards. Other attempts and the80-backward reproducibility probe also count toward total costs.
- `scripts/monitor_retention_learning.py --attempt 004` is read-only and records all campaign jobs in final accounting, including failures/cancellations.
- After worker completion, `scripts/review_retention_learning.py prepare` checks hashes, initial noise, all1440 Euler transitions, counts and replay gates. It creates216 anonymous image views:72 inherited S1/historical-all views,144 new views. Read `pending.json`, not `key.json`, while observing new views.
- Use the unchanged rubric, at native512 resolution, including unknowns. Write exactly one complete record per case to `observations_draft.jsonl`, preserving inherited observations verbatim. Keep any pre-freeze revisions explicitly; do not revise after key join.
- Run review stages `freeze` then `analyze`. They produce six score tables and four current primary contrasts, with18 prompt clusters and unknown-adverse sensitivity. No automatic continuation gate is based on MSE.
- `scripts/report_retention_learning.py` embeds216 original PNGs. `scripts/plot_retention_learning.py` exports the four current arms as PNG/SVG/CSV. Run the browser checker with the bundled Playwright Node and `FONTCONFIG_FILE=.tools/report-browser/fonts/fonts.conf`, then `scripts/verify_retention_learning.py` for independent numeric reconstruction.
- Write outcome-specific `STATUS.md`, `NEXT.md` and `interpretation.json`, preserving negative/unknown outcomes. `scripts/archive_retention_learning.py` validates all five job states, previous archives and report checks, creates a portable ZIP, and records final hashes. Do not edit after final archival.

Current visual observations remain provisional implementing-assistant labels on an exposed cohort. Fresh-prompt confirmation, independent human assessment, native OPD equivalence, efficiency, and sampler robustness are separate unanswered questions.

# Phase25 reproducibility

Run from `opd_related/opd_lab`; Python is `../../conda_envs/envs/diffusionopd/bin/python`, with `PYTHONPATH=python:scripts`. All72 Phase24 cases are now observed mechanism-development data. The only variable is OCR duration1/2/7 after the same3G prefix, with recomputed G suffix for short interventions. No training or deployed verifier.

Before allocation20 relevant CPU tests passed in2.78s:

```bash
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python -m pytest -q tests/test_short_correction.py tests/test_handoff_confirmation.py tests/test_capability_probe.py tests/test_interval_probe.py tests/test_headroom_probe.py
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_short_correction.py
```

Prepared run `runs/le25_short_correction_001`; framework/upstream snapshots, input hashes, inherited controls and all source trajectories are frozen. Campaign PROTOCOL.md and REVIEW_RUBRIC.md match run copies. Exact job1059181 command is recorded in experiments/local_expert_v25/ledger.jsonl; one L40,30min, no automatic retry. The worker is `env CUBLAS_WORKSPACE_CONFIG=:4096:8 <python> -m opdlab.short_correction --run <absolute-run-path>`.

Expected total805G+223O calls,1251CFG-expanded examples,146decodes,72positive/negative text encodings,0behavior/student/backward calls. These include two complete historical replays before any short candidate; saved prefix computation is inherited and excluded from new queries.

Post-run, in this order:

```bash
../../conda_envs/envs/diffusionopd/bin/python scripts/refresh_campaign.py --campaign experiments/local_expert_v25
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_short_correction.py prepare --run runs/le25_short_correction_001 --review runs/le25_short_correction_review_001
# Inspect all144 new anonymous native images; write observations_draft.jsonl; do not inspect key.json or aggregate results.
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_short_correction.py freeze --run runs/le25_short_correction_001 --review runs/le25_short_correction_review_001
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_short_correction.py summarize --run runs/le25_short_correction_001 --review runs/le25_short_correction_review_001
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/report_short_correction.py
FONTCONFIG_FILE="$PWD/.tools/report-browser/fonts/fonts.conf" .tools/report-browser/node_modules/node/bin/node runs/short_correction_report_validation_001/check.cjs
```

Old fullG/long-handoff observations are copied through the frozen plan by pair/image hash, not rerated. Unknowns are preserved. All results and confidence intervals are descriptive, with36prompt clusters and2seeds per prompt; no new confirmation gate. All288 comparison PNGs are embedded in the standalone report. Final audit and completion manifest are produced after the full review; actual results/costs are in STATUS.md,results.json,slurm_status.json.

## Completed run

Job1059181 completed with exit0:0 in209s on one L40 (0.05805556 GPU-hour). Actual counters match the pre-run budget. There was one premature read-only review preparation attempt before result.json was visible; it stopped before creating review artifacts. The successful preparation ran after worker finalization. No GPU failure, rerun or automatic retry occurred.

All144 new images were observed before joining method keys. Four native-image rechecks are recorded in review_revisions.json; only case37-B visibility changed from unknown to true before freezing. The initial draft and revisions remain available. Old control observations were inherited unchanged. Final complete-scene counts: fullG38, ocr1=40, ocr2=43, handoff41 out of72. These are provisional implementing-assistant labels on observed development cases, not independent human validation.

Read-only final audit (omit --write after the initial completion record exists):

```bash
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/verify_short_correction.py
```

The initial audit used --write once. It checks immutable Phase22/23/24 files, all worker inputs/outputs, copied observations/image hashes, all point metrics and10 independently reconstructed descriptive/unknown-adverse bootstrap intervals, report pair mapping and all288 embedded PNG hashes. Browser validation passed offline decoding, pair/four-view switching, family/seed filtering, observation display, zoom/Escape, and mobile overflow checks with no JS errors or network requests. Saved desktop/mobile screenshots were visually inspected. Worker verification separately includes all1440 saved Euler transitions and both exact historical replays. Twenty relevant CPU tests passed before allocation; no worker changes were made after that run.

STATUS.md records findings and limitations; NEXT.md specifies the proposed bounded verifier audit. Phase25 completion_manifest.json freezes the completed campaign, code, run/review manifests and browser checks. No next-phase job has been submitted.

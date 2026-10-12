# Phase24 reproducibility

From `opd_related/opd_lab`, use Python `../../conda_envs/envs/diffusionopd/bin/python` and `PYTHONPATH=python:scripts`. Frozen protocol and full data provenance are in `experiments/local_expert_v24`. The immutable worker run is `runs/le24_confirmation_001`; hidden review will be `runs/le24_confirmation_review_001`.

Before allocation,16 tests passed:

```bash
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python -m pytest -q tests/test_handoff_confirmation.py tests/test_capability_probe.py tests/test_interval_probe.py tests/test_headroom_probe.py
```

Data and run preparation were separate steps, both already completed:

```bash
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_handoff_confirmation.py dataset
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_handoff_confirmation.py run
```

The campaign ledger records exact Slurm arguments for job1059045, one L40,30-minute limit, no automatic retry. Worker: `env CUBLAS_WORKSPACE_CONFIG=:4096:8 <python> -m opdlab.handoff_confirmation --run <absolute-run-path>`. The framework and upstream code are snapshotted by runtime.prepare; `training=True` loads teacher adapters but no optimizer/gradient step is executed.

Expected:730behavior calls,936G,1224OCR,4844CFG-expanded examples,289decodes,74cached positive/negative encodings,0student/gradient calls. Includes one historical baseline replay before new images/teacher queries. Never overwrite a completed attempt.

Post-run commands, in order (review/analysis refuses overwriting frozen data):

```bash
../../conda_envs/envs/diffusionopd/bin/python scripts/refresh_campaign.py --campaign experiments/local_expert_v24
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_handoff_confirmation.py prepare --run runs/le24_confirmation_001 --review runs/le24_confirmation_review_001
# Inspect all288 anonymous native images; write observations_draft.jsonl without reading key.json.
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_handoff_confirmation.py freeze --run runs/le24_confirmation_001 --review runs/le24_confirmation_review_001
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_handoff_confirmation.py summarize --run runs/le24_confirmation_001 --review runs/le24_confirmation_review_001
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/report_handoff_confirmation.py
FONTCONFIG_FILE="$PWD/.tools/report-browser/fonts/fonts.conf" .tools/report-browser/node_modules/node/bin/node runs/handoff_confirmation_report_validation_001/check.cjs
```

Additional full-tensor verification: use review_handoff_confirmation.py verify with the same run. It checks all2880 Euler transitions,72 handoff prefixes,72 CPU noise reconstructions and36 repeated contexts, plus run source/asset/output attestations. All image hashes and review-copy hashes are checked before freezing.

Summary statistics average two seeds per prompt and resample prompts within task families; the72pairs are not72independentprompts. Primary positive-gain tests are fixed in PROTOCOL.md, with unknown-adverse sensitivity reported separately. Assistant labels remain provisional. Final results/report/completion manifest are created only after complete review and verification.

Completed on2026-09-20: job1059045 COMPLETED0:0,467seconds,0.129722GPU-hour. All288 images were observed with method IDs hidden; observations_initial_draft.jsonl and review_revisions.json preserve native-image rechecks before freezing. Case24-B text uncertainty became true after resolving outlined glyphs; case14-D literal transcription was refined without changing labels. All72 rows froze before key join. No rubric or protocol changes, dropped cases, retraining or automatic retries.

Frozen counts: clean9/72,fullG38/72,fullO24/72,handoff41/72. The prespecified two-comparison gate failed: handoff minusG+4.2pp,95% interval[-11.1,+18.1]; minusO+23.6pp,[+9.7,+37.5]. Full interpretation and a separate descriptive oracle analysis are in STATUS.md and interpretation.json. No new student was trained.

report.html embeds all288 PNGs (~68.27MB), with baseline/handoff default, strong-G comparison shortcut, all-four view, task/seed filters, frozen observation notes, zoom and offline operation. Browser verification and desktop/mobile screenshots are retained. Completion verification checks oldPhase22/23 manifests, review/output provenance, independently reconstructed point counts and report image mapping. The new completion manifest records the final artifacts and source. No next-phase job is active; NEXT.md is a finite proposed mechanism experiment, not a completed or preregistered run.

Read-only final audit (also independently reconstructs the two primary bootstrap comparisons and unknown-adverse intervals, and hashes every embedded PNG):

```bash
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/verify_handoff_confirmation.py --check-only
```

Without --check-only this command initially wrote completion_verification.json; it refuses to overwrite that frozen artifact. The Phase22 and Phase23 completion manifests retain41 and42 unchanged files respectively. Slurm accounting was rechecked as COMPLETED0:0; the completed job had already left squeue, whose job-specific query returned Invalid job id specified. No restart or other user's job was touched.

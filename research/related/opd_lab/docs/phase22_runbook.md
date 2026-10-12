# Phase22 reproduction and artifacts

Campaign: `experiments/local_expert_v22/`; worker: `python/opdlab/interval_probe.py`. Previous Phase21 sources/artifacts were not edited.

Use the existing lab environment, from opd_lab:

```bash
export PYTHONPATH=python:scripts
../../conda_envs/envs/diffusionopd/bin/python -m pytest -q tests/test_interval_probe.py tests/test_headroom_probe.py
```

Fresh preparation (dataset already exists; do not overwrite it): `scripts/prepare_interval_probe.py dataset`, then `development --attempt <fresh-number>`. Confirmation preparation requires frozen development results and confirmation_release.json. Each run snapshots framework/upstream/config/environment/assets; worker executes with the run's own framework PYTHONPATH through `submit_slurm.py`. Exact executed commands are in ledger.jsonl and run/submission.json, including the determinism environment variable. Do not edit source snapshots to rerun a completed attempt.

GPU worker: `python -m opdlab.interval_probe --run <absolute-run>`. It rejects reused outputs, writes failure.json on failure, records initial adapter and all baseline/branch tensors, executes a full-state/image noop replay gate, and asserts exact forward/decode counters and frozen adapters.

Postprocess each completed run in order:

```bash
python scripts/review_interval_probe.py prepare --run <run> --review <fresh-review>
# Inspect all12 native512 contact sheets and write observations_draft.jsonl.
python scripts/review_interval_probe.py freeze --run <run> --review <review>
python scripts/review_interval_probe.py summarize --run <run> --review <review>
```

Original runs: `runs/le22_development_001`, `runs/le22_confirmation_001`; reviews: `runs/le22_development_review_001`, `runs/le22_confirmation_review_001`. Each frozen cohort has60 observations. Key joins happen only in summarize, after freezing. Never revise these labels in place; independent rechecks require a separate review source.

After both cohorts and Slurm accounting finish:

```bash
python scripts/refresh_campaign.py --campaign experiments/local_expert_v22
python scripts/analyze_interval_failures.py
python scripts/report_interval_probe.py
```

These write-once analysis/report builders reject existing outputs. `failure_audit.json` is explicitly post-hoc unknown sensitivity, not a redefinition of the primary gate. The final HTML embeds original PNGs; report_manifest.json records all image provenance and input hashes. `--development-only` can build a separately named interim page while confirmation is pending; do not treat its scheduler snapshot as live status. The unpublished interim page in this session was removed when confirmation started, before final publication.

Browser checks: `runs/interval_report_validation_001/check.cjs` uses cached Playwright/Chromium under `.tools/report-browser`; use its absolute fonts/fonts.conf as FONTCONFIG_FILE. Checks offline loading,24 cases, all120 original512 PNGs, pair/all5/oracle switching, modal Escape, mobile overflow, and no external requests/JS errors. This is presentation validation, not model validation.

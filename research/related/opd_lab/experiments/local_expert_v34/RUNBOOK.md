# Phase34 execution and review

All commands run from opd_related/opd_lab with the existing diffusionopd environment. Primary generation/analysis files are bound by analysis_attestation.json; supporting report/verification files by supporting_attestation.json. Do not rerun into existing output directories.

## Current attempt

- Run: runs/le34_amplitude_001; Slurm job 1061762, one L40, 45-minute cap.
- Monitor: runs/le34_monitor_001/latest.json; slurm_status.json is finalized when accounting reports termination. NFS negative lookup caching may temporarily lag newly created files; worker log progress and directory listing can cross-check this.
- Student CFG1 and OCR teacher CFG4.5; no optimizer or parameter update.

## After successful generation

```bash
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_amplitude_probe.py prepare
```

This validates immutable inputs, all600 images and6000 saved Euler transitions, then creates runs/le34_review_001. View only its items.json and native image-CASE-LETTER.png files. Do not inspect key.json or run automatic method/utility joins before observations_frozen.json exists. Five images per120 cases, no inherited labels. The original two-action-mask membership is also hidden during review.

Append observations through the record subcommand using a JSON object keyed by case and A–E. Each image value is [text_boolean_or_null, task_boolean_or_null, visibility_boolean_or_null, literal_visible_text, notes]. Never turn uncertainty into a forced yes/no. Do not revise labels after method join.

```bash
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_amplitude_probe.py freeze
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_amplitude_probe.py analyze
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/report_amplitude_probe.py
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/verify_amplitude_results.py
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/plot_amplitude_probe.py
FONTCONFIG_FILE="$PWD/.tools/report-browser/fonts/fonts.conf" .tools/report-browser/node_modules/node/bin/node scripts/check_amplitude_report.cjs
```

Inspect desktop/mobile screenshots and the static figure before documenting visual_inspection.json. Write STATUS.md, NEXT.md, interpretation.json and final ledger event after interpreting every planned stratum and unknown sensitivity, not just successful cases. Then use scripts/archive_amplitude_probe.py and recheck its completion_manifest.json and ZIP CRC/hashes. Previous Phase33 archive must remain unchanged. There is no automatic next training job.

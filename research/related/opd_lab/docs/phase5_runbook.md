# Measurement audit runbook

Use the pinned diffusionopd interpreter from `opd_lab`, with `PYTHONPATH=python`.
Freeze `experiments/local_expert_v5/PROTOCOL.md` before prepare. Completed outputs
are immutable; use a fresh attempt and output names for re-execution.

```bash
OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python
export PYTHONPATH=python
"$OPDLAB_PY" scripts/prepare_measurement_audit.py --attempt 002
```

Submit the run through `scripts/submit_slurm.py`, stage `le5_measurement`, one L40,
20-minute limit, phase5 ledger. Command: absolute Python executable followed by
`-m opdlab.measurement_audit --run <absolute run>`. The run's frozen framework is used.

```bash
"$OPDLAB_PY" scripts/run_measurement_ocr.py --run runs/le5_measurement_002 \
  --cohort matrix --out runs/le5_ocr_matrix_002
# After GPU completion:
"$OPDLAB_PY" scripts/prepare_calibration_review.py --run runs/le5_measurement_002 \
  --out runs/le5_calibration_review_002
"$OPDLAB_PY" scripts/run_measurement_ocr.py --run runs/le5_measurement_002 \
  --cohort calibration --out runs/le5_ocr_calibration_002
```

View every calibration sheet before reading its key/scores. Populate the template,
retain uncertainties, and freeze observations SHA/UTC time/rubric hash in
`observations_frozen.json` with `scores_revealed=false`. Attempt001 supplies the
schema, not reusable labels. Do not revise observations after revealing scores.

```bash
"$OPDLAB_PY" scripts/summarize_measurement_audit.py --run runs/le5_measurement_002 \
  --matrix runs/le5_ocr_matrix_002 --calibration runs/le5_ocr_calibration_002 \
  --review runs/le5_calibration_review_002 --out runs/le5_report_002
"$OPDLAB_PY" scripts/refresh_campaign.py --campaign experiments/local_expert_v5
```

`scripts/explain_measurement_audit.py` separately renders the explicitly post-hoc
illustrative examples and utility decomposition; do not label these as additional
predeclared tests. Every CPU report records its source hash. The detector auditor
inspects the pinned upstream scorer closure to access raw predictions; it does
not patch upstream. It refuses unexpected filtering constants. None of the new
measurement rules is imported by training or used to overwrite v1 labels.

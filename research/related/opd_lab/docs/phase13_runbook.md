# Phase13 evidence-verifier development

Run from opd_lab using the existing OPD environment. All192 source images and
phase12 observations are immutable development inputs. No diffusion model is
loaded by this worker; it only replays OCR/GenEval on existing images.

New modules evidence_verifier.py and evidence_audit.py are separate from the
original verifiers.py/method.py and the older phase5 measurement_audit.py.
The initial CPU suite caught an accidental filename collision with the phase5
worker. The old file was restored byte-for-byte from8e9c655 before any run was
prepared; new worker uses evidence_audit. The failed collection log is retained
in tests.txt. After separation77 tests passed; a JSON-roundtrip selection test
was then added and the full78-test suite passed in tests_final.txt.

```bash
export PYTHONPATH=python
../../conda_envs/envs/diffusionopd/bin/python -m pytest -q
../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_measurement.py
../../conda_envs/envs/diffusionopd/bin/python scripts/submit_slurm.py \
  --run runs/le13_measure_001 --stage le13_measure --gres gpu:l40:1 \
  --time 00:30:00 --ledger experiments/local_expert_v13/ledger.jsonl -- \
  ../../conda_envs/envs/diffusionopd/bin/python -m opdlab.evidence_audit \
  --run runs/le13_measure_001
../../conda_envs/envs/diffusionopd/bin/python scripts/summarize_evidence.py \
  --run runs/le13_measure_001
../../conda_envs/envs/diffusionopd/bin/python scripts/refresh_campaign.py \
  --campaign experiments/local_expert_v13
../../conda_envs/envs/diffusionopd/bin/python scripts/plot_evidence_audit.py
../../conda_envs/envs/diffusionopd/bin/python scripts/explain_evidence_audit.py
../../conda_envs/envs/diffusionopd/bin/python scripts/verify_evidence_audit.py
```

The candidate pool, v1 choices, rules and old-label hashes are frozen in the
audit plan/attestation. Upstream evaluate_image details are obtained from the
pinned function closure; no source patch or new thresholds are introduced.
Original-view outputs are compared to all prior OCR/GenEval values. Raw and
padded boxes, all OCR rows and rereads, reasons, counters, decisions and nulls
are retained. Same-model repeated reads are correlated evidence.

v1_discrete_event is a mechanism control: original binary labels, at least one
false->true event, protection of original true clauses; visibility unmeasured.
It is not a renamed original baseline. v2 additionally requires known evidence
for every baseline/candidate label and protection of its bbox visibility support.

Historical README revisions for verification: phase6 e384a09,7 8c1e160,
8/9 d9c392c,10 9dcc18b,11 3923b6a,12 8e9c655. After archive, phase13's own README
hash refers to its archive revision. Local run artifacts are indexed, not copied
into git. Do not overwrite prepared attempts or the current/old review labels.

## Completed results and posthoc separation

Job1049204 completed on ai-gpu13,226 allocated seconds /0.062777778 GPU-hours,
with no failed GPU attempt.192 evidence images,384 OCR full-image views,376
recognition crops and480 GenEval clause views; all raw original-view outputs
match phase12. No student generation, teacher action query or optimizer update.

Frozen v2 accepted6:3 safe,0 harmed,2 non-repairs,1 unresolved. Development
readiness failed. The discrete-event control accepted12:5 safe,1 harmed,4
non-repairs,2 unresolved. All64 discrete choices match retaining the original
utility ranking after adding only the discrete repair filter; the equivalence
is observed on this cohort, not asserted as a general algorithm identity.

After inspecting v2 results, evidence_dominance_posthoc1 was implemented separately:
unknown baseline can be preserved by known-true candidate evidence, but cannot
earn repair credit. This recovers only case34 (4 safe /7 accepted); it still
fails the precision requirement. Two new soundness tests passed, including all
possible completions of nullable baseline labels. No new GPU calls or v2 output
edits; posthoc rules/decisions live in separate source and posthoc_diagnosis.json.

summarize_evidence uses the original case order for ratio resampling, so all
v1 calibration statistics, including finite2000-bootstrap intervals, exactly
reproduce phase12. Scalar totals were unchanged. Its result hash is recorded
by both the figure and posthoc diagnosis. No confirmation cohort was launched.

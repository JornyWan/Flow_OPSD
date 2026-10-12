# Phase26 reproducibility

Run from `opd_related/opd_lab`. Python is `../../conda_envs/envs/diffusionopd/bin/python`; set `PYTHONPATH=python:scripts`. Campaign `experiments/local_expert_v26`; scorer run `runs/le26_verifier_001`. All images and reference labels are inherited; no diffusion model is instantiated or trained.

## Preparation and execution

Before allocation, 15 related CPU tests passed in 2.42s (`tests.txt`):

```bash
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python -m pytest -q tests/test_short_verifier_audit.py tests/test_evidence_verifier.py tests/test_headroom_probe.py
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_short_verifier.py
```

Preparation validates the Phase25 completion manifest, all 144 input images and paired noise hashes, copies the unchanged 36 records, snapshots the framework/upstream/environment, and fully hashes 11 scorer asset files in addition to existing inventory checks. The worker plan contains no reference observations. The reference results file is attested by hash but never read by the scorer.

The exact submission command and script are recorded in `ledger.jsonl` and `runs/le26_verifier_001/submission.json`. Job1059556 originally requested one L40 in minilab-gpu for 30 minutes. Its initial start estimate was the following day. A read-only `sbatch --test-only` check predicted an available L40 in preempt_gpu immediately; the same pending job was moved using:

```bash
scontrol update JobId=1059556 Partition=preempt_gpu QOS=low Requeue=0
```

This is documented in `scheduling_update.json` and the ledger. No duplicate job, changed GPU class, amended scorer or extended budget; automatic requeue is disabled. Old unrelated jobs were untouched. A read-only attempt to tail the log before allocation reported that it did not yet exist; this was not a worker failure.

Worker: snapshotted `python -m opdlab.short_verifier_audit --run <absolute-run-path>`. Environment paths and `CUBLAS_WORKSPACE_CONFIG=:4096:8` are recorded in job.sh. OCR runs on CPU; original GenEval detector/CLIP on the allocated GPU. All endpoint predictions are frozen by evidence.jsonl SHA in result.json before joining reference labels. No label-based adjustment of thresholds is allowed in this attempt.

## Postprocessing

The following commands require a completed worker result and no RUNNING/failure marker:

```bash
../../conda_envs/envs/diffusionopd/bin/python scripts/refresh_campaign.py --campaign experiments/local_expert_v26
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/summarize_short_verifier.py
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/report_short_verifier.py
FONTCONFIG_FILE="$PWD/.tools/report-browser/fonts/fonts.conf" .tools/report-browser/node_modules/node/bin/node runs/short_verifier_report_validation_001/check.cjs
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/verify_short_verifier.py --write
```

The summary and report refuse to overwrite existing outputs. The final audit uses --write once; omit it for a read-only repeat. All policy counts and six prompt-cluster bootstrap intervals are independently reconstructed. Both labels of every image, raw evidence, all 72 selections and abstention reasons remain inspectable in a standalone HTML with 144 embedded original PNGs. Scope remains observed development data with provisional assistant reference labels, not independent human validation or a trained-method result.

One additional summary test passed in 1.93s after submission (`analysis_tests.txt`), checking that known neutral accepts stay in the precision denominator and that an accepted case can be both harmful and unknown. No worker change followed its snapshot. Actual completion, scorer counters, scheduler cost and next decision will be recorded in STATUS.md and the completion manifest.

## Completed result and audit incidents

Job1059556 completed on ai-gpu20, exit0:0, 275 allocated seconds (0.07638889 GPU-hour); scorer body 264.36s. Counters: 144 evidence images, 288 full-image OCR detections, 322 row rereads, 288 GenEval clause views. No generation, diffusion forward or optimizer update. All 72 decisions are frozen. The policy accepts eight, all eight are safe according to inherited provisional labels; recall8/13, known harmful/unknown accepts0. Complete-scene counts G38, fixedOCR2=43, verified46. The predeclared readiness gate passes; this is development only. The old-text diagnostic makes exactly the same72 choices; the new whitespace handling is not the source of this gain. No next-phase job was submitted.

The first summary attempt followed scheduler completion but result.json was not yet visible on the login node; it stopped before any reference join or output. After the file became visible, exact replay validation caught case45's harms list order changing when JSON sorted the evidence dictionary keys. Both lists contain identical fields and all72 decisions/booleans match; comparison now sorts only the repairs/harms lists. The original summarizer is archived in runs/le26_analysis_audit_001. A regression test checks serialization invariance while still rejecting a changed acceptance flag; both summary tests passed in1.90s (`analysis_tests_after_order_fix.txt`). The original worker, predictions, protocol and thresholds remain unchanged.

The initial standalone report failed the browser check because textContent, a property, was mistakenly called. The HTML-only fix and initial report/template/manifest are archived under runs/short_verifier_report_validation_001/initial_failed_render. The premature final audit consequently stopped at the missing browser pass record without writing completion_verification.json. The corrected report passes all144 native PNG decodes/aspect ratios, full-cohort/family/decision filters, zoom/Escape, evidence visibility, fixed summary under filtering, mobile overflow, zero network requests and zero JS errors. Screenshots were visually inspected. Analysis code changes from its pre-completion attestation are separately recorded in analysis_changes.json; no science rule changed.

Final audit independently reconstructs all four policy count/ratio tables and six descriptive bootstrap intervals, checks all source images/reference observations/raw predictions and scorer weight hashes, and verifies the unchanged Phase22–25 completion manifests. The completed campaign, executable sources, worker artifacts, review/report evidence and audit records are frozen in completion_manifest.json.

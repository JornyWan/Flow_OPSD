# Phase27 reproducibility

From `opd_related/opd_lab`, use `../../conda_envs/envs/diffusionopd/bin/python` and `PYTHONPATH=python:scripts`. New campaign `experiments/local_expert_v27`, worker `runs/le27_confirmation_001`, anonymous review `runs/le27_confirmation_review_001`.

## Frozen design

Read Phase26/NEXT.md and its immutable completion manifest. New target vocabulary/seeds only: 60 prompts, two seeds each, within familiar Phase24 scene templates. All60 words are new against60 prior lab dataset manifests; exact prompt overlap zero, word-masked template overlap60. Historical seed scan includes every prior lab run/campaign top-level JSON/JSONL, restoring Phase17 and failed attempts,869 distinct excluded integer seed values. This does not establish semantic independence from familiar task templates or from model pretraining.

Before allocation:

```bash
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python -m pytest -q tests/test_verified_confirmation.py tests/test_short_correction.py tests/test_short_verifier_audit.py tests/test_evidence_verifier.py
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_verified_confirmation.py dataset
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_verified_confirmation.py run
```

Initial CPU test output is preserved in tests.txt: a toy-test assertion indexed a missing zero-valued text-encoding counter after conversion from Counter to dict. The production count check passed; fix only reads missing keys as zero. All16 relevant tests then passed in2.29s (`tests_after_counter_fix.txt`) before allocation. The additional paired-unknown/zero-acceptance analysis test passed in2.08s after submission (`analysis_tests.txt`). Analysis sources were attested before review.

## Worker

Job1059655: one L40,4CPUs,96GB,30min, minilab-gpu. Submit script and exact command in run/submission.json and ledger. `scontrol update JobId=1059655 Requeue=0` disabled automatic requeue; no unrelated jobs touched. Total campaign budget0.5 allocated GPU-hour including failures, no automatic retry.

Worker executes snapshotted `python -m opdlab.verified_confirmation --run <absolute-run-path>`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`. It reuses tested rollout/short-branch functions and unmodified short_correction_verifier_v1. Both old G/O2 trajectories and PNGs must replay exactly before new images. Generation reuses the first3G actions within each new pair and requeries G at perturbed states. Diffusion weights are released before loading scorer. An old paired scorer output must replay exactly before scoring new images. All predictions freeze before anonymous review.

Expected generation including replay: G1818/O242,2302CFG-expanded examples,242decodes,122text encodings,0student/behavior/backward/updates. New scorer counts240images,480whole-image OCR,480GenEval; two scorer replay images and row-dependent rereads separately charged. Actual counters, exit status and allocation time are recorded at completion.

## Review and finalization

Wait for result.json visibility and absence of RUNNING/failure.json; scheduler completion alone is insufficient on this filesystem. Inspect no scorer predictions/decisions, aggregate results or method key before observations are frozen.

```bash
../../conda_envs/envs/diffusionopd/bin/python scripts/refresh_campaign.py --campaign experiments/local_expert_v27
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_verified_confirmation.py prepare
# Observe all120 A/B pairs at native512px; record draft; freeze before method/score join.
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_verified_confirmation.py freeze
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/summarize_verified_confirmation.py
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/report_verified_confirmation.py
FONTCONFIG_FILE="$PWD/.tools/report-browser/fonts/fonts.conf" .tools/report-browser/node_modules/node/bin/node runs/verified_confirmation_report_validation_001/check.cjs
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/verify_verified_confirmation.py --write
```

Use --write once; omit for read-only rechecks. All240 saved paths are checked for exact Euler transitions, CPU noise and common prefixes. Final audit independently reconstructs policy counts, precision denominators, six paired bootstrap intervals and gate, verifies images/frozen observations/report and unchanged Phase22–26 manifests. Browser checks all240 native images, aspect ratios, filters, evidence access, mobile overflow and offline operation. No automatic training follows a gate pass; no rule changes follow a failure.

## Actual completion

Slurm1059655 completed on ai-gpu12, exit0:0, allocated682s =0.1894444444 GPU-hour; worker elapsed648.7196s. No failed GPU attempts, retries or next-phase jobs. All generation counters matched the frozen protocol. Actual new verifier work:240 endpoint images,480 OCR detection views,480 GenEval clause views,509 OCR recognition crops. Including the two historical scorer replay images:242 endpoints,484 OCR detection views,484 GenEval clause views,513 recognition crops. Both generation replays and the raw-evidence/decision scorer replay were exact.

Saved-trajectory verification checked2400 Euler transitions (2040 newly computed transitions, with shared prefix states duplicated in saved pairs),120 paired prefixes,120 CPU-seeded initial noises and60 paired text contexts. It also validated the saved model, runtime and input hashes. The initial student adapter was bookkeeping only; no student or behavior predictions, backward queries or optimizer updates entered this experiment.

Anonymous review inspected all120 contact sheets at native512px. Preserve observations_initial_draft.jsonl, observations_draft.jsonl, observations.json, observations_frozen.json, review_revisions.json and record_observations.py. Cases14/19/29/97 were rechecked before key joining; no draft labels changed. Ambiguous text in29A/97B remained unknown. Original predictions, rubric, analysis code and template hashes remained unchanged from their pre-review attestations. Contact-sheet hashes and anonymous image hashes are archived separately.

Analysis ran only after the complete observation freeze. Confirmed complete-scene counts: G49/120, fixedOCR2 70/120, verified61/120. Verified accepted16, with13 confirmed repairs,0 known harms,1 unknown and2 neutral; recall13/26. Primary verified-minus-G mean0.1,95% interval[0.05,0.1583333], unchanged under the paired unknown-adverse analysis. The prospective gate passed. FixedOCR2 exceeds verified by9 successes, so the conclusion is conservative selection, not superior overall generation or learning. See STATUS.md and interpretation.json for accepted partial repairs, ambiguous container classification and15 false-positive text certifications caused by additional lettering.

Report generation produced a standalone72.02MB HTML with240 embedded PNGs. Playwright Chromium successfully decoded every image, checked native512px dimensions/aspect ratios, all family and decision filters, fixed summary counts, evidence access, zoom/escape, mobile overflow and zero external requests/errors. Desktop and mobile screenshots were inspected. Independent audit reconstructed all policy counts and six clustered intervals, verified the gate and HTML image hashes, and confirmed unchanged Phase22–26 completion manifests (41/42/55/53/55 files respectively). Initial CPU-test failure output remains archived; no analysis/browser/GPU failure occurred after it.

Completion manifest anchors this campaign's documents, source/tests, worker top-level manifests/output hashes, anonymous review/observations and browser artifacts. Large worker trajectories/images and copied upstream sources are transitively anchored by the worker manifest/result and were validated. NEXT.md is a proposed small learning study only; no Phase28 allocation or model update was performed.

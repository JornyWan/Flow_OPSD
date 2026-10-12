# Phase20: collection before conditional acceptance learning

Run commands from `opd_lab`. The frozen scientific design is
`experiments/local_expert_v20/PROTOCOL.md`; the unchanged observation criteria are
in `REVIEW_RUBRIC.md`. This is an assistant-labelled development mechanism study.

## Prepare and execute

```bash
export PYTHONPATH=python:scripts
export OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python
"$OPDLAB_PY" scripts/prepare_acceptance_collection.py --attempt 001
"$OPDLAB_PY" -m pytest -q
```

Preparation refuses an existing run. Collection executes
`python -m opdlab.acceptance_collection --run runs/le20_collection_001` inside the
prepared framework snapshot with `CUBLAS_WORKSPACE_CONFIG=:4096:8`. The exact
allocation and command are retained in the run's submission artifacts and the
campaign ledger. Never run a replacement over an existing attempt.

The worker must reproduce twelve reference images and six saved states/actions
exactly before collecting 48 fresh states. It records 96 new endpoints, zero
optimizer updates, and no images from the 32 reserved evaluation seeds.

## Observe, freeze, then join

```bash
"$OPDLAB_PY" scripts/refresh_campaign.py --campaign experiments/local_expert_v20
"$OPDLAB_PY" scripts/review_acceptance_collection.py prepare
```

Inspect every native-resolution view on all 24 pages. Consult only `items.json`
and inherited observations during annotation, keeping `key.json`, endpoint scores
and candidate identities hidden. Record explicit clause/visibility flags, literal
text and uncertainty in `runs/le20_review_001/observations_draft.jsonl`. Preserve
raw proposed observations and inherited/duplicate-pixel overrides separately.

```bash
"$OPDLAB_PY" scripts/review_acceptance_collection.py freeze
"$OPDLAB_PY" scripts/review_acceptance_collection.py summarize
"$OPDLAB_PY" scripts/verify_acceptance_collection.py
```

Freeze is one-way. Subsequent verification/report recomputation is allowed;
changing frozen labels or choosing a more convenient randomization is not.
The target manifest contains all 48 states, including noop targets and rejection
reasons. Training is conditional on all three predeclared coverage requirements.
On a failed gate, leave reserved evaluation seeds unrendered and archive the
complete collection. On a passed gate, use separate fitting and evaluation jobs
with the fixed arms, schedule, checkpoints and budget from the protocol.

The verifier recomputes seeds, tensor/action/image hashes, frozen labels, target
selection and gate statistics, and checks the Phase19 archive against `9cbbf01`.
Large shared assets retain the framework's stat audit; this is not a fresh full
hash of every model weight. The preparation index retains immutable pre-run
inputs; final archive indexing must exclude its own index and transient caches.

## Current fitting and automatic continuation

Collection job1050595 completed with 9 safe states, 8 safe complete endpoints and
3 prompts with at least 2 safe states. All coverage requirements passed.
`results.json` and `verification.json` in this campaign are immutable collection
reports. Student results go to `learning_results.json` and separate verifications.

Fitting is job1050602 in `runs/le20_fit_002`; do not submit it again. Attempt001 was
retired during CPU preparation, before any GPU allocation, because evaluation
cohort metadata was missing. Its original preparation script is archived by hash.
All three arms have 128 updates / 1536 backward queries; certified and random arms
each include 32 all-noop batches, which execute the same loss with unit weights.
The new saved-action entry point is CPU-tested against the previous fitter for
weights, optimizer state and arm reset. Shared historical code was not edited.

A bounded continuation process runs from `runs/le20_controller_001/plan.json`.
Its PID and log are in that directory. It checks frozen source hashes and the
Slurm exit/result state before each transition, verifies the completed fit,
prepares and submits one separate 20-minute evaluation allocation, then verifies
outputs and prepares 64 hidden-identity review pages (256 slots / 224 outputs).
It never labels images, joins identities, retries a failed allocation, chooses a
checkpoint, or changes the experimental gate. The controller has an 85-minute
wall-clock bound. Its live stage is in `experiments/local_expert_v20/LIVE.md` and
`progress.json`. An explicit `STOP` file in the controller directory stops further
continuation; inspect existing Slurm jobs separately if stopping a running job.

The following are manual recovery commands only when the controller is no longer
running and the relevant stage has not already been created:

```bash
"$OPDLAB_PY" scripts/verify_acceptance_learning.py fit --fit-run runs/le20_fit_002
"$OPDLAB_PY" scripts/prepare_acceptance_learning.py evaluate --fit-run runs/le20_fit_002
```

After the controller has prepared review pages, inspect all views with source
identity and endpoint scores hidden. Identical initial pixels occur once in each
path's case; inherit the first recorded observation for duplicate pixels. Keep
all raw observations, uncertainty and the original rubric. Freeze every slot
before unblinding:

```bash
"$OPDLAB_PY" scripts/review_acceptance_learning.py freeze
"$OPDLAB_PY" scripts/review_acceptance_learning.py summarize
"$OPDLAB_PY" scripts/verify_acceptance_learning.py evaluate
```

The primary analysis is certified versus initial and both controls on 12 positive
fresh seeds, single-action path. All 32 single-action cases and full generation
results must remain in the report. A successful GPU job does not imply a
successful scientific pilot.

## Text-counter recovery (current)

The original continuation stopped because fit job1050602 ended FAILED after all
three arms were saved. Its only terminal exception was text inputs24 versus the
incorrect planned12. See `experiments/local_expert_v20/RECOVERY.md`. Never restart
controller001 or rewrite the failed source run.

`recover_acceptance_fit.py` created `runs/le20_fit_recovered_001`, an audited
artifact view with explicit counter provenance, original failure and all source
hashes. It performed zero training. `prepare_acceptance_evaluation.py` then
prepared the unchanged evaluation with40 text inputs; job1050650 is running.
Future `verify_acceptance_learning.py outputs|evaluate` commands must use
`--fit-run runs/le20_fit_recovered_001`. Review commands still use the same
evaluation/review paths. The old preparation script remains immutable historical
evidence; use the corrected evaluation preparer for this continuation.

## 最终评测与分析

job1050650 COMPLETED0:0，607秒。outputs核验首次遇到结束附近RUNNING标记而拒绝，worker移除后CPU重验通过，无GPU重跑。64页/256槽完整复核，freeze后summarize，evaluate核验重建全部统计通过。report_acceptance_cases.py展示全部32组224图，plot_acceptance_learning.py绘制预定主指标/完整路径/损伤/训练拟合。最终主门槛失败，结果见campaign STATUS.md；全部图像标签冻结后未改。122项测试通过。三个分配总0.894167 GPU-hours，保留原FAILED训练分配，恢复不增加训练。下一阶段尚未启动。

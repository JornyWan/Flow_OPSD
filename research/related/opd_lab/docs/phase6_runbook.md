# Phase6: fixed independent-model audit and exposed-image interface control

Use the lab working directory. The original `diffusionopd` environment runs SD3
generation, preparation, reports and tests. Qwen runs in the separate existing
`mi_reasoning` environment, without installing or upgrading any package. Keep the
exact Python path stored in each run's `site.json`: the environment attestation
compares the interpreter string as well as package versions and key source hashes.

## Original audit

Read `experiments/local_expert_v6/PROTOCOL.md` and `REVIEW_RUBRIC.md` first.
The source manifest `vlm_assets.json` hashes every Qwen asset including all five
weight shards; `model_index_audit.json` checks exact index/shard tensor coverage.
`effective_processor.json` records the actual fast processor chosen by the pinned
Transformers version. All inference records have image grid [1,36,36], i.e.504x504
pixels before spatial merging, with 324 image tokens. The processor's serialized
legacy `size` and explicit `max_pixels` differ; actual grids verify the effective
request size. Do not silently switch to the slow processor in a reproduction.

```bash
export PYTHONPATH="$PWD/python"
export OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python
"$OPDLAB_PY" scripts/prepare_vlm_audit.py --stage generation --attempt 001
"$OPDLAB_PY" scripts/prepare_vlm_audit.py --stage smoke --attempt 001
```

These prepare immutable `le6_generation_001` and `le6_vlm_smoke_001` runs. Existing
names must fail; the completed attempts in this workspace are never overwritten.
Each run's `submission.json`, `job.sh` and campaign ledger retain the exact Slurm
command. Generation worker is `python -m opdlab.vlm_calibration --run ABS_RUN`;
VLM worker is `SITE_PY ABS_RUN/worker.py --run ABS_RUN`. Submit oneL40,4CPU,96G;
generation/smoke time limits10min, main30min. The VLM site deliberately omits the
training environment's CUDA_HOME and uses offline local model paths.

After generation, prepare all16 review images. Read images/prompts only, apply
the fixed rubric, save observations, and freeze their SHA/UTC before reading
automatic scores. Labels are provisional implementing-assistant observations.

```bash
"$OPDLAB_PY" scripts/prepare_vlm_calibration_review.py \
  --run runs/le6_generation_001 --out runs/le6_calibration_review_001
"$OPDLAB_PY" scripts/prepare_vlm_audit.py --stage main --attempt 001
```

Main preparation requires four valid smoke responses and completed generation.
It excludes the two smoke images, yielding74; final analysis merges smoke+main
for76 unique images. The original helper binds smoke to `campaign-attempt`; a
failed smoke recovery needs an explicit new campaign/recovery design, not an
overwrite or reuse of a partially written response file.

```bash
"$OPDLAB_PY" scripts/summarize_vlm_audit.py \
  --smoke runs/le6_vlm_smoke_001 --main runs/le6_vlm_main_001 --out runs/le6_report_001
"$OPDLAB_PY" scripts/plot_vlm_audit.py --run runs/le6_report_001
"$OPDLAB_PY" scripts/explain_vlm_audit.py \
  --run runs/le6_report_001 --out runs/le6_report_001/failure_analysis.json
```

The summary checks plan, source, image and reference hashes, replays all parsing,
and verifies complete UID/call/counter coverage. Null references are excluded;
null predictions retain denominator coverage and cannot inflate usable recall.
All15 original invalid outputs remain invalid. The failure export inspects raw
JSON for explanation only and never salvages predictions into the main metrics.

## Post-audit relation control

`RELATION_CONTROL_PROTOCOL.md` was frozen after the original audit and before
these additional requests. All20 relation images are already exposed. The new
literal subject/direction/reference interface has two independent calls/image,
given and inverse; it is not a new calibration benchmark. It also changes request
scope and output schema, so it does not isolate index encoding as the sole cause.

```bash
"$OPDLAB_PY" scripts/prepare_relation_control.py --attempt 001
# Submit SITE_PY ABS_RUN/run_relation_control.py --run ABS_RUN; oneL40,10min.
"$OPDLAB_PY" scripts/summarize_relation_control.py \
  --run runs/le6_relation_control_001 --out runs/le6_relation_report_001
"$OPDLAB_PY" scripts/refresh_campaign.py --campaign experiments/local_expert_v6
```

Original responses/plans/scores remain unchanged. The control produced15 top-level
`null` replies, which remain unknown/invalid under its object schema; no retry or
reinterpretation improves the reported coverage. An absent object does not make
the inverse relation true, so no inverse reference labels are fabricated.

Cost comes from Slurm allocation including imports and full asset checks, not
only model-loop timers. CPU initial model hashing took363.17 seconds. All other
timed measurements are in their JSON artifacts; preparation and visual review
wall time was not instrumented as a complete total. No optimizer updates or
teacher queries occurred in this phase. Keep `report_index.json` as the evidence
index; the previous README version is retained at commit40e9e39.

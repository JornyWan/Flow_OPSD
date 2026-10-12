# Phase7 execution and review

Scope/criteria are in `experiments/local_expert_v7/PROTOCOL.md`. The runtime
snapshot is the authority for every submitted job; later working-tree scripts
do not alter an existing run. No prior environment/upstream/reward edits.

## Recorded attempts

| Job | Run | Purpose |
|---|---|---|
|1048850|`runs/le7_native_ocr_001`|32-update native SOPD-OCR subset control|
|1048860|`runs/le7_fit_001`|five independent single-query fits and one joint fit|
|1048861|`runs/le7_capability_001`|64 prompts × five teacher/CFG configurations, ten teacher-copy checks|

Each requests one L40 GPU, four CPUs,96GB RAM, one-hour maximum. Scheduler
accounting in the campaign ledger includes startup and failed attempts. Native
`progress.jsonl` records actual optimizer updates; native EMA eval is at0/8/16/24,
not32. Checkpoints8/16/24/32 save raw and EMA LoRA, optimizer/scaler/RNG separately.

Native was prepared with the original `sopd_ocr` preset and only these explicit
overrides: `num_epochs=32`, `sample.train_batch_size=1`, `train.batch_size=1`,
`sample.test_batch_size=1`, `sample.eval_num_steps=10`, `eval_freq=8`, `save_freq=8`;
`--world-size 1 --smoke-examples 16`. Config audit/data transformation/runtime
patches record all bindings and subset changes. Native loss/optimizer/EMA and
multi-time supervision are unchanged. Seed42, flow10,512px. This is a subset
diagnostic, not the default multi-GPU paper recipe.

Local preparation:

```bash
export PYTHONPATH=python
../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_learnability.py --site site.local.json
```

The preparer refuses an existing run, deterministically selects old positives,
hashes the old tensors/labels/images and snapshots code/data/assets. Exact
submission and worker commands are in `ledger.jsonl` and each `job.sh`.
The selected queries remain fixed;128 updates on a single query and128 on five
queries cost different amounts. Total local backward queries=1280;768 optimizer
updates across six independent arms.60 metric rows,100 fitted endpoint images,
ten replayed reference images. Store full optimizer/scaler/RNG only at each
arm's final checkpoint, raw student tensors at every audit milestone.

Review after the fitting completion marker:

```bash
../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_learnability_review.py \
  --run runs/le7_fit_001 --out runs/le7_review_001
```

Read only items/sheets first. Record all20 final images using the frozen rubric,
then save `observations_frozen.json` with observations/key/rubric SHA256 before
opening the key. These remain provisional observations by the implementing
assistant, not an independent human annotation set.

`native_history.json` is extracted offline from a temporary copy of the completed
native `.wandb` binary with the installed W&B DataStore/protobuf reader; no sync
or network write. It preserves all32 original history records. The source path
and source SHA256 are recorded in results. Metrics refer to the native OCR
reward, distinct from local v1 exact text matching.

After both local workers and the review are complete:

```bash
../../conda_envs/envs/diffusionopd/bin/python scripts/summarize_learnability.py
../../conda_envs/envs/diffusionopd/bin/python scripts/refresh_campaign.py \
  --campaign experiments/local_expert_v7
```

The summary checks completed output/image hashes and full matrix sizes. Use the
predeclared update128, never choose a different checkpoint for each image. Later
CPU reporting files are hashed separately from the frozen worker source.

For NFS visibility, refresh only a phase7 output directory with `os.utime` when
needed; never change source inputs, global mount options or unrelated jobs.
Phase6's recorded README hash refers to its historical revision `e384a09`;
subsequent phase navigation updates do not rewrite the old phase6 report/index.

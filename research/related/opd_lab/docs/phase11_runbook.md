# Phase11 frozen-checkpoint transfer evaluation

Protocol and rubric are frozen in experiments/local_expert_v11 and copied into
the run. No changes to old checkpoints, observations, environments or upstreams.
Raw tensor states are loaded through the existing backend; no temporary LoRA
export, checkpoint conversion, new training, teacher query or candidate selection.

```bash
export PYTHONPATH=python
../../conda_envs/envs/diffusionopd/bin/python -m pytest -q
../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_transfer_eval.py --site site.local.json
../../conda_envs/envs/diffusionopd/bin/python scripts/submit_slurm.py \
  --run runs/le11_transfer_001 --stage le11_transfer --gres gpu:l40:1 --time 00:30:00 \
  --ledger experiments/local_expert_v11/ledger.jsonl -- \
  ../../conda_envs/envs/diffusionopd/bin/python -m opdlab.transfer_eval --run runs/le11_transfer_001
```

Job1049137, one L40 GPU/four CPUs/96GB,30-minute cap. Each model generates42
fresh pairs and one known apple anchor. The4 anchors must match prior pixel
hashes and are excluded from transfer statistics.172 total decoded/scored images,
1720 model forwards/3440 CFG examples expected.168 rows form the transfer matrix.

Both training prompts/new seeds and existing validation prompts are included in
the dataset snapshot. The generic config retains split=train from the source
run; this dedicated worker explicitly loads both splits and obeys the frozen
42-pair plan. It never samples the64 test prompts. Fresh noise seeds are stored,
and generated initial-noise hashes must match across all four models.

```bash
../../conda_envs/envs/diffusionopd/bin/python scripts/review_transfer.py prepare \
  --run runs/le11_transfer_001 --review runs/le11_review_001
# Read only items and paired A/B/C/D sheets; record all168 views and freeze hashes.
../../conda_envs/envs/diffusionopd/bin/python scripts/review_transfer.py summarize \
  --run runs/le11_transfer_001 --review runs/le11_review_001
../../conda_envs/envs/diffusionopd/bin/python scripts/refresh_campaign.py \
  --campaign experiments/local_expert_v11
```

Pair order and model positions within each pair are independently permuted.
The model key/scores stay unopened until all observations are frozen. Draft
observations may be accumulated by sheet; a draft is not a final result. Unknown
remains unknown; complete-scene success requires all clauses and full visibility.
Review is implementing-assistant assessment, not independent human ground truth.

Color labels respect each encoded color clause's count requirement; notes retain
observed colors independently. Extra text/plate lettering, cropped objects and
ambiguous multiplicity remain explicit. Prompt-clustered paired bootstrap uses
2000 resamples/seed0; conditional regression metrics expose their denominators.
The existing validation split is development evidence, not a pristine test set.

Only refresh this new run directory for NFS visibility if needed. Historical
README references: phase6 e384a09,phase7 8c1e160,phase8/9 d9c392c,phase10 9dcc18b.

## Completed execution and review

Job1049137 completed with exit0:0,340 allocated seconds,0.0944444444 GPU-hours,
no failed attempt. All168 transfer images and4 exact replay anchors are present.
No training or new teacher queries. CPU suite:68 passed in12.83s (tests.txt).

All42 sheets were inspected using the public items and A/B/C/D images, with
native-resolution views for ambiguous lettering. record_draft.py accumulated
observations_draft.jsonl without reading the key. Completeness, clause schema and
identical-pixel consistency were checked before writing observations.json and
observations_frozen.json; only afterward was summarize run. These review files,
including the draft and its helper, remain in runs/le11_review_001. SHA256 of the
unopened key was recorded without reading its model mapping during review.

```bash
../../conda_envs/envs/diffusionopd/bin/python scripts/plot_transfer_eval.py
../../conda_envs/envs/diffusionopd/bin/python scripts/verify_transfer_eval.py
```

Verification checks immutable source/inputs, complete image matrix and hashes,
same seeds/noise for all models, raw tensor hashes, exact anchors, count totals,
frozen review/image copies, independently recomputed aggregate counts, Slurm
accounting, historical phase6–10 indices and clean pinned upstreams. It does not
re-hash the full shared pretrained weights or repeat GPU generation. Source and
large assets are covered by the existing manifests/attestations and inventories.

Results: complete scenes joint128/sparse32/sparse320/dense32 are2/4/4/2 out of10
fresh training-prompt seeds, and9/11/12/9 out of32 existing validation prompts.
Known-harm sample counts are0/1/1/0 and0/3/5/4 respectively (baseline compared to
itself has zero by definition). Apple transfer succeeds on one of two new seeds
for both sparse checkpoints. Full denominators, unknowns and paired intervals
are in results.json; STATUS.md explains why this is partial transfer with a
preservation problem, rather than a stable improvement claim. No phase12 job
was submitted. NEXT.md describes a prospective train-only collection gate.

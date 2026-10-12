# Phase16 frozen-action transfer diagnostic

Run from opd_lab with `OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python`
and `PYTHONPATH=python:scripts`. Historical inputs are immutable. The worker is
snapshotted before submission; no optimizer or new checkpoints are created.

```bash
"$OPDLAB_PY" -m pytest -q tests/test_action_transfer.py
"$OPDLAB_PY" scripts/prepare_action_transfer.py --attempt 001
"$OPDLAB_PY" scripts/submit_slurm.py --run runs/le16_transfer_001 \
  --stage le16_transfer --gres gpu:l40:1 --time 00:20:00 \
  --ledger experiments/local_expert_v16/ledger.jsonl -- \
  ../../conda_envs/envs/diffusionopd/bin/python -m opdlab.action_transfer \
  --run runs/le16_transfer_001
"$OPDLAB_PY" scripts/refresh_campaign.py --campaign experiments/local_expert_v16
"$OPDLAB_PY" scripts/review_action_transfer.py prepare
```

Job1049629 completed on ai-gpu12, allocated158s. Run starts with full joint128
replays, capturing z3 and the noop action during the original trajectory. All18
behavior noops must match. The six old state/context/action payloads must match
phase12 exactly. Replay six selected teacher endpoints and all18 full learned
images, checking all42 old PNG hashes before new branching. Expert adapters are
loaded via training=True only; execute is no_grad and never creates an optimizer.
Every new suffix has six behavior calls at indices4..9. OCR CFG4.5, GenEval CFG1.

Review all42 new views through public items.json and 18 sheets, without reading
key.json or endpoint scores. Six single-view sheets disclose old cohort; the other
12 sheets randomize the three paths. Prior phase15 images were already exposed.
In this run no new image has identical inherited pixels. The public-only helper
`runs/le16_review_001/record_draft.py` records explicit clause/visibility bool/null,
literal text and notes. Thirteen additional native images were inspected; exact
list and ambiguity decisions are in REVIEW_CAVEATS.md, recorded before key join.

```bash
"$OPDLAB_PY" runs/le16_review_001/record_draft.py
"$OPDLAB_PY" scripts/review_action_transfer.py freeze
"$OPDLAB_PY" scripts/review_action_transfer.py summarize
"$OPDLAB_PY" scripts/explain_action_transfer.py
"$OPDLAB_PY" scripts/plot_action_transfer.py
"$OPDLAB_PY" scripts/verify_action_transfer.py
```

Keep all42 reference labels inherited, then join42 new labels. Compare encoded
clauses and visibility separately; no safe repair if either endpoint has unknown.
Report complete-scene unknown as not complete, without calling it a known error.
Report fresh teachers independently and oracle availability separately. A prompt
cluster has two fresh seeds and one old anchor; bootstrap averages seeds within
prompt. A zero-width observed local-vs-full CI on12 identical indicators is not
proof of population equivalence. No independent confirmation claims.

All97 tests passed (23.18s). First verification used strict float equality for
CPU-recomputed CUDA action reductions and failed at4.44e-16; corrected only the
verification tolerance to1e-12. Source tensors, images and checkpoints retain
exact hash checks. Second verification passed, including all phases6–15 archived
indices and both clean pinned upstreams. No rerun GPU cost. Large shared model
assets are verified by their recorded inventory/stat contract, not newly full
content hashes. New scripts, results, frozen review, costs, protocol, hashes and
STATUS/NEXT are indexed; old indices remain unchanged.

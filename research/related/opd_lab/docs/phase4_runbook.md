# Phase4 action diagnostic runbook

From `opd_lab`, use the pinned local Python and a fresh attempt for GPU reruns.
The completed attempt001 outputs must not be overwritten.

```bash
OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python
export PYTHONPATH=python

# Freeze PROTOCOL.md before prepare; pins every accepted phase3 query.
"$OPDLAB_PY" scripts/prepare_action_probe.py \
  --parent runs/le3_probe_001 --site site.local.json --attempt 002
"$OPDLAB_PY" scripts/prepare_target_review.py \
  --run runs/le4_action_002 --out runs/le4_target_review_002
```

Submit through `scripts/submit_slurm.py`: one L40-class GPU, 35-minute limit,
stage `le4_action`, phase4 ledger, and command `<absolute python> -m
opdlab.action_probe --run <absolute run>`. The launcher selects the frozen framework
and records the exact job command, environment and source hashes.

Review every A/B sheet with the frozen rubric before opening `key.json` or viewing
new ray outcomes. Populate all observations, including uncertainties. Freeze SHA
and UTC time in `observations_frozen.json` before reveal; completed attempt001
provides the schema. Never copy observations into a different attempt or edit
them after reveal. Separate later caveats explicitly.

```bash
"$OPDLAB_PY" scripts/summarize_target_review.py \
  --run runs/le4_action_002 --review runs/le4_target_review_002
"$OPDLAB_PY" scripts/summarize_action_probe.py \
  --run runs/le4_action_002 --out runs/le4_report_002
"$OPDLAB_PY" scripts/refresh_campaign.py --campaign experiments/local_expert_v4
```

The reporter checks the complete tensor/image grid, parent inputs, frozen behavior,
checkpoint identity, recomputed projections and 132 reference comparisons. Its
source-hash-specific correction for attempt001's tuple/list comparison flag is
documented in `experiments/local_expert_v4/INCIDENTS.md`; unexpected discrepancies
still fail. Preserve original outcomes and failed attempts; count all GPU time.

PNG/PDF charts are export artifacts. Bootstrap intervals condition on selected
TRAIN prompts and one seed. This diagnostic is not normal deployment evaluation.

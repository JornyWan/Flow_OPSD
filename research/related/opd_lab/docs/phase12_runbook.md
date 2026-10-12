# Phase12 prospective candidate and selector audit

Run from opd_lab with the existing OPD environment and unchanged upstreams.
The dedicated worker loads the exact joint128 raw tensors; it does not export
or convert the adapter. It calls existing collect_query/select at index3/alpha1.

```bash
export PYTHONPATH=python
../../conda_envs/envs/diffusionopd/bin/python -m pytest -q
../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_prospective.py --site site.local.json
../../conda_envs/envs/diffusionopd/bin/python scripts/submit_slurm.py \
  --run runs/le12_collect_001 --stage le12_collect --gres gpu:l40:1 --time 00:30:00 \
  --ledger experiments/local_expert_v12/ledger.jsonl -- \
  ../../conda_envs/envs/diffusionopd/bin/python -m opdlab.prospective --run runs/le12_collect_001
```

192 candidate images plus one excluded exact-replay anchor; zero training updates.
New config changes only alpha grid, descriptive query/accumulation counts and
save_candidates. Actual inference/selection settings and old losses remain intact.
All64 train prompts are used once in balanced order. Config retains the same
dataset bytes with other splits present, but worker explicitly loads train only.

Candidates and decisions are saved before review. During generation/review,
inspect status/counts only, not candidate scores or selected action. Review all
64 independently permuted A/B/C sheets before freezing observations and joining
the private key. The protocol's stronger baseline-unknown safety requirement is
a reporting contract; it does not alter the automatic selector being audited.

## Attempt history

Attempt001/job1049193 failed before any completed candidate query: the backend's
training=False mode omitted expert adapters, causing KeyError('ocr'). It produced
one exact anchor but no candidate results or updates. Its source snapshot, failure
and Slurm log are retained, and its37 allocated seconds are included in total cost.
The corrected worker uses training=True to load teachers; every prediction is still
no-grad, with no optimizer. This mode does not itself perform training.

After73 CPU tests passed, attempt002/job1049199 was submitted with the same30-minute
cap. Its full probe_plan is identical to attempt001: same seeds, candidates and
selection rules. Worker source differs only in teacher-enabled initialization.

```bash
../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_prospective.py --site site.local.json --attempt 002
../../conda_envs/envs/diffusionopd/bin/python scripts/submit_slurm.py \
  --run runs/le12_collect_002 --stage le12_collect --gres gpu:l40:1 --time 00:30:00 \
  --ledger experiments/local_expert_v12/ledger.jsonl -- \
  ../../conda_envs/envs/diffusionopd/bin/python -m opdlab.prospective --run runs/le12_collect_002
../../conda_envs/envs/diffusionopd/bin/python scripts/review_prospective.py prepare \
  --run runs/le12_collect_002 --review runs/le12_review_001
# Read public items/sheets only, record observations_draft.jsonl for all64 cases.
../../conda_envs/envs/diffusionopd/bin/python scripts/review_prospective.py freeze \
  --run runs/le12_collect_002 --review runs/le12_review_001
../../conda_envs/envs/diffusionopd/bin/python scripts/review_prospective.py summarize \
  --run runs/le12_collect_002 --review runs/le12_review_001
../../conda_envs/envs/diffusionopd/bin/python scripts/refresh_campaign.py --campaign experiments/local_expert_v12
../../conda_envs/envs/diffusionopd/bin/python scripts/plot_prospective.py
../../conda_envs/envs/diffusionopd/bin/python scripts/diagnose_prospective.py
../../conda_envs/envs/diffusionopd/bin/python scripts/verify_prospective.py
```

Historical README references for verification: phase6 e384a09,phase7 8c1e160,
phase8/9 d9c392c,phase10 9dcc18b,phase11 3923b6a. Old artifacts remain immutable.

## Completed audit

Attempt002 completed with exit0:0 in391 allocated seconds. Together with the
37-second failed attempt, cost is428 seconds /0.118888889 GPU-hours. All64 states,
all192 actions/endpoints and the pre-review decisions are retained. The old anchor
matches exactly; no optimizer or backward updates occurred. The frozen draft covers
all64 cases and192 views before the key join. No phase12 job remains active.

Selected endpoint strict successes are25/64 versus21/64 for noop. Accepted queries
split into6 certified safe repairs,6 non-repairs,3 known harms and5 unresolved.
All six available safe-repair states were captured, but conservative precision is
6/20. All four harmed baseline-true clauses were proxy-false and consequently
outside the original protection set. This is post-unblind descriptive diagnosis,
not a tested verifier fix. All four preregistered learning conditions failed.

The next verifier design belongs to a new development/confirmation stage; the
current selector, labels and candidate grid are not adjusted. See campaign STATUS.md,
NEXT.md and measurement_diagnosis.json. Runs are local retained artifacts referenced
by the report index; large tensors/images are not duplicated into git. Verification
checks historical phase6–11 indices and clean pinned upstream repositories.

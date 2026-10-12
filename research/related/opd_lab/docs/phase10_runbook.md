# Phase10 execution and review

Frozen protocol: experiments/local_expert_v10/PROTOCOL.md. The phase9/NEXT
proposal is now concretized with common reset AdamW/scaler/RNG, weight warm start
from joint128. This is a selected-case learning diagnostic, not full resume,
online training or a held-out comparison. Shared environments/upstreams unchanged.

```bash
export PYTHONPATH=python
../../conda_envs/envs/diffusionopd/bin/python -m pytest -q
../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_trajectory_fit.py --site site.local.json
../../conda_envs/envs/diffusionopd/bin/python scripts/submit_slurm.py \
  --run runs/le10_fit_001 --stage le10_fit --gres gpu:l40:1 --time 01:00:00 \
  --ledger experiments/local_expert_v10/ledger.jsonl -- \
  ../../conda_envs/envs/diffusionopd/bin/python -m opdlab.trajectory_fit --run runs/le10_fit_001
```

Job1049122, one L40 GPU, four CPUs,96GB, one-hour cap. The worker runs the frozen
source snapshot. Plan JSON uses sorted keys so dense runs before sparse; both
arms restore identical initial state. Actual points per update are in the arm
plan (50 and5), overriding the generic accumulation metadata for this diagnostic.
Loss is averaged across those points by the existing fitting-probe helper with
preservation coefficient1. Only the apple intervention is marked accepted; other
points are explicitly current-student preservation, not accepted expert targets.

Five baseline full trajectories and six reference images must match phase9.
All50 branch transitions are checked, and all49 preservation actions must exactly
match frozen joint128 at their saved states. No new teacher query is made. This
counts saved forward calls at the model boundary; activation recomputation in
gradient checkpointing is not counted as separate API forwards. Report GPU-hours
alongside the equal1600-backward-query budgets, including audits and exports.
Expected API counts from the frozen loop:3600 student forwards (3200 training,
250 fixed-point audits,150 full-rollout steps),223 behavior forwards (50 target
checks,50 baseline steps,6 target-reference suffix steps,117 local suffix steps),
7646 CFG examples,36 decoded/verifier images. These accounting expectations are
derived from the loop; they are not additional model-quality selection criteria.

At audits dense0/32 and sparse0/32/320, all50 points are measured. Raw weights are
saved at all five audits; complete Adam/scaler/RNG only at each terminal state.
Full trajectories and local/full endpoints are saved at the three nonzero audits.
Local endpoints always use frozen phase9 joint128 prefix/state and continuation.

After completion:

```bash
../../conda_envs/envs/diffusionopd/bin/python scripts/review_trajectory_fit.py prepare \
  --run runs/le10_fit_001 --review runs/le10_review_001
# Inspect items/sheets only; freeze observations and their hashes before key join.
../../conda_envs/envs/diffusionopd/bin/python scripts/review_trajectory_fit.py summarize \
  --run runs/le10_fit_001 --review runs/le10_review_001
../../conda_envs/envs/diffusionopd/bin/python scripts/refresh_campaign.py \
  --campaign experiments/local_expert_v10
```

Every endpoint participates. Exact phase9 pixel matches inherit labels; newly
generated unique images are reviewed without arm/step/mode/score. Unknowns remain
unknown, and implementing-assistant review is not independent annotation.
Only refresh the new output run directory with os.utime if NFS listings lag.
Historical phase8/9 README hashes refer to d9c392c; phase6/7 history remains intact.

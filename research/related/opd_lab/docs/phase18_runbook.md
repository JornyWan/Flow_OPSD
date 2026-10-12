# Phase18 saved-query supervision study

## Completed analysis: evaluation recovered after the time limit

All200 observations are now frozen, statistics rebuilt, and verification passed.
Do not rerun freeze or overwrite observations. See the final campaign STATUS/NEXT.
The commands below document the completed sequence.

Source le18_supervision_003 completed all384 updates and saved all three final
training states, but job1050296 timed out at158/185 endpoints. Preserve source
TIMEOUT and stale lock. Run le18_completion_001/job1050332 validates and reuses
training, replays31 reference outputs exactly, and fills27 missing slots without
any optimizer update. See PROTOCOL_COMPLETION.md and completion_attempt.json.

Use the following run/review paths for finish, freeze, summarize and verify:

```bash
export OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python
export PYTHONPATH=python:scripts
"$OPDLAB_PY" scripts/finish_supervision_run.py \
  --run runs/le18_completion_001 --review runs/le18_review_completed \
  --campaign experiments/local_expert_v18
# Only after every image observation has been recorded:
"$OPDLAB_PY" scripts/review_supervision.py freeze \
  --run runs/le18_completion_001 --review runs/le18_review_completed \
  --campaign experiments/local_expert_v18
"$OPDLAB_PY" scripts/review_supervision.py summarize \
  --run runs/le18_completion_001 --review runs/le18_review_completed \
  --campaign experiments/local_expert_v18
"$OPDLAB_PY" scripts/refresh_campaign.py --campaign experiments/local_expert_v18
"$OPDLAB_PY" scripts/verify_supervision_study.py \
  --run runs/le18_completion_001 --review runs/le18_review_completed \
  --campaign experiments/local_expert_v18
"$OPDLAB_PY" scripts/plot_supervision_study.py --campaign experiments/local_expert_v18
```

Do not rerun preparation into an existing directory. Main result counts describe
unique scientific work reconstructed from records; source cumulative counters and
per-arm elapsed timers were lost at timeout. Measured completion counters and
Slurm allocation costs are recorded separately. The live worker now persists arm
reports and fitting/execution summaries before evaluation for future studies;
the already-frozen attempts are unchanged.


## Historical attempt003: deterministic fitting after measured replay failure

This source attempt ended TIMEOUT after training; use the completion run above.
The commands in this section are historical and are not active next steps.

Run `runs/le18_supervision_003`, GPU1050296; CPU1050297 prepares
`runs/le18_review_003` after GPU completion. Do not resubmit these directories.
Attempt001 failed its historical final-weight gate after128 single-arm updates;
attempt002 failed its first reference state before fitting. Neither generated new
evaluation images. The diagnostic run1050293 completed in144allocated seconds.
Default repeated training diverged; deterministic repetitions and the old/new
fitters matched exactly at two updates. See the numerical amendment and diagnostic
artifacts in the campaign. This does not claim historical128-update model replay.

Preparation/execution already performed:

```bash
export OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python
export PYTHONPATH=python:scripts
"$OPDLAB_PY" scripts/prepare_supervision_replay.py --attempt 003
"$OPDLAB_PY" scripts/submit_slurm.py --run runs/le18_supervision_003 \
  --stage le18_supervision_deterministic --gres gpu:l40:1 --time 00:45:00 \
  --ledger experiments/local_expert_v18/ledger.jsonl -- \
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /midtier/minilab/scratch/chw4026/projects/side_project/generative/conda_envs/envs/diffusionopd/bin/python \
  -m opdlab.supervision_study --run runs/le18_supervision_003
```

The diagnostic was a separate6minute-capped L40 allocation running the snapshotted
`opdlab.fitting_repeatability` with the same CUBLAS setting. Its plan contains all15
source states and the original12-state schedule; two updates are repeated three
times using the historical function under each numerical mode, then compared to
the current function. Its original arm names all receive identical targets here.

Next commands (review only after GPU success and manual observation):

```bash
"$OPDLAB_PY" scripts/refresh_campaign.py --campaign experiments/local_expert_v18
# The afterany callback normally prepares the review. If it failed, diagnose first;
# finish is idempotent and preserves an existing observation draft.
"$OPDLAB_PY" scripts/finish_supervision_run.py \
  --run runs/le18_supervision_003 --review runs/le18_review_003 \
  --campaign experiments/local_expert_v18
# Inspect only items/inherited/public images and fill every observation before:
"$OPDLAB_PY" scripts/review_supervision.py freeze \
  --run runs/le18_supervision_003 --review runs/le18_review_003 \
  --campaign experiments/local_expert_v18
"$OPDLAB_PY" scripts/review_supervision.py summarize \
  --run runs/le18_supervision_003 --review runs/le18_review_003 \
  --campaign experiments/local_expert_v18
"$OPDLAB_PY" scripts/verify_supervision_study.py \
  --run runs/le18_supervision_003 --review runs/le18_review_003 \
  --campaign experiments/local_expert_v18
"$OPDLAB_PY" scripts/plot_supervision_study.py --campaign experiments/local_expert_v18
```

The original354-artifact preparation index remains unchanged. The separate replay
preparation index covers the diagnostic, failed L40 outputs, numerical amendment,
new immutable run inputs and callback. Dynamic progress, Slurm accounting and the
unfinished main-run outputs are not a completed scientific archive.

## Historical commands and state recorded before the failure diagnosis

The following text describes attempts001/002 at submission time; their current
terminal state is FAILED. Do not treat its old queue/fallback instructions as active.


Use `OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python` and
`PYTHONPATH=python:scripts` from opd_lab. No old phase implementation changes.
`supervision_study.py` is the common schedule-driven worker; the preparation script
is the phase-specific inventory/compiler. Labels and random assignments are data
in the frozen plan. All arms use the existing `update_once`, snapshot/restore and
single-endpoint helpers. No new loss, optimizer, gate network or sampler module.

```bash
"$OPDLAB_PY" -m pytest -q
"$OPDLAB_PY" scripts/prepare_supervision_study.py --attempt 001
"$OPDLAB_PY" scripts/submit_slurm.py --run runs/le18_supervision_001 \
  --stage le18_supervision --gres gpu:l40:1 --time 00:45:00 \
  --ledger experiments/local_expert_v18/ledger.jsonl -- \
  ../../conda_envs/envs/diffusionopd/bin/python -m opdlab.supervision_study \
  --run runs/le18_supervision_001
```

Attempt001: L40 job1049851 queued. Because its predicted wait was about70minutes,
attempt002 is a predeclared H100 equivalence attempt, job1049852, same45min cap.
Plans and worker bytes identical; resolved configs differ only by isolated dataset
paths. It must pass all24 exact image/state references and reproduce the old six-
state final model tensor hash before any multi/random fitting. No tolerance was
relaxed. If it fails, keep the L40 fallback; if both gates pass, cancel only the
unused pending fallback. Record every attempt and its allocated cost. See the
protocol amendment and ledger rather than treating a failed qualification as data.

Attempt002 outcome: H100 ai-gpu14 failed the first stored-state bitwise comparison
before any training update or new evaluation image. Allocated39seconds,
0.010833GPU-hours. L40 attempt001 remains the active run; no tolerance changes.

```bash
"$OPDLAB_PY" scripts/prepare_supervision_study.py --attempt 002
"$OPDLAB_PY" scripts/submit_slurm.py --run runs/le18_supervision_002 \
  --stage le18_h100_equivalence --gres gpu:h100:1 --time 00:45:00 \
  --ledger experiments/local_expert_v18/ledger.jsonl -- \
  ../../conda_envs/envs/diffusionopd/bin/python -m opdlab.supervision_study \
  --run runs/le18_supervision_002
"$OPDLAB_PY" scripts/refresh_campaign.py --campaign experiments/local_expert_v18
```

After a complete run, bind OPDLAB_RUN to its path and use one review directory;
do not combine outputs from partial attempts. Read only public items, sheets and
inherited labels until the entire draft freezes. All50sheets need annotations,
200slots corresponding to185evaluation outputs with15repeated baselines. Retain
unknowns and exact historical/duplicate labels. Fresh outputs are model/path/seed/
score hidden; fitting-diagnostic inherited images do not become new blind evidence.

```bash
"$OPDLAB_PY" scripts/review_supervision.py prepare --run "$OPDLAB_RUN" \
  --review runs/le18_review_001 --campaign experiments/local_expert_v18
"$OPDLAB_PY" scripts/review_supervision.py freeze --run "$OPDLAB_RUN" \
  --review runs/le18_review_001 --campaign experiments/local_expert_v18
"$OPDLAB_PY" scripts/review_supervision.py summarize --run "$OPDLAB_RUN" \
  --review runs/le18_review_001 --campaign experiments/local_expert_v18
"$OPDLAB_PY" scripts/verify_supervision_study.py --run "$OPDLAB_RUN" \
  --review runs/le18_review_001 --campaign experiments/local_expert_v18
"$OPDLAB_PY" scripts/plot_supervision_study.py --campaign experiments/local_expert_v18
```

106 CPU tests passed before freezing/submission, including a synthetic exact
replay of the old fitter, duplicate-prompt scheduling, per-step random task/teacher/
noop matching, disjoint evaluation seeds, and separate full/local report joins.
Verify the real old final weights as well; CPU tests do not establish GPU parity.

Primary comparison: multi versus single full generation on12positive fresh seeds.
The two prompts with added supervision have only4fresh states; report them and the
other8separately. Original6 and added3 fitting errors are diagnostic. All32fresh
states contribute harm checks. Archive references, schedules, models/optimizer/RNG,
15local evaluation queries, all labels, tests, costs, STATUS/NEXT and a SHA index.
Historical phase17 README must be read at70a1213 when validating its old index.

CPU follow-up1049871 is submitted on minilab-cpu,4CPUs/16GB/10min, with
`afterany:1049851`. Its scripts are copied and hashed in the primary run's
`postprocess/` directory. On success it checks frozen inputs/counters/reference
counts and exact old-model replay, then prepares review sheets and updates
progress to `ready_for_image_review`. On GPU failure it records incomplete status.
It never writes observations or scientific conclusions. Success-path idempotence
and preservation of existing drafts are tested; an actual failed002 dry-run also
confirmed it does not create a review or claim results. Full suite after adding
the callback:107passed; the earlier106passed record is the pre-training check.

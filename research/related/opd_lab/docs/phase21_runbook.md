# Phase21 execution

Frozen design: [PROTOCOL.md](../experiments/local_expert_v21/PROTOCOL.md).
All commands run from opd_lab. Use the existing OPD environment and prepared
framework snapshot; never execute over an existing attempt or edit frozen inputs.

```bash
export PYTHONPATH=python:scripts
export OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python
"$OPDLAB_PY" scripts/prepare_headroom_probe.py dataset
"$OPDLAB_PY" scripts/prepare_headroom_probe.py development
```

The dataset and development attempt have already been prepared. Development job
1056776 and confirmation job1056865 have both completed with gpu:l40:1 and a
15-minute limit. Both full reviews are frozen and summarized. See ledger.jsonl
and each run's submission.json. Do not resubmit, regenerate, or refreeze them.
Current findings: [STATUS.md](../experiments/local_expert_v21/STATUS.md).

After the worker completes and removes RUNNING:

```bash
"$OPDLAB_PY" scripts/review_headroom_probe.py verify --run runs/le21_development_001
"$OPDLAB_PY" scripts/review_headroom_probe.py prepare --run runs/le21_development_001 --review runs/le21_development_review_001
```

Inspect all48 native-resolution sheets /156 views. Consult only items.json and
REVIEW_RUBRIC.md; do not open key.json or endpoint identities until observations
are frozen. Save every explicit observation to observations_draft.jsonl. Shape:
case integer, views mapping A..M to clauses/full_visibility/visible_text/notes.
Null remains null. Any independent human feedback is stored separately.

```bash
"$OPDLAB_PY" scripts/review_headroom_probe.py freeze --run runs/le21_development_001 --review runs/le21_development_review_001
"$OPDLAB_PY" scripts/review_headroom_probe.py summarize --run runs/le21_development_001 --review runs/le21_development_review_001
```

Summarization locks development_selection.json. Only then prepare a separate
confirmation run with `prepare_headroom_probe.py confirmation`; use the same
worker/module and resource cap. Generate the entire frozen confirmation cohort,
independent of whether the development gate looked good. Repeat the same blinded
review flow with confirmation paths. No refitting, rule changes or test-time
selection of the fixed control. All policy scores and unknowns must be reported.

Each review.html embeds all PNG files for offline use. It contains no key or
method identity. It is a viewing page, not an annotation export application.

Use `scripts/refresh_campaign.py --campaign experiments/local_expert_v21` to log
Slurm state and allocated cost. GPU completion only verifies generation; the
scientific result is unavailable until complete review and frozen analysis.

Independent tensor audit commands used after each completed run:

```bash
"$OPDLAB_PY" scripts/verify_headroom_tensors.py --run runs/le21_development_001 --out experiments/local_expert_v21/development_verification.json
"$OPDLAB_PY" scripts/verify_headroom_tensors.py --run runs/le21_confirmation_001 --development-run runs/le21_development_001 --out experiments/local_expert_v21/confirmation_verification.json
"$OPDLAB_PY" scripts/report_headroom_probe.py
"$OPDLAB_PY" scripts/prepare_phase21_human_review.py
```

These artifacts already exist; commands intentionally reject overwriting. The
report embeds38 nonduplicate-policy views; full cohort review pages each embed156.
The separate targeted human review embeds20 views and exports JSON without server
access. It is a five-case post-outcome annotation audit, not a new prevalence test.
An explicit directory listing resolved transient run-file visibility after Slurm
completion; require result.json plus noRUNNING before starting postprocessing.

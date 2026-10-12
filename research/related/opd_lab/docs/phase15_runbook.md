# Phase15: labelled safe actions, matched fitting, fresh-noise transfer

Read experiments/local_expert_v15/PROTOCOL.md and REVIEW_RUBRIC.md. This pilot
reuses all six phase12 states with known safe teacher repairs, without changing
the original task or labels. It is a selected-cohort mechanism diagnostic; the
label source is the implementing assistant, not independent human annotation.
It does not resume the stopped automatic-verifier confirmation or claim a native
OPD comparison. The dense frozen-state teacher arm is explicitly named as such.

From opd_lab, using the existing diffusionopd environment:

```bash
export PYTHONPATH="$PWD/python"
export OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python
"$OPDLAB_PY" -m pytest -q
"$OPDLAB_PY" scripts/prepare_annotation_probe.py --attempt 001
"$OPDLAB_PY" scripts/submit_slurm.py \
  --run runs/le15_annotation_001 --stage le15_annotation \
  --gres gpu:l40:1 --time 01:00:00 \
  --ledger experiments/local_expert_v15/ledger.jsonl -- \
  "$OPDLAB_PY" -m opdlab.annotation_probe --run runs/le15_annotation_001
```

Existing run paths cannot be reused. Preparation freezes the framework, config,
12 states and 36 cached actions, target assignments, checkpoint, 38 evaluation
pairs and source image/label hashes. Worker checks 12 exact saved noop actions,
12 baseline pixels and 6 selected counterfactual pixels BEFORE fitting.

Each arm resets the same raw weights, empty AdamW state, scaler and RNG, fits all
12 points for 128 updates and preserves the frozen behavior. Audits at 0/32/128
save numerical metrics and raw LoRA weights; terminal training.pt includes optimizer,
scaler and RNG. Only terminal128 gets full-generation evaluation. No teacher model
query or target recollection occurs. Save the complete GPU log and Slurm accounting.

After GPU completion, prepare the blinded review:

```bash
"$OPDLAB_PY" scripts/refresh_campaign.py --campaign experiments/local_expert_v15
"$OPDLAB_PY" scripts/review_annotation_probe.py prepare \
  --run runs/le15_annotation_001 --review runs/le15_review_001
```

Before reading key.json, endpoints scores, fitting metrics or per-arm results,
inspect all 38 case sheets / 152 native512px images using public items.json.
Only images with an exact archived SHA inherit existing labels from inherited.json;
these are explicitly not newly blinded observations. Write observations_draft.jsonl
with one row per case: case, views A/B/C/D, each with clauses, full_visibility,
visible_text and notes. Use explicit true/false/null, including small-print ambiguity.
Require identical semantic labels for identical pixels and requirements.

```bash
"$OPDLAB_PY" scripts/review_annotation_probe.py freeze \
  --run runs/le15_annotation_001 --review runs/le15_review_001
"$OPDLAB_PY" scripts/review_annotation_probe.py summarize \
  --run runs/le15_annotation_001 --review runs/le15_review_001
"$OPDLAB_PY" scripts/plot_annotation_probe.py
"$OPDLAB_PY" scripts/verify_annotation_probe.py
"$OPDLAB_PY" scripts/explain_annotation_probe.py
```

Analysis retains fitting-seed, positive fresh-seed, preservation fresh-seed and
unfitted development cohorts separately. Cluster both replicates by prompt for
2000 bootstrap resamples (seed0). Unknowns never become false-to-true repairs;
known harm can coexist with unknown labels. Report actual Slurm allocation plus
the inherited phase12 collection cost separately; annotation time is not completely
instrumented. No post-evaluation checkpoint, step, learning-rate or seed tuning.

Archive results, observations, hashes, counts, tests, plot and STATUS/NEXT. Keep
phase6–14 artifacts unchanged, resolving their README index entries to historical
commits (phase14 a23a429) after updating current navigation. Native on-policy OPD,
independent annotations, larger fresh cohorts and sampler transfer remain required
before a method claim, even if this small pilot passes its prespecified checks.

The explanation script adds an explicitly posthoc per-prompt failure report and
an illustration contrasting the fitting and fresh seeds of the same prompt. It
does not alter the labels, aggregate results or predeclared decision gate.

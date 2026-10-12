# Phase14: paired expert repair evidence development

Read experiments/local_expert_v14/PROTOCOL.md. Preparation and reports use the
existing diffusionopd Python; Qwen inference uses the exact independent executable
stored in run/site.json. Do not resolve its symlink or upgrade either environment.

```bash
export PYTHONPATH="$PWD/python"
export OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python
"$OPDLAB_PY" -m pytest -q
"$OPDLAB_PY" scripts/prepare_paired_audit.py --attempt 001
```

Preparation refuses an existing run. It checks all192 image hashes, freezes all128
real pairs, both orders,64 self controls, source inputs, templates, parser, selector,
Qwen assets/environment and request order. The model never sees the analysis key.
The committed phase6 audit worker is only reused for environment attestation;
its historical parsing behavior and archived responses are unchanged.

Submit the run's own run_paired_audit.py using its site Python and site.json:
scripts/submit_slurm.py --stage le14_paired --gres gpu:l40:1 --time 00:45:00,
ledger experiments/local_expert_v14/ledger.jsonl. The submission.json and job.sh
retain exact executable/arguments. One L40,4CPU,96GB. The worker verifies complete
model hashes and environment before requests; this may consume several minutes.

All192 target-free observations finish and are hashed before any two-image request.
Pairs run in frozen hash order and receive separate A/B image labels. Raw answers,
errors, token counts and actual image grids are retained. No retry or repair occurs.

```bash
"$OPDLAB_PY" scripts/refresh_campaign.py --campaign experiments/local_expert_v14
"$OPDLAB_PY" scripts/summarize_paired_audit.py --run runs/le14_paired_001
"$OPDLAB_PY" scripts/explain_paired_audit.py
"$OPDLAB_PY" scripts/summarize_paired_array_diagnostic.py
"$OPDLAB_PY" scripts/explain_paired_audit.py --array-diagnostic
"$OPDLAB_PY" scripts/plot_paired_audit.py
"$OPDLAB_PY" scripts/verify_paired_audit.py
```

Reporting replays every parse and request before joining unchanged phase12 review.
It reports all real pairs and self controls, not just selections. Pair-level bootstrap
ratios cluster both expert candidates within each of64 prompts; selected policies use
the same case ordering as phase12/13, reproducing historical finite-resample intervals.
Cost is Slurm allocation including imports, complete weight hashes and model loading.
Do not use the model-loop elapsed time as allocated GPU time.

The primary decision rule, mechanical ablations and gates were frozen before inference.
Failed gates leave Qwen as a diagnostic; no automatic training or prospective expansion.
Keep the old phase13 report_index.json immutable and use its archived README atbb804ca
when checking historical evidence after updating the current README navigation.

See ARRAY_DIAGNOSTIC.md for a separately named posthoc compatibility analysis. The
model emitted top-level clause arrays instead of the requested wrapper object. Keep
strict results.json and array_diagnostic.json separate. Only the outer container is
mapped; values and per-field validation remain unchanged. The diagnostic adds no GPU
requests and cannot retroactively make the frozen interface pass. Future runs require
a real two-image schema smoke before freezing their prospective audit contract.

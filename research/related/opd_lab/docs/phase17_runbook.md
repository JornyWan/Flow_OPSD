# Phase17 prospective time/strength coverage

From opd_lab set `OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python` and
`PYTHONPATH=python:scripts`. Historical phases are immutable. This phase collects
targets; it never fits a model. Worker, review/selection script, protocol and
historical seed inventory are attested before GPU submission.

```bash
"$OPDLAB_PY" -m pytest -q tests/test_coverage_probe.py tests/test_coverage_review.py
"$OPDLAB_PY" scripts/prepare_coverage_probe.py --attempt 001
"$OPDLAB_PY" scripts/submit_slurm.py --run runs/le17_coverage_001 \
  --stage le17_coverage --gres gpu:l40:1 --time 00:15:00 \
  --ledger experiments/local_expert_v17/ledger.jsonl -- \
  ../../conda_envs/envs/diffusionopd/bin/python -m opdlab.coverage_probe \
  --run runs/le17_coverage_001
"$OPDLAB_PY" scripts/refresh_campaign.py --campaign experiments/local_expert_v17
"$OPDLAB_PY" scripts/review_coverage_probe.py prepare
```

Run le17_coverage_001, Slurm1049670, singleL40 ai-gpu12. Seed inventory contains
156 historical files and263 excluded integers extracted under seed-named fields;
this conservative union can include RNG constants/counters, not only image seeds.
The12 new sampling seeds are unique and disjoint. Before collection, all12 old reference PNGs must match exactly; six
saved state/context/noop payloads match. New full trajectories retain11 states,
10 actions and sigma/context; query payloads at indices1,3,5 retain raw experts
and both interpolated strengths. No intermediate state is taken from a changed
trajectory. Check actual168 image/2280 CFG-example counters against frozen plan.

Review only public items.json and48 case-page sheets (13 views over four pages
per case). Each512px image is preserved at native size in the sheets. Baseline,
teacher, index, alpha, seed and automatic score are hidden. Write explicit
true/false/null for every clause and visibility, literal text and notes in
observations_draft.jsonl. Preserve duplicates and unknowns. Record caveats and
extra native-view inspections before joining key.json.

```bash
"$OPDLAB_PY" scripts/review_coverage_probe.py freeze
"$OPDLAB_PY" scripts/review_coverage_probe.py summarize
"$OPDLAB_PY" scripts/plot_coverage_probe.py
"$OPDLAB_PY" scripts/verify_coverage_probe.py
"$OPDLAB_PY" scripts/explain_coverage_probe.py
```

The six predeclared pools share all12 baselines. Report oracle safe availability,
unresolved cases, every candidate's harms, and per-prompt safe-state counts0/1/2.
Target ranking is frozen: determinate safe repairs only, most repaired clauses,
then least action-distance×dt², then candidate name. No safe candidate means noop.
target_manifest.json is untrained provisional assistant supervision, not output
from the failed automatic verifier and not a deployed policy. A larger pool uses
more teacher queries and suffix rollouts; report actual costs, no efficiency claim.

Verify source/seed/image hashes, CPU noise, every stored state update, all144 action
interpolations,36 query payloads,156 review copies and rebuilt target rankings.
Use1e-12 tolerance only for CPU/CUDA FP64 action-distance reductions; image/tensor
hashes stay exact. Preserve phase6–16 report indices, using README at historical
commits (phase16:198dc28) when current navigation changes. Archive complete results,
labels, costs, target provenance, tests, figures, STATUS/NEXT and SHA index.

Completed: job1049670,246 allocated seconds,0.068333GPU-hours, no failedGPU attempt;
102 full CPU tests passed. All12 reference images,12 trajectories,36 queries and
144 interpolations verified. All48 sheets and38 additional native image views
inspected;156 observations frozen before key join, including every baseline.

The post-join explain script partitions baseline headroom and inventories earlier
safe supervision without changing labels or the phase17 frozen target selection.
Mixed times provide11 states; fixed index3 provides9. These are development
inventories, not ready training runs: preservation points, controls and independent
evaluation still need their own phase18 frozen protocol. Analysis asserts source
query/action hashes, common prefix/suffix model and distinct(id,seed) identities.

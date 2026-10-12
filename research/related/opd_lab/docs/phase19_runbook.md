# Phase19 completed teacher-opportunity diagnostic

Working directory: opd_related/opd_lab. GPU1050350 completed in130allocated seconds.
All24 references and24 new images are complete; all48 observations frozen and statistics rebuilt.
Do not rerun prepare/freeze into existing directories or change labels after unblinding.

```bash
export OPDLAB_PY=../../conda_envs/envs/diffusionopd/bin/python
export PYTHONPATH=python:scripts
# Read-only scientific audit (writes verification.json):
"$OPDLAB_PY" scripts/verify_teacher_opportunity.py
```

Preparation and execution already performed with the snapshotted module:

```bash
"$OPDLAB_PY" -m opdlab.teacher_opportunity prepare --run runs/le19_opportunity_001
"$OPDLAB_PY" scripts/submit_slurm.py --run runs/le19_opportunity_001 \
  --stage le19_teacher_opportunity --gres gpu:l40:1 --time 00:15:00 \
  --ledger experiments/local_expert_v19/ledger.jsonl -- \
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /midtier/minilab/scratch/chw4026/projects/side_project/generative/conda_envs/envs/diffusionopd/bin/python \
  -m opdlab.teacher_opportunity run --run runs/le19_opportunity_001
"$OPDLAB_PY" scripts/review_teacher_opportunity.py prepare
# Explicit image observations were recorded, all before:
"$OPDLAB_PY" scripts/review_teacher_opportunity.py freeze
"$OPDLAB_PY" scripts/review_teacher_opportunity.py summarize
"$OPDLAB_PY" scripts/plot_teacher_opportunity.py
```

Do not refresh finalized accounting/ledger files unless making a new versioned audit: their
current exact bytes are in report_index.json. Historical phase18 index validation reads README
at bd585a7. Post-hoc scalar geometry uses all24 saved teacher comparisons and does not render
new images; it cannot establish endpoint quality after rescaling. The paired figure shows all
cases satisfying the predeclared determinate teacher-complete/student-incomplete condition.

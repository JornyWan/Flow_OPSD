# Phase34 revised evaluation workflow

The original RUNBOOK.md full600 assistant-review commands are superseded. Do not resume direct image inspection or run the old full-manual freeze/analyze/archive stages. User resource policy is in ../../AGENTS.md and EVALUATION_AMENDMENT.md.

Already completed:

- Anonymous cases1–70 /350 images frozen in runs/le34_review_001/partial_observations_frozen.json before method/mask join.
- assistant_partial_results.json contains70 pairs/140 queries only; no invented remaining observations or full60-cluster intervals.
- Tesseract600 predictions: runs/le34_secondary_001/result.json, same pinned existing executable/model/config on supported CPU host.
- Main automatic run: runs/le34_auto_eval_002, job1061861. Failed attempt001/job1061860 retained; it failed before any image scoring. No automatic retry/requeue.

After main automatic predictions freeze, from opd_lab:

```bash
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python - <<'PY'
import switch_amplitude_evaluation as s
s.AUTO=s.ROOT/'runs/le34_auto_eval_002'
s.analyze_auto()
PY
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/report_budgeted_amplitude.py
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/verify_budgeted_amplitude.py
FONTCONFIG_FILE="$PWD/.tools/report-browser/fonts/fonts.conf" .tools/report-browser/node_modules/node/bin/node scripts/check_budgeted_amplitude.cjs
```

The browser check visits all120 cases/600 bindings, decodes images at native size, checks labels and mobile width. It takes no screenshots and sends no images to the agent. Read JSON summaries and disagreement counts only. Do not tune rules on reference labels or merge assistant and automatic labels. Record scheduler GPU time for jobs1061860/1061861 separately from generation1061762.

For future evaluation: fixed local OCR/object scorers, all-image caching, explicit unknowns/errors, and a small independent human audit when needed. Agent image inspection defaults to zero; any concrete debugging exception must be bounded and explained. Automatically validating an HTML gallery does not require the agent to visually inspect it.

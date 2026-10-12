# Phase23 reproducibility

From `opd_related/opd_lab`, Python is `../../conda_envs/envs/diffusionopd/bin/python`; use `PYTHONPATH=python:scripts`. Sources are `python/opdlab/capability_probe.py`, `scripts/prepare_capability_probe.py`, `scripts/review_capability_probe.py`, `scripts/report_capability_probe.py`, and `tests/test_capability_probe.py`. Historical Phase22 sources and artifacts were not edited.

Protocol: `experiments/local_expert_v23/PROTOCOL.md`. Run: `runs/le23_capability_001`. Hidden review: `runs/le23_capability_review_001`. Attempt configuration, upstream/source snapshots, asset hashes, historical controls and trajectory references are saved inside the run. Do not overwrite a completed attempt.

The worker ran through `scripts/submit_slurm.py`, with the exact sbatch and worker arguments saved in the campaign ledger. Slurm1059027: COMPLETED0:0, one L40,139seconds,0.038611GPU-hour. The worker command was `env CUBLAS_WORKSPACE_CONFIG=:4096:8 <python> -m opdlab.capability_probe --run <absolute-run-path>`. No training updates occurred; `training=True` only causes the backend to load teacher adapters.

Related tests, run before allocation:

```bash
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python -m pytest -q tests/test_capability_probe.py tests/test_interval_probe.py tests/test_headroom_probe.py
```

12 passed. New tests cover the actual-state teacher switch, model/counter preservation and replay failure before any teacher query. The real run additionally checked all references and exact saved trajectories.

Independent post-run tensor verification (read-only run validation; writes campaign verification):

```bash
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_capability_probe.py verify --run runs/le23_capability_001
```

Review preparation, freezing and summarization have already completed. They deliberately refuse to overwrite frozen artifacts. All72 new observations were recorded against anonymous A/B/C images before the key join. The24 baseline and24 old local labels came directly from attested historical results. The inputs record the v2 rubric hash and the frozen observations identify the reviewer as the implementing assistant, not an independent human.

```bash
OMP_NUM_THREADS=4 PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_capability_probe.py prepare --run runs/le23_capability_001 --review runs/le23_capability_review_001
# Record all observations_draft.jsonl against images/items; do not read key.json.
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_capability_probe.py freeze --run runs/le23_capability_001 --review runs/le23_capability_review_001
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/review_capability_probe.py summarize --run runs/le23_capability_001 --review runs/le23_capability_review_001
PYTHONPATH=python:scripts ../../conda_envs/envs/diffusionopd/bin/python scripts/report_capability_probe.py
```

The report builder validates image hashes and pairing across all five methods, embeds the original PNG bytes, and writes provenance to `report_manifest.json`. The standalone28MB report needs no server or external assets. Browser verification uses the cached Node/Playwright/Chromium installation:

```bash
FONTCONFIG_FILE="$PWD/.tools/report-browser/fonts/fonts.conf" .tools/report-browser/node_modules/node/bin/node runs/capability_report_validation_001/check.cjs
```

Browser results are `experiments/local_expert_v23/browser_verification.json`; screenshots are in `runs/capability_report_validation_001`. Checks cover complete image decoding, arbitrary pairs, all five views, filters, modal/Escape, mobile overflow, JavaScript errors and external requests. Screenshots were visually inspected.

Final counts, interpretations and limitations: `STATUS.md` and `results.json` in the campaign. `completion_manifest.json` attests final artifacts; raw worker trajectories and image hashes are additionally attested in run/result.json and endpoints.jsonl. The next confirmation experiment is described in `NEXT.md` and has not been submitted.

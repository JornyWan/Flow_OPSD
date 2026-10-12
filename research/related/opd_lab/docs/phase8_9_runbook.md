# Phases8 and9: frozen-policy diagnostics

Phase8 executes the pre-existing phase7/NEXT plan. Its completed evidence caused
the separate phase9 protocol to be frozen before another GPU job. No training,
old-run edits, environment changes or upstream modifications were performed.

| Job | Run | Work | Allocated L40S time |
|---|---|---|---:|
|1048955|`runs/le8_trajectory_001`|seven execution masks ×five queries|94 seconds|
|1048968|`runs/le9_requery_001`|current baseline +four fresh teacher actions ×five queries|66 seconds|

Both use one GPU/four CPUs/96GB with a30-minute upper limit. Ledgers record exact
commands, source snapshot hashes and accounting. All jobs exited successfully.

```bash
export PYTHONPATH=python
../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_trajectory_probe.py --site site.local.json
../../conda_envs/envs/diffusionopd/bin/python scripts/prepare_state_requery.py --site site.local.json
```

The second command requires completed phase8 results/review/STATUS, which are
attested as inputs. Preparers refuse existing attempt directories. Use each run's
snapshotted module via recorded `job.sh`, not a mutable later working tree.
Workers are `opdlab.trajectory_probe` and `opdlab.state_requery`.

Phase8 records full state/action trajectories for all execution rules. Base,
local and full must exactly replay15 phase7 image hashes. Two index-zero queries
give four identical-rule checks. Query-state displacement measures drift, not
target MSE or a universal quality threshold.

Phase9 regenerates the current joint128 policy's states. Teacher blend is
relative to the CURRENT student prediction, and all candidates use that same
frozen CURRENT student suffix. GenEval CFG1 and OCR CFG4.5 are original teacher
settings. The old state's target vector is never transplanted. Five baseline
images match phase8; full branch sequences and teacher predictions are retained.

After a worker completes:

```bash
../../conda_envs/envs/diffusionopd/bin/python scripts/review_trajectory_probe.py prepare \
  --run runs/le8_trajectory_001 --review runs/le8_review_001
../../conda_envs/envs/diffusionopd/bin/python scripts/review_state_requery.py prepare \
  --run runs/le9_requery_001 --review runs/le9_review_001
```

Read only items/sheets, record every unique unseen image under the frozen rubric,
then save observations_frozen.json with hashes of observations/key/inherited
before opening the key. Replace `prepare` by `summarize` after freezing. Exact
pixel matches inherit prior frozen labels; duplicates are not independent data.
Phase8 has16 new unique images plus14 inherited mode rows; phase9 has20 new unique
images plus five inherited baselines. Reviews are by the implementing assistant,
not independent human labels. Unknowns remain unknown.

The first full test suite had62 passes; after phase9 and its fresh-prefix/current-
suffix and strict visual-protection tests,64 passed. Syntax/outputs/images and
state identities are verified before final indexing. Old phase7 report-index
README hash refers to revision8c1e160; old phase6 README refers to e384a09. Their
historical indexes are preserved when the current README navigation advances.

Only refresh these new output directories with os.utime for NFS visibility;
leave unrelated jobs and old input mtimes alone. Phase10 is a planning decision
in phase9/NEXT.md, not a pending background job.

Post-review illustration and artifact verification:

```bash
../../conda_envs/envs/diffusionopd/bin/python scripts/plot_state_requery.py
../../conda_envs/envs/diffusionopd/bin/python scripts/verify_state_probes.py
```

The figure is made after unblinding, from unchanged source images and frozen
labels; its source list and hashes are retained. In phase8, "retained repairs"
refers to the encoded clauses. In particular, apple with old prefix/new suffix
retains RIGHT and CLOUD but crops the bowl; the figure explicitly shows that
visibility failure. Phase9's accepted apple GenEval1 repair preserves all three
reviewed attributes. Neither statistic exhausts every aesthetic or prompt detail.

The verification script checks the full current-baseline state/action replay
between phases8 and9 on CPU, each candidate prefix and proposed action, all60
image hashes, frozen reviews/inputs, unchanged historical indexed artifacts and
both clean upstream commits. Current-student continuation provenance comes from
the snapshotted worker plus the behavioral unit test; it is not a second GPU
re-execution. Two reporting-script setup errors (figure visibility assumption,
phase8 rubric path) were fixed without changing any labels or experimental output.

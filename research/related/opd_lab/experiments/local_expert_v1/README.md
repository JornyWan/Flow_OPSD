# Local expert v1 experiment ledger

User authorized implementation and experiment execution on 2026-09-10.

Every attempt has a distinct run directory and Slurm job ID. Source/config/data
snapshots are frozen before submission. Failed attempts remain in the ledger;
they are not overwritten or included in successful comparisons.

Sequence: CUDA/dependency probe → four-prompt GPU smoke → frozen intervention
diagnostic → independent image review → controlled training only if justified.
The 64-prompt pilot is a constructed development set, not an established benchmark.

`ledger.jsonl` records submitted jobs and subsequent results. `STATUS.md`
records validated findings, limitations, and the next decision.

Run `python scripts/refresh_campaign.py` from the lab root to refresh
`slurm_status.json` and append completed-job accounting, including failed runs.
This command never submits/cancels jobs and never treats Slurm success as
scientific validation. `scripts/submit_slurm.py` freezes each job script, records
submission intent/ID, and allocates a job-specific local temporary directory.

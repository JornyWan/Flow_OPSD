# Codex OPD research log

This directory is the append-only handoff area for Codex work on the OPD project.
It is intentionally separate from the existing Claude notes, upstream paper sources,
training code, and result directories.

## Ordering rule

Every research action receives a four-digit sequence number. The sequence is assigned
before an experiment is submitted and is never reused.

- `NNNN_*.md`: question, reasoning, preregistered criteria, result, or correction.
- `code/NNNN_*`: code or a patch used by that action.
- `slurm/NNNN_*`: the exact submission script used by that action.
- `runs/NNNN_*/`: persistent logs, run manifests, and checkpoints.
- `INDEX.md`: chronological index and current status.

Corrections do not rewrite history. They get a new sequence number and point back to
the superseded entry. A result is not marked valid until its runtime invariants and
comparison protocol have been checked.

## Reproducibility rule

The existing DiffusionOPD and DiffusionOPSD trees are treated as mutable working
directories, not pristine upstreams. New Codex jobs use a recorded source SHA-256 and,
where code changes are needed, a stored patch applied to a temporary job-local source
copy. Existing trainers are not silently edited underneath running jobs.

Each submitted run must record:

1. research question and single-variable comparison;
2. source file hashes and patch hash;
3. full environment variables and `sbatch` command;
4. SLURM job ID and output path;
5. smoke-test invariants;
6. preregistered success/failure criteria;
7. final status: `planned`, `submitted`, `running`, `valid`, `invalid`, `confounded`,
   or `superseded`.


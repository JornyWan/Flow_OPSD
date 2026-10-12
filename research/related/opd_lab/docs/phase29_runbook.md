# Phase29 refreshed-behavior cycle

The immutable predecessor is Phase28 completion manifest `779cd4e0ec69425d5a42462bfa8ae97c04b94f9e97812851c5bb73e507727f9f`. Its engineering gate passed; verified21/36 versus initial13/36 and matched random19/36, with an inconclusive verified-minus-random interval. This authorizes the predeclared refreshed-behavior cycle, not the failed-gate local-transfer diagnostic.

Work from `opd_lab`, using `OMP_NUM_THREADS=4 PYTHONPATH=python:scripts` and `../../conda_envs/envs/diffusionopd/bin/python`. Existing run directories are immutable attempts; do not rerun commands into them.

## What changes between cycles

All arms start from the exact Phase28 verified checkpoint S1. S1 is also the new frozen behavior. All156 active pair seeds change, while60 train and18 development prompts stay the same. Reserved final prompts stay ungenerated. External teachers, verifier, CFG, flow10 schedule, two teacher-action positions, loss, eight AdamW updates and mask construction stay fixed. Fitting directly reuses `opdlab.verified_learning.fit`; there is no separate reimplementation of the loss or reset logic.

The legacy storage key `full_geneval` now refers to S1. Report and human-review displays identify the starting model correctly. Results measure incremental improvement from a common S1, not three independent two-cycle learning histories or a fresh prompt cohort. Do not pool Phase28/29 observations as independent replicates.

## Actual sequence and accounting correction

1. `scripts/prepare_refreshed_learning.py dataset` freezes the same prompt records, a complete historical seed inventory and156 new seeds.
2. Collection attempt001, job1059738, passed all four GPU replays. A CPU audit then identified an incorrect inherited text-encoding counter expectation: the historical replay prompt remains in the64-entry FIFO cache when it appears in the development sequence. Expected encodings are156, not158. The job was manually cancelled before the incorrect assertion could fail, at141 allocated seconds (0.03916667 GPU-hours). No images were visually inspected and no outcome informed the fix. Its frozen framework, partial outputs and logs remain intact.
3. `counter_correction.json`, the original protocol/source copies and revised analysis attestations record the exact administrative correction. Attempt002, job1059745, has the same model, data and numerical algorithm. All82 retained attempt001 endpoint hashes exactly matched the corresponding attempt002 outputs. Old attempt001's RUNNING file is a stale process marker from cancellation; Slurm state and ABORTED.json identify it as stopped.
4. Successful collection is `runs/le29_collect_002`. Verify all hashes and `selection.json`. Fitting eligibility requires6 training accepts across2 families. If eligibility fails, stop before optimization and report that result.
5. `scripts/prepare_refreshed_learning.py fit` binds attempt002 explicitly and prepares `runs/le29_fit_001`. Submit `python -m opdlab.refreshed_learning --run <absolute run>` through `scripts/submit_slurm.py`, one L40,45-minute cap, no automatic requeue. Each arm resets S1, optimizer, scaler and RNG. No development image review before all outputs finish.
6. `scripts/review_refreshed_learning.py prepare` verifies all masks, counters, initialization hashes, paired noise and4200 saved Euler transitions, then prepares36 anonymous five-view cases. Freeze the complete initial draft and record any pre-freeze revisions; do not read the key or join automatic development evidence before all180 labels freeze.
7. Run `scripts/review_refreshed_learning.py freeze`, then `scripts/summarize_refreshed_learning.py`. Preserve every result, including unknowns and regression. The same numerical engineering checks are descriptive; they do not automatically launch a third cycle.
8. Build `scripts/report_refreshed_learning.py` and `scripts/plot_refreshed_learning.py`. Browser verification uses the bundled Node on `scripts/check_refreshed_learning_report.cjs`, with `FONTCONFIG_FILE` set to `.tools/report-browser/fonts/fonts.conf`.
9. Refresh final Slurm accounting, including the cancelled attempt, and run `scripts/verify_refreshed_learning.py --write`. Its separate reconstruction checks scores, all14 paired intervals, embedded image hashes and the engineering gate. Finalize status/interpretation/NEXT and archive the campaign. Ongoing session monitoring remains outside its manifest.

## Human observations

`scripts/prepare_learning_human_review.py --phase 28` creates all36 cases with initial/random/verified images, each case independently shuffled, with no existing label shown. The standalone HTML includes108 original PNGs and exports partial answers safely. The same generator supports Phase29 only after its archive completes. All-case inclusion avoids selecting only visually favorable examples.

`scripts/check_learning_human_review.cjs 28` checks offline image decoding, blank controls, navigation, local persistence and export. Its synthetic browser export is explicitly not human feedback and is never placed in `human_rating` or analyzed as evidence.

When actual answers arrive, `scripts/analyze_learning_human_review.py --phase 28 --answers <file> --output <new directory>` preserves the exact input and the old labels. Missing fields stay missing; uncertain fields stay uncertain. A partial response subset yields descriptive common-case counts only. Paired intervals require complete categorical responses for all36 cases. Preferences remain verbatim and separate from constraint accuracy.

## Monitoring

Session deadline remains2026-09-21T08:49:00Z, with at most3 allocated GPU-hours and one concurrent L40. The read-only monitor includes both campaigns and cancelled attempts. Do not touch unrelated user jobs. No automatic third training cycle is part of this session.

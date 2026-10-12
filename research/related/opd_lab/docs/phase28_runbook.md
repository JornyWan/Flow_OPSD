# Phase28 controlled learning runbook

The frozen protocol is `experiments/local_expert_v28/PROTOCOL.md`. This runbook explains execution and provenance; it does not override the protocol. Work from the `opd_lab` directory with `PYTHONPATH=python:scripts`, `OMP_NUM_THREADS=4`, and `../../conda_envs/envs/diffusionopd/bin/python`.

## What is being tested

An external OCR expert provides two actions at actual branch states. The same frozen initial student finishes both counterfactual trajectories. A terminal verifier selects a subset of expert targets. Three students start at exactly the same GenEval LoRA and fit all targets, a family-density-matched random subset, or the verified subset. Their full noise-to-image outputs are compared on development prompts after every model is frozen.

This is external-expert distillation. The initial student is the existing GenEval LoRA. It is neither a bare backbone baseline nor a native-upstream OPD training reproduction. The all-target arm uses the same local two-action objective as the other arms. A native DiffusionOPD baseline should remain a separately named and separately costed experiment.

## Frozen sequence

1. `scripts/prepare_verified_learning.py dataset` freezes96 disjoint target words and192 paired seeds; only train and development are active in this study. Preserve the provenance inventory and reserved test.
2. `scripts/prepare_verified_learning.py collect` freezes `runs/le28_collect_001`, including a complete framework/upstream snapshot, environment, assets, protocol and replay inputs.
3. Submit `python -m opdlab.verified_learning --run <absolute collection run>` through `scripts/submit_slurm.py`. Use one L40 and30min. Disable Slurm requeue and record that action. The job must exactly replay four historical trajectories and one scorer comparison before fresh evidence.
4. Wait for successful completion and validate output hashes. Training eligibility requires at least6 accepted training pairs in at least2 families. Only inspect training selection for this decision; do not review development images or use development evidence to alter training.
5. `scripts/prepare_verified_learning.py fit` freezes `runs/le28_fit_001`. Submit the same module with the fit run, one L40 and75min. Every arm resets weights, AdamW state, scaler and RNG. Do not resume from a partially fitted arm or reuse an optimizer across arms.
6. Wait for all three students and108 development outputs to finish. `scripts/review_verified_learning.py prepare` checks tensors, counters, paired noise and hashes, then prepares36 anonymous five-view cases in `runs/le28_review_001`. It also freezes the analysis script hash before observation.
7. Read the unchanged rubric. Observe all180 images without reading `key.json` or joining automatic development evidence. Write `observations_draft.jsonl` with one row per case and views A–E. Each view contains `clauses`, `full_visibility`, `visible_text`, and `notes`; unknown values are JSON null. Preserve the initial draft and any pre-freeze revisions explicitly.
8. `scripts/review_verified_learning.py freeze` validates every observation and image, then writes the observation freeze record. Only now may `scripts/summarize_verified_learning.py` join method identities and evidence.
9. `scripts/report_verified_learning.py` writes a standalone HTML with all180 PNGs embedded. Use the bundled Node executable on `scripts/check_verified_learning_report.cjs`; set `FONTCONFIG_FILE` to `.tools/report-browser/fonts/fonts.conf`. The browser checks offline decoding, native image size, case identities, filters, pagination and mobile layout.
10. Refresh Slurm accounting after all jobs complete, run `scripts/verify_verified_learning.py --write`, write final status/interpretation/next-step records, and freeze a completion manifest. Do not refresh or mutate an archived campaign afterward.

## Runtime facts to verify

Collection:156 pairs,312 new endpoints, four historical replay images. Counter expectations are specified in the protocol. The scorer also replays two old endpoints; all scoring costs remain counted.

Fitting:240 training slots per update. The observed frozen selection has24 accepted pairs, with random matching by family. Thus all has1920 backward queries and each masked arm has384. All three arms have8 optimizer updates. Action metrics add960 inference predictions; independent full generation adds1080. Total fit worker student forward queries4728, backward queries2688, decoded images108, text encodings36, no teacher or behavior forward query. Model setup and weight copying do not count as prediction calls.

Exact initialization equality is checked on a real GPU. CPU tests cover loss normalization, masked-target independence, state requery, family matching, optimizer reset and split seeds. A separate analysis implementation reconstructs the final counts and all14 reported paired intervals.

## Evidence-driven next step

The engineering gate is numerical and predeclared, not a statistical success test. A pass requires a separately frozen refreshed-behavior/new-seed cycle. A failure enables `scripts/prepare_local_transfer_cycle.py` only after Phase28 completion is archived. That diagnostic runs `opdlab.local_transfer_cycle`: two sequential learned-student actions from the same saved old prefix, then the frozen initial behavior suffix. It performs no optimization and does not change inherited observations. Its own protocol, result, review and report must remain in Phase29.

## Three-hour session monitoring

The user requested active monitoring from2026-09-21T05:49:00Z to08:49:00Z. `scripts/monitor_learning_session.py` writes read-only snapshots into `runs/le28_monitor_session_001` every30s and discovers Phase28/29 jobs from their ledgers. It neither submits jobs nor edits campaign records. Keep its changing files outside completed campaign manifests, then archive a session manifest at the deadline. All new jobs together may allocate at most3 GPU-hours, with one concurrent L40. Unrelated user jobs are outside this experiment.

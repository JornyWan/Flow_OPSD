# How the "reward/score vs training step" curves are made — and what is actually recoverable

## TL;DR
- DiffusionOPD's training loop **does compute and log** a per-epoch proxy-reward
  trajectory (`monitor_pickscore` / `monitor_ocr` / `monitor_geneval`, plus
  `eval_*monitor_*` on periodic held-out passes), exactly where you'd expect
  (`scripts/train_sd3_opd.py`) — **but it only ever reaches Weights & Biases**,
  and W&B is run in **`WANDB_MODE=offline`**. The offline run directory lives on
  the ephemeral local disk of the training pod (relative `./wandb/…`, under a
  fresh-per-job working directory), is **never synced to S3, `/logging-fs`, or
  `/home`**, and is gone once the pod recycles.
- The persisted, human-readable training logs at
  `/logging-fs/logs/pluto/jobs/cwan/<uniqueName>/run-*/pod-0*/main_scripts/1/current`
  contain **zero numeric reward/monitor values** — verified by exhaustive grep
  (see "Evidence" below) across `mopd_caopd`, the uniform `mopd` baseline,
  `mt_nft`, `mt_grpoguard`, `mopd_robust`, `mopd_snr`, `mopd_anneal`. They only
  contain: the one-time YAML config dump, tqdm progress-bar text (which shows
  epoch index and it/s, never postfix metrics), and library/model-loading noise.
- **So: the paper's "reward vs training step" curve is *not* literally an
  auto-saved, continuously-logged training-time series that can be pulled off
  disk.** It has to be *reconstructed* from the periodic, checkpoint-level held-out
  **eval pipeline** (`s3://dit-scale-up/cwan/opd_reproduction/eval/<name>_<dataset>/average_scores.json`),
  which is run manually/separately after training and is the true source of every
  number in `CONTEXT.md`'s results tables. This is coarser (one point per saved
  LoRA checkpoint, i.e. every 30 steps) but it is real, persisted, and correct.

## Evidence (why we're confident it's not recoverable from logs)
Checked `scripts/train_sd3_opd.py` (the OPD/CA-OPD trainer) and
`scripts/baselines/train_nft_sd3_multireward.py` / `train_sd3_GRPO_Guard_multireward.py`:
- `train_sd3_opd.py:1139-1146`: `monitor_log = {"epoch": epoch, "monitor_<key>": ...}` →
  `wandb.log(monitor_log, step=global_step)` — **wandb only, no `logger.info`/`print` companion.**
- `train_sd3_opd.py:403-414`: the periodic `evaluate()` pass logs
  `eval_{prefix}monitor_{key}` the same way — **wandb only.**
- Baseline scripts: identical pattern (`reward_{k}`, `mean_reward_*`,
  `reward_std_mean`, `zero_std_ratio` all go through `wandb.log` only).
- Confirmed offline mode in every job's log:
  `+ export WANDB_MODE=offline` / `wandb: W&B syncing is set to \`offline\``.
- Grepped the **full** `current` log of every run below for `monitor_`, `reward`,
  `Chebyshev`/`[CA-OPD]`, `[save_ckpt]`, `Saved checkpoint`, `loss=`, and any
  float following `avg`/`reward` — all zero hits for actual numeric values (the
  only "reward"-adjacent text is a debug `print(shape)` of tensor shapes, and
  tqdm captions used only inside `wandb.Image(...)`, which also never reaches
  the text log). `logger.info` calls *do* work in these logs (e.g. the one-time
  config dump, and 8-process "***** Running training *****" banners for
  multi-GPU jobs) — so this isn't a broken logger, it's simply that no code
  path prints numeric rewards outside of `wandb.log`.
- `s3://dit-scale-up/cwan/opd_reproduction/runs/<name>/logs/.../` contains
  **only `checkpoints/`** — no synced wandb files, no metrics json.
- `/mnt/localssd/opd/` (where `compare_ideas.py` and the `venv_mopd` env used to
  live, per project memory) is **gone in this session** — only `.cache/`
  remains under `/mnt/localssd/`. The local SSD is ephemeral per-node/session;
  `compare_ideas.py`'s exact normalization logic could not be re-read and had
  to be reconstructed (see below).

## What we did instead: reconstruct from checkpoint evals
For each run, we already have (a) exactly which LoRA checkpoints were saved
(from S3 `checkpoints/checkpoint-<step>/lora/adapter_model.safetensors`, saved
every 30 training steps for every `mopd*` run) and (b) their real S3
`LastModified` timestamp (used as ground-truth wall-clock time — far more
reliable than trying to re-derive it from the "Epoch N" tqdm counter, whose
relationship to `global_step` differs across scripts/restarts and is not worth
the epistemic risk; see `data/*.csv`'s `wall_clock_utc`/`elapsed_hours`
columns). Where a checkpoint was also run through the held-out eval pipeline
(`eval/<name>_<dataset>/average_scores.json` for `drawbench`/`geneval`/`ocr`),
we join in:
- `geneval`, `ocr` — the two rule-based accuracy metrics (already 0-1, directly
  comparable to `CONTEXT.md`'s numbers).
- `pickscore`, `clipscore`, `aesthetic`, `hpsv2` — from the `drawbench` eval
  (note: the `pickscore` field in `average_scores.json` is on a ~0.84-0.93
  scale, not `CONTEXT.md`'s ~20-24 "Pick" display scale; empirically
  `Pick ≈ pickscore × 26`, but we kept the raw field for internal consistency
  since only the *relative* min-max mapping matters for our figures).
- `reward_proxy_pickscore_ocr_geneval` = mean(pickscore, ocr, geneval) — this
  is the direct analogue of what `monitor_pickscore`/`monitor_ocr`/`monitor_geneval`
  would have shown live during training, since those three are exactly
  DiffusionOPD's three teacher-reward slots (PickScore≈AesTeacher,
  OCRTeacher, GenEvalTeacher). Used for `figures/convergence_reward.pdf`.
- `normalized_score_paper_anchored` = a **reconstructed** paper-anchored
  min-max score, averaged over all 6 metrics: floor = zero-shot SD3.5-Medium
  (+CFG, gs=4.5, `eval/base_gs4.5_*`), ceiling = the best value observed per
  metric among `{AesTeacher, GenEvalTeacher, OCRTeacher, curve_721, caopd721,
  caopd781}`. This is **our own reconstruction** (not a byte-for-byte replay
  of the now-missing `compare_ideas.py`) and its absolute values differ
  somewhat from `CONTEXT.md`'s original table, though the ranking/direction
  matches (CA-OPD > mopd ≈ robust ≫ SNR > anneal at ckpt-541).
  **Where a more authoritative number was available** — `data/RESULTS_at721.md`,
  which captures a same-scale @721/@781 comparison from an earlier
  `compare_ideas.py` run (before `/mnt/localssd/opd` was wiped this session) —
  **we overrode our reconstruction with those exact numbers** (see
  `score_source` column in each CSV: `authoritative_RESULTS_at721` vs
  `reconstructed_minmax`). That file's honest verdict — CA-OPD converges to
  the lagging (aesthetic) axis ~180 steps faster, but uniform-mopd's final
  aggregate ceiling at ckpt-721 (0.985) is slightly higher than CA-OPD's early
  plateau (~0.974–0.978) — is what the figures now show.

## Gaps (be aware of these when reading the CSVs/figures)
1. **mt_nft / mt_grpoguard have no eval scores at all.** Their training
   checkpoints exist in S3 (`mt_nft` up to step 570, `mt_grpoguard` up to step
   780 as of this session, both still training), but per `CONTEXT.md` these
   baselines still need a full eval pass. We did **not** run GPU inference to
   backfill this (out of scope / would require loading SD3.5 + LoRA + reward
   models here). Their CSVs contain only `step` / `wall_clock_utc` /
   `elapsed_hours`, all metric columns empty.
2. `mopd301` (uniform-mopd checkpoint-301) has no eval folder in S3 at all —
   only `caopd301_geneval` exists (missing `caopd301_drawbench`/`caopd301_ocr`
   too), so CA-OPD's normalized score at step 301 could not be computed; that
   row's `normalized_score_paper_anchored` is empty. `sopd_ckpt301_drawbench`
   exists but has no matching `_geneval`/`_ocr`, same issue.
3. `robust`/`snr`/`anneal` eval at step 301 (requested in the task) does not
   exist in S3 either — only 541 and 721 are evaluated for those three arms.
4. Checkpoint→epoch mapping is **not** 1:1 across scripts (e.g. `mopd_caopd`'s
   `global_step ≈ epoch+1`, but `mt_grpoguard`'s checkpoint numbering runs
   ~1.9× faster than its outer "Epoch N" tqdm counter, likely multiple inner
   epochs per outer epoch). We therefore report **wall-clock time from S3
   object metadata**, not an epoch-derived time, to avoid propagating that
   ambiguity into the CSVs.

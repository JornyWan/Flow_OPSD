# 0004 — Constrained-retention smoke result

Date: 2026-08-28  
Status: valid control-flow smoke; not a performance result

## Run identity

- valid job: `1035891` (`COMPLETED`, exit `0:0`, elapsed `00:04:34`)
- invalid predecessor: `1035889` (recorded in 0002/0003)
- node/resources: `ai-gpu14`, 2 x NVIDIA H100 NVL
- output: `runs/0003_cr_smoke_retry/slurm_1035891.out`
- metrics: `runs/0003_cr_smoke_retry/metrics.tsv`
- artifacts: `runs/0003_cr_smoke_retry/artifacts/job_1035891/`
- offline W&B run: `offline-run-20260828_224543-8bc8u9mr`
- SLURM-output SHA256: `14c9620626d155679d286aa9116cb5471fbe4d69b255bf519e1b118bfc829f72`
- W&B-record SHA256: `13f0dc0bba0bc92ae9096b6e5b46190addd52e5a75178cdbe6aa2ae66c8b6684`

## Acceptance results

| Invariant | Evidence | Result |
|---|---|---|
| Audited shared source | both base SHA256 values matched before overlay | pass |
| Isolated implementation | 0002 then 0003 patches applied to job-local temporary copy | pass |
| Official Student initialization | resolved `train.lora_path` points to released Student LoRA | pass |
| Correct task routing | log shows PickScore/Aes, OCR, and GenEval slots; masks are unit-tested | pass |
| Same-state frozen reference | all 12 epoch-0 batches logged `max_rel=0.000000e+00` | pass |
| Direct branch differentiates | direct parameter-gradient norm is finite/nonzero in all epochs | pass |
| Retention hinge differentiates when violated | epoch 1 active fraction `0.679784`, retention grad norm `2.2683e-4` | pass |
| Feasible initialization | epoch 0 current/reference KL are identical; hinge and retention grad are zero | pass |
| No legacy RTC short-circuit | global step reached 3 and checkpoints were written each epoch | pass |
| Finite optimizer state | all final raw tensors finite; Adam state exists for 382 tensors | pass |
| Clean termination | SLURM and launcher both exited zero | pass |

## Observed dynamics

| Epoch | Direct loss | Direct grad norm | Current KL | Reference KL | Active fraction | Retention loss | Retention grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.290501 | 0.033055 | 0.00168186 | 0.00168186 | 0.000000 | 0 | 0 |
| 1 | 0.271082 | 0.023786 | 0.00178346 | 0.00162476 | 0.679784 | 1.44717e-4 | 2.26830e-4 |
| 2 | 0.254544 | 0.017627 | 0.00097902 | 0.00146372 | 0.002315 | 1.50507e-8 | 0 |

The qualitative sequence is the intended one: the released Student starts feasible;
after a direct-reward update the OCR/GenEval constraint activates and supplies a
nonzero correcting gradient; the next epoch is almost entirely back inside the
allowed region. This is evidence that the two-source optimization plumbing is real,
not that image quality improved.

The branch-gradient probe is one local batch per rank and epoch. In particular, the
epoch-2 retention probe is zero because its first probed retention batch was feasible,
while the epoch-wide active fraction still contains `0.23%` active examples. Treat
these norms as scale diagnostics, not unbiased epoch averages.

## Checkpoint audit and one important pitfall

- checkpoints `checkpoint-1`, `checkpoint-2`, and `checkpoint-3` exist;
- final `train_state.pt` reports `epoch=2`, `global_step=3`;
- Adam reports step 3 for every state entry, with nonzero first moments in 380/382
  tensors (`||exp_avg|| = 2.26493e-4`), proving raw parameters were updated;
- all three exported LoRA safetensors are exactly equal to the initial Student.

The last point is expected from the existing trainer: exported LoRA weights are EMA
weights and `EMAModuleWrapper` updates only every 8 steps. A three-step smoke therefore
cannot be evaluated through its exported LoRA. The updated raw parameters are present
in the final train state, which is enough for this control-flow test. Any calibration
or performance run must last at least 8 optimizer steps (preferably much longer), or
explicitly export raw LoRA weights.

## What this changes about the next step

Do not launch a long four-arm performance sweep from this smoke alone. The most useful
next experiment is a short, at-least-20-step calibration run that adds epoch-averaged
branch gradient measurements and tests whether a fixed retention weight of 1 is strong
enough. In epoch 1, the sampled direct branch gradient norm is about 105 times the
retention branch norm. A primal-dual multiplier or explicit gradient-scale calibration
is more principled than guessing a larger fixed coefficient.

After that calibration passes, run the preregistered four arms from the released
Student with matched prompts/seeds/budget:

1. no update;
2. direct PickScore only;
3. direct PickScore plus fixed-weight retention;
4. direct PickScore plus calibrated/adaptive constraint.

Offline evaluation must report PickScore, OCR, and GenEval for every checkpoint. A
method win requires PickScore improvement without a statistically credible drop on
either constrained axis; a weighted average is not sufficient.

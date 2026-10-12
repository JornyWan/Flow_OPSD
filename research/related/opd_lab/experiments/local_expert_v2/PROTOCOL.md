# Phase 2: controlled screening of verified external-expert distillation

Frozen before this phase's training or validation results, 2026-09-11.
This is a development screening experiment, not a final benchmark or paper reproduction.

## Question and causal comparisons

The hypothesis is that an external expert's action can improve one requirement while
damaging another, and that terminal verification can identify supervision worth fitting.
Three separate claims require separate evidence: useful local actions exist; a student
can fit them; the updated student improves normal complete sampling on unseen prompts.
The previous phase established proxy-level local opportunity and a fitting-step issue,
not the third claim. We continue the external-teacher OPD direction.

| Arm | Proposal selected for fitting | Purpose |
|---|---|---|
| base | No training | Starting performance under each evaluation protocol |
| fixed_geneval | Always GenEval alpha=1 | Strong fixed-action control chosen using phase-1 training/development diagnostics |
| verified_fixed | Accept that same GenEval alpha=1 only if terminal utility improves and satisfied clauses survive | Isolates filtering from proposal search |
| verified_multi | Choose among OCR/GenEval alpha=0.5/1 with improvement and clause protection | Tests whether extra expert choice helps |
| utility_only | Same four proposals, highest improved utility, no satisfied-clause protection | Isolates the contribution and cost of protection |

All arms compute the same five candidate continuations for mechanism diagnostics.
Consequently this comparison matches diagnostic queries, not the optimal runtime of
an implementation that never needs to evaluate rejected or unused proposals. Native
OPD remains the independent algorithm baseline; its different data/routing must not
be silently called the causal control for this selector. Native full-size reproduction
and compute-optimized controls follow a positive screen.

## Frozen training recipe

- Start from the same SD3.5-M backbone and new rank-32/alpha-64 LoRA; never the phase-1
  two-update checkpoints. External teacher files, CFG and source commits are unchanged.
- Same 64 train prompts, 512 pixels, fp16, deterministic flow-10, behavior CFG 4.5;
  OCR teacher CFG 4.5 and GenEval teacher CFG 1. A guidance-only control is required in
  the confirmation stage before attributing all GenEval gains to expert training.
- Training seed 42; 12 outer epochs, 16 queries per outer epoch, accumulation 16,
  one fitting pass. This is 192 queries (three full data passes) and at most 12 updates
  per arm. Each batch contains four prompts from each of the four task groups.
- `training_queries.json` freezes prompt identity, initial noise and query index.
  Each pass shuffles within groups, interleaves groups, and covers every prompt once.
  Time indices retain the existing uniform hash sampling over 0..8; they are not
  restricted to the favorable early positions observed in phase 1.
- All arms share this query schedule, initialization and optimizer. On-policy states
  are expected to diverge after updates; do not require different trained policies to
  visit identical later states. Initial query/candidate equality is audited separately.
- AdamW learning rate 3e-5, selected from the prior training-only fitting check;
  other optimizer settings and transition-mean OPD loss unchanged. Rejected examples
  contribute zero and the denominator remains all queries. All-rejected batches skip
  AdamW. Log actual updates; equal outer epochs are not equal accepted supervision.
- EMA interval 1 for all arms, checkpoints every 4 epochs, per-query post-fit audits
  and candidate images retained. Primary evaluation uses final epoch-12 **raw** LoRA,
  preselected before results. No picking a better checkpoint or averaging convention.

## Verification and measurement

Keep `joint_constraints_v1` as the common training and primary evaluation metric for
this controlled screen. Changing both the selector and reward would break attribution.
Known OCR errors remain a material limitation: “valid” means the scorer ran, not that
its answer is correct. This phase is allowed to produce a negative/inconclusive result.

Before interpreting training outcomes, add an audit-only Tesseract reader, isolated
from training and target selection. Freeze English LSTM, full-image sparse-text mode
11, original PNG, no target-word whitelist and no prompt supplied to the reader.
It is a second OCR proxy, not ground truth. Replacing only the OCR clause leaves the
GenEval detector shared, so the resulting secondary joint metric is only partially
independent. Preserve both sets of transcripts, disagreement cases and image hashes.
No retrospective edits to phase-1 scores or training labels.

Audit a deterministic sample of 32 unique training-prompt candidate images (eight per
task group), with image identity/target/teacher/scores hidden during assistant
transcription. Record uncertain readings and extra text instead of forcing a label.
This is an implementing-assistant audit, not an independent human study. Evaluate both
readers against those provisional readings; never treat reader agreement as proof.
The second reader's CLI/model provenance follows the [official documentation](https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html).

## Evaluation pipeline

1. Verify all training artifacts, query schedule and checkpoint identities.
2. In one common frozen evaluation context, generate base and four final raw adapters
   on all 32 validation prompts with seeds 42 and 43. The 64 test prompts stay unused.
3. Predeclared screening cells: flow-10, flow-40, DPM2-40. Total: 5 weights × 3 cells
   × 32 prompts × 2 seeds = 960 images. Initial CPU-FP32 noise is keyed by prompt/seed;
   every within-cell comparison uses the same GPU model, settings and software.
4. Report paired utility and joint-success deltas, OCR/count/color/relation changes,
   and losses of previously satisfied clauses. Average seeds within prompt before
   bootstrapping 32 prompts; do not count states or seeds as independent prompts.
5. Primary mechanism contrasts: verified_fixed − fixed_geneval;
   verified_multi − verified_fixed; verified_multi − utility_only. Also compare each
   trained arm against base in each cell. Intervals are exploratory, not corrected for
   multiple comparisons, and one training seed cannot establish reproducibility.
6. Score the same saved PNGs using the audit-only reader. Report OCR and secondary
   joint deltas and disagreements; no new sampling or best-of-N selection in evaluation.

## Decisions after the screen

- Local target utility alone can never pass the screen. If local fit improves but
  normal validation sampling does not, inspect post-fit local/full differences and
  drift on rejected states; do not merely increase training length.
- If an apparent gain reverses with the audit-only OCR, classify it as verifier-
  dependent and prioritize measurement repair before a main efficacy experiment.
- If only flow-10 improves, classify it as grid-dependent. Investigate multi-grid
  continuation supervision in a separate controlled experiment; the current method
  offers no sampler-invariant guarantee.
- If the fixed expert matches the verified variants, do not claim selection is
  necessary. Add the best fixed early-time control and random acceptance matched on
  training calibration data before claiming an advantage over simpler policies.
- A promising result must survive larger data, multiple independent training seeds,
  full flow/DPM2 10/20/40 evaluation, independent image adjudication and a locked test.
  Efficiency then requires reward/quality versus measured GPU-hours and native OPD
  controls, including the cost of experts and candidate verification.

## Records and budget

Each attempt has a new run directory with frozen source/config/data and a Slurm job ID.
All failures count toward the ledger. Planned training jobs request one L40-class GPU
and at most 90 minutes each; the unified evaluation requests at most two hours.
The expected screen is a few GPU-hours; allocation limits are not claimed actual cost.
Do not start 8-GPU full baseline training, additional seeds, or a longer training sweep
automatically before examining this screen. Changes require a new named phase/config,
not overwriting this protocol, results, or the original native implementations.

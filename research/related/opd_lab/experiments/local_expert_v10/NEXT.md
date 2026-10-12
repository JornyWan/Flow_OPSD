# Next: test transfer before adding more mechanisms

Phase10 completed the bounded learning question: fresh external-expert sparse
targets can change ordinary full generation beneficially on the chosen apple
case, with four preservation controls. Dense32 did not improve that terminal
result despite smaller drift. Keep both results; no further single-case tuning
or claim that all trajectory supervision is ineffective.

The next concrete gate should be evaluation, with NO additional training:

1. Freeze four models: initial joint128, sparse32, sparse320, dense32. Include
   both sparse checkpoints rather than selecting a checkpoint separately per
   image. Exported raw adapters exist in phase10; sparse32 has saved raw tensors
   and needs a separately logged export if using the standard eval launcher.
2. Freeze two fresh seeds per each of the five trained prompts (10 pairs), plus
   one fresh seed on ALL32 existing validation prompts (32 pairs). Four models
   give168 images. Derive seeds deterministically from a new phase identifier,
   record them before generation, and verify they differ from all five fit seeds.
   These validation prompts are an existing development split, not a newly
   untouched test set. Keep the64 test prompts for a later frozen method comparison.
3. Use ordinary flow10/512/CFG4.5 generation with shared per-pair CPU FP32 noise.
   Freeze the same strict clause/visibility review, preserve all images and
   unchanged proxy scores, and review without model or score. Report every
   prompt/seed pair, losses and unknowns, not only successful apple pictures.
4. Analyze fresh-seed transfer separately from the existing validation split.
   Report paired clause/joint/visibility changes and failure categories. Aggregate
   seeds within prompts for uncertainty; ten seeds are not ten independent prompts.

Expected diagnostic outcomes, not promises:
- If improvement exists only at the trained seed, the result is memorization;
  expand state/prompt coverage in training before any sampler study.
- If new seeds preserve gains with limited collateral damage, run a prospective
  cohort through the existing collect/verify/fit/refresh loop. Include all64 train
  queries, with no selection of cases using future full-generation results.
- If broader prompts regress, measure the preservation tradeoff on that cohort
  before adding dense supervision or changing teacher/strength/timing.

Any subsequent online comparison needs common initialization, budgets, data and
evaluation: continued original OPD, best fixed expert/timing/strength chosen only
on development data, and compute/density-matched screening controls. The current
one-repair oracle target is not evidence that an automatic verifier can select
useful actions reliably across the distribution.

Existing engine already refreshes behavior each outer epoch. Phase10's sparse
diagnostic additionally assigns unit-weight preservation targets to four controls;
the default online engine currently assigns zero loss to rejected queries. Carry
that distinction into any new configuration explicitly. Dense trajectory capture
is a separate experimental option, not a proven requirement or a default change.

After this transfer gate and a prospective online cohort, evaluate multiple step
counts/samplers on fixed checkpoints. Do not infer sampler robustness from this
single-grid experiment. This file is a plan, not an executed/frozen phase11 run.

# Phase15: label-assisted fixed-target mechanism pilot

Freeze before GPU execution. This refines phase14/NEXT: retain the ORIGINAL task
and unchanged labels for an inexpensive precursor, before simultaneously changing
task scope, annotation, collection and learning. No global-text or visibility clause
is removed. Do not repeat a general-purpose VLM audit or call this the stopped
automatic-verifier128-state confirmation. This is a small selected-cohort diagnostic
using provisional implementing-assistant labels, not a true independent oracle.

## Question and sources

Do the existing labels identify local teacher actions that are more useful to fit
than a matched random selection, and does the benefit survive complete generation
from fresh noise? Use the same joint128 raw checkpoint and exact saved phase12
states/actions/context. Read all64 query labels; include EVERY known-safe state
(cases2,6,17,34,42,58). Choose its safe candidate by most repaired encoded clauses,
then smaller saved action distance, then canonical OCR-before-GenEval order.
The resulting teacher mixture is2 OCR+4 GenEval. Never use phase14's predictions.

Select six additional preservation queries from the remaining phase12 states,
two from each of the three positive groups (count,color,relation), by fixed hash
order. This is a12-query selected-cohort pilot, not representative all64-prompt
training. Uncertainty excludes positive supervision; existing known labels are
never rewritten. Actions are external experts, not self-distillation targets.

## Matched arms and optimization

All arms use these same12 states, in sorted id order. A noop target is the saved
joint128 action at that exact state; unit weight, not a dropped example.

- label_selected:6 labelled safe expert actions+6 noop preservation targets.
- random_matched:6 expert targets+6 noop, with exactly the same accepted-count,
  task-stratum and teacher-count distribution as label_selected. Within each group,
  choose query identities by fixed hash and assign the teacher slots by another
  fixed hash. The control routine sees ids/groups/teacher quotas, not outcomes.
- all_teacher:12 external targets with twice the above teacher/group counts. Its
  six random_matched accepted actions are identical, and the other six also get
  a teacher target. This is an UNFILTERED FROZEN-STATE control, not an exact native
  DiffusionOPD run. Native on-policy OPD is still needed for a later method claim.
- joint128:unchanged initialization used as the generation reference.

Use existing transition_mean_loss and fitting_probe.update_once,keep=1; average
over12 points. All3 trained arms get128 successful AdamW updates=1536 backward
queries each,4608 total. Common reset optimizer/scaler/RNG and weight warm start,
not full resume. Same learning rate3e-5,betas(.9,.999),eps1e-8,decay1e-4,clip1.
FP16,LoRA rank32/alpha64,CFG4.5,512px,flow10,query index3,alpha1. Like phase10's
fixed-target probe, behavior stays joint128, no EMA update or behavior refresh.
No new teacher query. Audit only numerical fitting at steps0/32/128; evaluate
ONLY the preselected terminal128 checkpoint, no checkpoint selection.

Before fitting, exact-replay all12 baseline images and all6 selected counterfactual
targets; verify all12 saved noop actions and all saved action hashes. Any mismatch
fails the attempt before learning. Keep all logs/checkpoints; never overwrite runs.

## Frozen complete-generation evaluation

Four models each generate the SAME38 prompt/noise pairs,152 images:

- 6 positive fitting prompts×2 fresh seeds=12 primary repair-transfer pairs.
- 6 preservation fitting prompts×2 fresh seeds=12 preservation-transfer pairs.
- 8 other train prompts (2 per original group)×1 fresh seed=8 broader development
  pairs. These receive no supervision in THIS probe, not necessarily in history.
- 6 positive original fitting seeds=6 in-sample full-rollout diagnostics.

All32 fresh seeds are fixed by stable_seed(15015,version,id,replicate), disjoint
from phase10 fitting, phase11 evaluation and phase12 collection seeds. No validation
or test sampling. Input noise hashes must agree across models. The broader prompts
are selected by hash before images exist. Do not mistake fewer unknowns for repairs.

Original v1 OCR/GenEval is logged for diagnosis only; review ALL152 outputs with
per-pair randomized A/B/C/D slots and randomized pair order. Apply unchanged encoded
clauses and full_visibility; copy literal text, preserve small print/extra letters,
cropping/occlusion and nulls. Freeze every observation before reading the model key
or scores. Review is by the implementing assistant and is not independent annotation.
Images identical to archived/other outputs must have identical semantic labels.

Report complete_scene, joint clauses, every clause, full_visibility and unknowns;
paired safe-repair/known-harm and baseline-true-to-false/unknown rates; target fitting
alignment and noop drift; all costs. Bootstrap2000 whole prompts,seed0,keeping the
two seeds clustered. Primary cohort has ONLY6 prompts: intervals will be wide.
Aggregate fresh-seed results as an additional descriptive view, never hide cohort
regressions. Target or prompt selection based on old outcomes limits interpretation.

## Budget and decision

OneL40,4CPU,96GB,60min cap (1GPU-hour). Expected totals:4608 backward queries,
6356 student forwards,48 behavior forwards,12808 CFG forward examples,170 decoded
and v1-scored images (18 exact references+152 evaluation). Record actual per-arm
and total counts and Slurm allocation including loading/replay/evaluation. The
old phase12 collection cost0.118889 GPUh is a separate inherited data cost, not
zero-cost supervision; annotation wall time was not fully instrumented.

Prespecified promising-pilot checks (not significance or method validation):
median labelled-target relative MSE<=0.5; at least2 additional complete primary
fresh-seed endpoints vs joint128 AND at least2 vs random_matched; label_selected
has zero known harm on all32 fresh pairs; and complete-scene totals on the other
two fresh cohorts are each no worse than joint128. These strict checks separate
fitting, transfer, selection value and preservation. No arm/model is chosen by
the old proxy score. If a check fails, diagnose underfitting, trajectory mismatch
or collateral drift; do not tune LR/steps/seeds on this evaluation. If promising,
prepare a new multi-seed/larger-cohort protocol and an actual native OPD baseline;
do not yet claim automation, convergence or sampler robustness.

# Phase18: fixed-index multistate supervision (freeze before training)

Question: do additional endpoint-certified states improve fresh-noise generalization?
Keep raw joint128, SD3.5-M LoRA32/alpha64, flow10,512px,fp16,index3/alpha1,
student/behaviorCFG4.5 and original expert actions/CFGs. Original OPD local transition
MSE, AdamW lr3e-5/betas.9,.999/eps1e-8/weight_decay1e-4, gradclip1,128updates,
12 accumulated queries/update; raw model without EMA or behavior refresh, as phase15.
This is assistant-label fixed-query mechanism testing, not native online OPD or an
automatic verifier method. No further teacher queries, no reward/model/label tuning.

Inventory: original six positive and six preservation states from phase15; add
phase16 count08/1272885951/GenEval, color11/15411324/GenEval and phase17
count08/1451226237/OCR, all index3/alpha1. Canonical action names map explicitly to
source payloads (phase17 i3_ocr@1, NOT its expanded-pool chosen i1_GenEval@0.5).
Keys use(id,seed,index), never prompt id alone. Check source query/action/image
hashes. Replay all15 baseline trajectories/state/context/noop and9 chosen target
endpoints exactly,24 reference images, before any optimization.

One common fitting/evaluation implementation reads three explicit schedules:

- single: original12 states, same order/loss as phase15 label_selected. Require
  final model tensor hash to reproduce the prior model exactly, before other arms.
- multi: same12 prompt slots/update. Each positive prompt alternates uniformly
  over its own certified states; six preservation prompts retain noop. Each prompt
  gets128 queries. color11 two seeds get64each; count08 three get43/43/42.
- random_matched: exact same state schedule as multi. Deterministically shuffle
  expert/noop ROLE labels within each task stratum at each update, with no outcome
  access. Match each update's expert counts, positive/noop counts and task counts.
  A role always uses the receiving state's own expert action, never another state's
  tensor. This control can assign teachers to original protection slots: the single
  and multi arms retain their six explicit protection labels; the random arm tests
  the consequences of random acceptance/routing. It is not native OPD. Preserve
  the complete schedule, including coincidental correct choices and any varying
  role for repeated states, rather than cherry-picking a randomization.

All arms reset identical weights, empty AdamW state, scaler and RNG. Each1536
backward queries, total4608. Audit all15 saved states at0/32/128; only step128 is
evaluated, no best-checkpoint choice. More states change repair/expert diversity as
well as sample count; only two prompts gain states. Do not claim an isolated
sample-count effect or six-prompt multistate coverage.

Freeze32 new evaluation seeds, disjoint from the saved historical seed-field
inventory including phase17 and failed attempts. Same12 training prompt identities
get two seeds each:12positive+12preservation. Eight prior unfitted TRAIN prompts get
one fresh seed each; these prompts were exposed as development evaluations, not
claimed unseen prompt texts. None of the32 fresh-noise images exists before fitting.
Add the three new fitting states as diagnostics, not test samples. Keep all35pairs.

Evaluate joint128 and three trained models on all35 full generations (140 images).
Also evaluate each trained model as one index3 action on the joint128 prefix/suffix
for all12 fresh positive states and3 added fitting states (45images). The baseline
for single-action comparisons is the same full joint128 image. Total185 evaluation
images +24references=209decodes/verifier calls. Store the15 common evaluation
query states/noops/contexts/noise hashes; identical paired noise across all paths.

Budget: single L40,45min cap. Expected exact counts: student6338, behavior339,
CFG forward examples13354, backward4608, teacher0, decode/verifier209,
text_encode_prompts40. Report allocatedGPU-hours including failures, fitting time
by arm, inherited collection costs separately. Equal updates do not imply equal
total research/collection cost.

Review50 sheets with four randomized views each (200view slots of185 outputs;
15baselines repeat for local comparisons). Keep native512pixels. Model/path/seed/
scores hidden for fresh data; exact historical pixel matches inherit prior labels
and are explicitly counted. Keep every image, uncertainty and duplicate; use the
unchanged original-task rubric, freeze all200 observations before joining keys or
scores. Assistant labels remain provisional; no automatic acceptance claim.

Primary: multi minus single complete-scene rate on12 positive fresh-noise states,
with prompt-cluster bootstrap2000/seed0, six prompts rather than12independent
observations. Also compare joint128 and matched random. Separately report the
four fresh states of the two prompts receiving additions versus eight states of
the other four prompts. Report all32fresh, preservation and unfitted cohorts;
known harms, true-to-false/unknown transitions, unknowns, full versus single paths
and original6/added3 target relativeMSE. Unknown never counts complete or safe.

Predeclared promising-pilot checks (not statistical proof): added3 target median
relativeMSE<=.5; multi full primary at least2more complete than initial,1more than
single,2more than random; zero known harms across all32fresh; preservation/unfitted
complete counts no worse than initial. Report all checks even if one fails. Stop
scaling training when only target fit improves; use local/full comparison to decide
whether state generalization or trajectory interference remains the main limit.
No changes to index1, strength grid, loss, verifier or sampler in this comparison.

## Resource amendment before attempt002

The original L40 attempt001 is pending (scheduler estimate about70minutes at
submission). Permit one H100 attempt002 with the identical scientific plan,
source/config/targets/seeds/schedules and exact replay gates. Cap45minutes; zero
tolerance change. Any reference image/state/noop mismatch stops before fitting;
any original-six final-weight mismatch stops before the other two arms. Retain
the L40 queued fallback until the H100 attempt passes both gates, then cancel the
unused pending job. Failed qualification costs are counted. The attempt001 frozen
run copy retains its original L40 protocol. This is a resource-equivalence attempt,
not new algorithm tuning or an additional seed sweep.

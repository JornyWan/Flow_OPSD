# Phase11: transfer gate after the selected-case learning result

Freeze before GPU execution. No training or teacher queries. Four fixed raw
states: phase7 joint128, phase10 sparse32, sparse320, dense32. Load saved tensor
states directly through the existing backend; no adapter export or precision
conversion is introduced. Keep previous runs, labels and shared assets unchanged.

Two cohorts, reported separately:
- Five fitted prompts, each with two fresh seeds:10 prompt/seed pairs.
- ALL32 prompts in the existing validation split, each with one fresh seed:32
  pairs, eight per count/color/relation/count_color family. This is an existing
  development split, not a new untouched test set. No test prompt is generated.

Four models times42 pairs gives168 transfer images. Seeds are
stable_seed(11011,"phase11_transfer_v1",cohort,id,replicate), frozen in probe_plan.
Require distinct pair IDs/seeds and no collision with any of the five fitting
seeds. Each model uses the same per-pair CPU FP32 noise. Ordinary flow10,512px,
CFG4.5,fp16, same negative prompt/max_sequence_length/scheduler/verifier as phase10.
No expert action, candidate selection, inference mask or per-image checkpoint choice.

Additionally regenerate the known apple training-seed image once per model:
four anchor images must exactly match previous image hashes. Anchors are checks,
excluded from all transfer metrics. Total172 images,1720 student forwards,
3440 CFG examples,172 decoded/verifier images,zero backwards/teacher forwards.
Record image/noise/model hashes and exact input/source snapshots. One L40 GPU,
four CPUs,96GB,30-minute cap; account loading/checks/failed attempts.

Review all168 images before exposing model identity or scores. Present each pair
as a2x2 sheet with independently permuted A/B/C/D model positions, and randomize
pair order. Same prompt is visible; labels remain model/score-blind. Freeze all
clause/visibility/text/notes before key join. Exact duplicates may share labels
but must retain every matrix entry; report implementing-assistant familiarity.
Use REVIEW_RUBRIC.md, retaining unknown and concrete non-encoded defects.

Primary measures: visually verified joint clauses plus visibility, paired change
versus joint128, and regression of each previously true clause/visibility. Report
joint-clause-only and component results too. Unknown is not success; report its
coverage and exclude unknown transitions from definite repair/harm counts. For
strict-success rate unknown is a non-success, not a claimed definite failure.
Bootstrap paired deltas by PROMPT with2000 resamples/seed0, averaging repeated
seeds within each prompt first. Give raw paired rows and denominators. Five
training prompts are too few for a strong inferential claim. No winner selection
or multiple-comparison significance claim from this development evaluation.

Also report unchanged v1 proxy scores, with the same pair/prompt aggregation;
never substitute them for visual labels. Color/count/visibility uncertainty and
plate lettering remain explicit. Report the two fresh apple seeds as part of the
complete ten-pair training cohort, not as the only displayed results.

If gains are confined to the fitted seed, broaden training coverage before a
sampler study. If new seeds benefit but validation regresses, investigate the
preservation tradeoff on a prospective cohort. If both cohorts show usable
headroom without clear collateral damage, proceed to a matched online experiment;
this test alone does not establish automatic verifier/selector superiority.

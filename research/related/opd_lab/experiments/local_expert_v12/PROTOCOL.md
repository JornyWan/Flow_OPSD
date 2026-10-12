# Phase12: prospective train-only candidate and selector audit

Freeze before GPU generation and before inspecting candidate outcomes. This is
an external-teacher action audit at fixed joint128, not further student training.
All64 existing train prompts, one fresh seed per prompt, four task families
interleaved. No validation/test generations and no per-image model selection.
Seeds: stable_seed(12012,"phase12_prospective_v1",record_id). Assert uniqueness
and exclude all five phase10 fitting seeds and all42 phase11 transfer seeds.

Flow10/512/CFG4.5/fp16, CPU FP32 initial noise, query index3, alpha1. Candidates
in canonical order: noop,ocr@1,geneval@1. Preserve teacher CFG (OCR4.5,GenEval1),
negative prompt, scheduler and verifier definitions from phase10/11. Each query
shares one frozen behavior prefix/query state and uses that same frozen model
for every continuation. Keep the original collect_query/select implementation.
The selector retains utility_joint_weight=.5,min_gain=0,protect_satisfied=true,
no quality floor/drop guard. It protects proxy-true encoded clauses; visibility
and extra-text defects are assessed independently, not silently added to it.

64 queries give192 candidate endpoint images and128 expert action queries.
At index3 there are3 prefix +1 query +3*6 suffix behavior calls per query:
1408 behavior forwards,64 OCR and64 GenEval forwards. One excluded historical
apple-seed anchor uses10 student forwards and must match old pixels exactly.
Total193 decoded/verifier images,3028 CFG examples,zero backward/optimizer steps.
One L40 GPU/four CPUs/96GB,30-minute wall cap; account loading and failed attempts.
Persist per-query state/context, all three actions, selected target, noise/state/
action/image hashes, actual counters and elapsed time. Never overwrite an attempt.

The automatic choice is written during collection, before visual review. Review
ALL192 images in64 three-view sheets: independently permute A/B/C per prompt,
randomize prompt order, hide candidate identities/scores/decisions. Freeze all
clause/visibility/text/notes labels before key join. Use REVIEW_RUBRIC.md and
retain null. Implementing-assistant review is not independent human ground truth.

Primary safe encoded repair: at least one baseline false clause becomes true,
no baseline true clause becomes false, no true visibility becomes false, and ALL
baseline and candidate labels are determinate. Candidate or baseline uncertainty
cannot certify safety. Definite harm is still counted if another label is unknown.
No determinate repair and no unknown => known non-repair; unknown without definite
harm => unresolved. Report visibility-only gain and complete-scene gain separately.
The stricter requirement on baseline unknown is intentional; report the old
phase9 compare_observations result separately if needed, never interchange them.

Report candidate headroom for each expert and either expert; frozen selector
acceptance coverage; certified safe repairs / ALL accepted (conservative precision),
precision on determinate accepted with its denominator, accepted unknown, known
harms, and successful safe selections / all queries with any certified safe repair
(recall including noop/rejections). Give missed repairs and false accepted cases,
all per-clause true->false/unknown denominators, complete/joint/visibility rates
for noop, each fixed expert and selected policy. For a rejected query selected
equals noop. Candidates from one prompt are dependent: use prompt bootstrap
2000 resamples/seed0 for rate/delta intervals. Ratio metrics resample whole
prompts, never independent candidate rows. Zero-denominator ratios remain null.

Preregistered engineering gate for a subsequent small learning experiment:
at least8 certified-safe accepted queries, certified-safe/ALL accepted >=80%,
zero accepted queries with a definite harm, and unresolved accepted <=10%.
These are conservative pilot criteria, not statistical proof or a tuned optimum.
If unmet, do not launch fitting from these automatically accepted targets as if
validated. If candidate headroom exists but the selector misses/falsely accepts,
diagnose measurement/routing and confirm any revised rule on new seeds. If little
headroom exists, evaluate a separately frozen intervention grid/teacher coverage.
No rule adjustment, manual target curation or fitting inside this audit phase.

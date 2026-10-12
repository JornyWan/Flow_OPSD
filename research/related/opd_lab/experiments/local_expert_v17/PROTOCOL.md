# Phase17 prospective candidate coverage (freeze before GPU)

No fitting/optimizer, checkpoint selection, prompt edits or automatic acceptance.
Six previously selected positive prompts, two new deterministic collection seeds
each. Exclude seeds in all prior lab top-level run/campaign plans, manifests,
configs, protocols, query and endpoint/result records, including failed attempts;
save the exact file inventory, hashes and seed union. These new images are
prospective development/target collection, never an untouched method test set.

Fixed raw joint128, SD3.5-M LoRA32/alpha64, flow10,512px,fp16, student/behavior
CFG4.5, OCR CFG4.5, GenEval CFG1.0. Replay six old full baseline endpoints and six
old selected teacher endpoints, checking PNG hashes plus saved state/context/noop.
No new teacher query until all12 references pass. Then generate each new full
baseline once and retain all11 states,10 actions, sigmas, context and CPU noise hash.

Query both experts at indices1,3,5 on that shared trajectory. At each index use
alpha0.5 and1.0 with exactly the existing FP32 expression noop+alpha*(expert-noop),
including alpha1; preserve raw expert actions separately. Only this action changes;
joint128 completes each suffix. Twelve branches plus one baseline per seed:
156 new images,72 teacher queries,0 updates. Save all36 query payloads and12 full
trajectories with action/state hashes and source model identity.

Budget: one L40,15min cap. Reference60 student calls +42 behavior (6 noops+36 suffix);
new baseline120 behavior; branches864 behavior (12 states×4×(8+6+4)). Total student60,
behavior1026,OCR36,GenEval36, CFG examples2280, decode/verifier168, text-encode12.
Actual counters must equal this frozen plan. Reference overhead is included.

Review every new image with hidden baseline/expert/index/alpha/seed/score identity;
12 cases,13 randomized views each, four native-resolution sheets per case. All
156 new labels, including the12 baselines, are newly reviewed. Original task
rubric unchanged, explicit bool/null for every clause and full_visibility plus
literal text and notes; freeze all labels before joining key or scores. Same
pixels and requirements require same labels. Assistant annotations remain
provisional, not independent human ground truth. Retain all unknowns and images.

Predeclared candidate pools: fixed_i3_a1 (2), i1_a1 (2), i5_a1 (2),
i3_all_strengths (4), all_times_a1 (6), expanded (12). All sets share the same
12 baselines and noise. Compare known-safe availability, unresolved-without-safe,
known harms, oracle complete availability including noop, and per-prompt count of
safe states (0/1/2). Report every individual candidate configuration. Main contrast
is expanded minus fixed safe-state availability, prompt-cluster bootstrap2000,
seed0; six selected prompts, not12 independent prompts. Expansion uses more
queries/rollouts and does not establish computational superiority.

Save an explicitly assistant-assisted, untrained target manifest. For each pool
choose only determinate safe repairs (at least one false→true encoded clause, no
true→false clause or visibility, all before/after labels known). Among those prefer
more repaired clauses, then smaller transition displacement MSE = action_distance
times dt², then canonical candidate name. If none, retain noop; do not turn
unknown into supervision. This is an oracle diagnostic/target preparation rule,
not an automatic verifier result or trained-policy metric. Inspect whether the
same prompt has safe targets on both seeds before deciding on multi-state fitting.

Do not revise the grid, labels, thresholds or seed set after observing outputs.
If coverage remains sparse, diagnose candidate limitations before adding training.
Future training comparisons must freeze independent evaluation seeds/prompts,
include the same-initialization continued native OPD and matched controls, and
account for collection/verification costs. No sampler transfer claim in this phase.

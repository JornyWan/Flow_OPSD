# Phase20: single-GenEval acceptance collection and conditional learning study

Freeze before any new image or label. Main question: with the same teacher and
same student states, does endpoint-validated acceptance improve learning relative
to unconditional teacher supervision and matched random acceptance?

## Collection first, with a binding stop condition

Use the six phase18 positive prompt identities, each with eight new collection
noise seeds (48states), without success/uncertainty prefiltering. Separately reserve
the same cohort composition as phase18:12positive+12preservation+8unfitted prompt
states,32 entirely new evaluation seeds. All80 seeds must be mutually disjoint and
absent from the complete prior lab seed-field inventory including phases18/19 and
failed attempts. Do not generate any reserved evaluation image during collection.
Prompt identities are familiar development prompts; only noise seeds are new.

Use raw joint128, SD3.5-M,512px,fp16,flow10,index3,direct alpha1 GenEval action.
Student and frozen behavior CFG4.5; GenEval CFG1. OCR adapter may remain loaded
by the unchanged backend but receives zero queries and supplies zero targets.
No optimization, student refresh, time/strength/sampler sweep or new reward.
Reuse existing state tracing and frozen-suffix endpoint functions.

Before collection, reproduce six phase19 reference states: replicate0 for each
prompt regardless of outcome. Verify state/context/noise/dt/noop and freshly
predicted GenEval action exactly; reproduce their12 baseline/teacher endpoints.
Same L40S/Torch as phase19, CUBLAS_WORKSPACE_CONFIG=:4096:8 and original evaluation
numerical mode. Any mismatch stops the attempt before collection. Never relax
hash/tensor equality or replace a failed sample to obtain a pass.

Each new state gets one baseline trajectory, one GenEval prediction and one
teacher-modified endpoint with the same frozen joint128 suffix. Save all48 state
payloads, both actions,96 endpoints and all references/counters/hashes. Expected
counts:student540,behavior330,GenEval54 (6reference+48collection),OCR0,
forward_examples1794,decode/verifier108,text_encode_prompts12,backward0.
One L40 allocation capped at20minutes; count failed attempts and reference overhead.

Review all48 randomized baseline/teacher pairs,96 native512px views on24 pages.
Model/role/seed/score hidden until all observations freeze. Exact known pixels
inherit prior frozen labels, otherwise every field is explicitly bool/null.
Use the unchanged original-task rubric, including extra letters, exact counts,
full visibility, within-word gaps and uncertainty. Implementing-assistant labels
are provisional development supervision, not independent human or automatic truth.
Keep all ambiguous images and raw observations. Freeze before joining identities.

Accept only safe_repair from the existing compare function: at least one known
false-to-true encoded clause, no lost true clause/visibility, all required labels
known. Others target noop, including unknown or already-correct/no-improvement
states. Partial safe repairs are accepted but reported separately from complete
images. Visibility-only improvement is reported, not an encoded repair target.

Proceed to training ONLY if all three predeclared coverage requirements hold:
- at least8 accepted safe states out of48;
- at least3 of the6 prompts each have at least2 accepted states;
- at least4 accepted states end in a fully correct scene, including visibility.

These are engineering feasibility thresholds: they require repeated-seed coverage
for at least half the prompts and multiple whole-scene successes. They are not a
statistical significance test. Do not lower them after seeing results. If any fails,
archive the complete collection, targets and reasons; perform no fitting, leave
reserved evaluation seeds unrendered, and diagnose coverage before more training.

## Training design if coverage passes

Reuse the existing schedule-driven local fitting core with128updates,12queries per
update, original OPD transition MSE/AdamW lr3e-5/betas.9,.999/eps1e-8/wd1e-4/clip1,
LoRA32/alpha64, deterministic fitting qualified in phase18, no EMA/behavior refresh.
Each arm resets identical raw joint128 weights, empty optimizer/scaler and RNG.
Six positive prompt slots cycle through all8 collection states,16visits per state;
six original preservation prompt slots retain their same noop targets throughout.
All arms use identical states/order and1536backward queries. Do not train only
accepted states or add the known phase19 failure states as special examples.

Arms: all_teacher uses GenEval at every positive prompt state; certified uses
GenEval only at accepted states, otherwise noop; random_matched statically permutes
the8 binary accept labels within each positive prompt, seeded and frozen without
outcome access. Random matches accepted counts per prompt and per complete8-update
cycle, not necessarily every update. Preserve accidental correct/identity matches,
never resample a convenient randomization. Protection slots remain noop in all arms.

Keep step128 only, no checkpoint selection. Audit accepted target and noop drift
at0/32/128. Persist each arm report and complete training states before evaluation.
Reserve at most60minutes L40 fitting and20minutes evaluation if the gate passes;
run fitting and evaluation separately to avoid losing completed fit reports.

Evaluate all32 reserved seeds with raw joint128 and each trained arm. For each arm,
both full generation and a single index3 action with the common initial-model
prefix/suffix are reported. Thus32baseline+3*32*2=224evaluation images. Single-action
results are primary because phase18 showed whole-trajectory side effects; full
results remain a separate deployment stress check. Equal inference setup across arms.

Primary: complete-scene gain on12positive new seeds, prompt-cluster bootstrap
2000/seed0 across6prompts; report each arm versus initial, certified versus both
controls. Report all32 safe/known-harm/unknown counts, preservation/unfitted cohorts,
full versus local, collection fit, teacher query cost and allocated GPU-hours.
A promising pilot requires certified local at least2more complete than initial,
at least1more than each control, zero known harms on32local evaluations, and no
loss of complete preservation/unfitted counts. Report failures and uncertainty.
This is a single-query OPD-style fixed-query mechanism test, not native online OPD,
self-distillation, a deployable verifier result, or evidence of sampler robustness.

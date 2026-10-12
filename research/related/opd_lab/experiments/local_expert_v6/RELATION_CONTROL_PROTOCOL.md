# Post-audit interface control, fixed before new requests

Motivation: phase6 initial Qwen judged some reversed relations true while its
evidence described the actual scene. The raw GenEval position=[direction,index]
representation may be ambiguous to the model. This is an interface hypothesis,
not proof that model perception itself cannot solve relations.

Use ALL20 relation images already exposed in phase6 (12 development,4 prior,
4 newest calibration). None is now a held-out test for this change. Make two
fresh independent requests/image: the given relation and the opposite direction.
Compile subject/object explicitly from GenEval indices, then use a literal
English sentence stating identities, exact counts and direction as seen by the
viewer. Invert direction only; do not swap roles. The unchanged image and all
other settings are paired. Fixed SHA order of conditions; no conversation history,
original prose/text target, prior decisions or reference labels in requests.

Same pinned Qwen/model/environment, bf16,SDPA,greedy seed0, processor pixel bounds;
max192 output tokens for this shorter single-clause request. JSON satisfied is
bool/null, evidence string/null. This is a new versioned schema, not a repair or
rescore of the original15 invalid scene outputs. No outcome-dependent retry.

Report original versus explicit-given decisions against unchanged provisional
labels, broken down by the original cohorts. Report given/inverse both-true,
both-false, complementary or unknown. Opposite direction is not automatically a
known positive when required objects are absent/unclear, so do not invent inverse
truth labels. Both-true is inconsistent for these singular left/right constraints.

Report all40 calls and allocation (oneL40, <=10min). No training updates, reward
promotion, target reselection or claim of independent generalization. Even if the
interface helps here, freeze it before evaluating newly sampled calibration images.

# Phase7: what sufficient fitting establishes

## Result

The prior2/4/12-update probes had not exhausted learnability. With old targets
and behavior fixed,128 updates at LR3e-5 reduced mean relative target MSE to1.52%
for individual fits and3.38% for joint fitting. This is substantially closer to
the actions than phase3's four-pass result, but the selected data/training
geometry differ, so it is not a matched method comparison.

Joint128 preserved4/5 annotated local repairs and3/5 under ordinary whole-student
flow10 generation in an arm/score-blinded provisional review. The two old
targets with all encoded clauses correct and full visibility (bowl+BLOOM and
two bananas+SUNNY) retained both local and full-generation repairs. This closes
a small selected training-example chain: useful target → substantial fitting
→ actual generation improvement. It does not establish generalization.

| Old case / task | Individual residual MSE | Joint residual MSE | Joint local repair | Joint full-generation repair |
|---|---:|---:|---|---|
|9: one bowl, BLOOM|2.2394%|5.7471%|yes|yes|
|11: cup right of bottle, BLOOM|1.0939%|3.0245%|yes, old visibility defect remains|yes, cup cropped|
|12: bicycle right of car, WAVE|1.8017%|4.0380%|no, additional plate text|no, additional plate text|
|17: two bananas, SUNNY|.7939%|1.1978%|yes|yes|
|19: apple right of bowl, CLOUD|1.6817%|2.8895%|yes|no, apple primarily below bowl|

Lower action MSE is not monotonically better visually. Individual banana fitting
has lower MSE but its single-action image loses an identifiable N diagonal and
its full image says SUNY. Joint fitting retains SUNNY. Bowl case9 repairs even
though its joint MSE is slightly above5%. No post-hoc LR/checkpoint/image/target
selection was used. Only3/5 joint queries simultaneously pass the MSE and local
repair tests; the protocol's separate4/5 aggregate checks pass on different sets.

Full generation changes both the prefix state and suffix actions relative to
the fixed-state single-action result. The lost apple/bowl relation indicates a
deployment-path gap in this sparse probe, not proof that a particular late step
or solver caused it. Prefix and suffix must be isolated next.

## Native OPD positive control

The pinned author's SOPD-OCR entrypoint trained for32 updates on the first16
upstream training prompts, three sampled trajectories/update and nine supervised
times/trajectory. LR3e-4, original loss, AdamW, teacher/student CFG4.5 and EMA were
preserved. Single GPU/batch1 and the finite subset/budget are explicit changes
from the default recipe. Native retains its own encoding/trajectory/reward path;
it is a learning control, not a causal comparison to the local five-query fit.

| Actual evaluation update |0|8|16|24|
|---|---:|---:|---:|---:|
|Native EMA OCR proxy,16 test-prefix prompts|.627025|.751765|.835341|.867042|

This is a positive learning signal. Eval noises differ across checkpoints and
the small prompt set is reused. First/last online losses were .000465843 /
.000140168, on changing states. Neither statistic is a paired fixed-query fit
estimate. Checkpoints8/16/24/32 include separate raw and EMA weights; the last
reward above is update24, not32. This is not a paper reproduction.

## Teacher capability and CFG

All64 old phase3 TRAIN prompts and their fixed seeds were evaluated without
reward-based prompt selection. These are unchanged v1 automated proxy values,
subject to the OCR/detector false positives documented in phases4–6.

| Full-generation configuration |v1 utility|OCR exact proxy|Mean non-text clause success proxy|Joint proxy|
|---|---:|---:|---:|---:|
|Backbone CFG4.5|.578887|.578125|.617188|.359375|
|Backbone CFG1|.206890|.171875|.234375|.015625|
|OCR teacher CFG4.5|.681207|1.000000|.562500|.531250|
|GenEval teacher CFG1, original setting|.857863|.843750|.945313|.765625|
|GenEval teacher CFG4.5, matched setting|.737198|.687500|.867188|.562500|

Teachers have useful capabilities here. Merely lowering backbone CFG does not
explain GenEval's advantage; GenEval also improves at matched CFG4.5. OCR raises
text scores but lowers mean non-text success, consistent with a constraint
tradeoff. These full-model comparisons do not establish each local action's
value or dynamic-selection headroom over a tuned fixed teacher/time/alpha.
Paired prompt bootstrap intervals in results.json do not correct systematic
verifier error or training-prompt bias.

Copying each teacher LoRA into the student with matched CFG reproduced10/10
fixed-state predictions exactly. This is an implementation/representability
check at those states, not an optimizer or universal-capacity proof. All10 old
noop/target image hashes also replayed exactly.

## Review limitations

All20 final images were shuffled and shown without arm/mode/MSE/proxy scores.
Observations were hashed before joining the key. The implementing assistant was
familiar with old cases; these are not independent human annotations, and all
proportions are descriptive on deliberately selected training positives.

The OLD bicycle-target review labelled text correct based on the main WAVE
sign. Here additional visible license-plate lettering violates the existing
only-word requirement. Car variants also retain cropping. Old labels remain
unchanged: this limits treating old provisional labels as clean positives and
does not necessarily represent a new training-induced failure. These variants
are conservatively counted as failed strict text repairs. The two complete-
visibility cases provide the cleaner positive evidence.

Blurred but identifiable BLOOM and spaced CLOUD were accepted; missing/extra
characters or an unidentifiable N were not auto-corrected. Visibility is
separate. Some encoded successes retain blur, poor sign placement or cropping;
repair retention is not full image quality.

## Evidence and cost

Jobs1048850/1048860/1048861 all completed on L40S: total .458333 allocated
GPU-hours including startup, no failed GPU attempt. Local fit:768 optimizer
updates across six arms,1280 backward queries,60 metric rows,100 fitted images
plus10 exact references. Capability:320 full images and10 copy checks. Native
additionally evaluates64 images and collects96 training trajectories.

All fit arms restore the same initial student, empty Adam state, scaler and RNG;
behavior never refreshes. Joint loss averages five queries: equal updates are
not equal compute. Final full checkpoints and audit-step student tensors remain
available. Joint-v1 validation/test was unused. Full-student inference uses no
teacher or per-image selection.

One CPU review-preparation call encountered NFS's cached absence of the completed
run marker and exited before creating outputs. Refreshing only the new output
directory exposed the completed run; the unchanged command succeeded. No GPU
rerun or output alteration occurred. See PROTOCOL/selection/results/ledger,
learnability.png and ../../docs/phase7_runbook.md for all recorded inputs.

# Phase43: complete frozen development candidate experiment

Claim C1 development selection: same published Student, OCRTeacher and shared
behavior prefix; vary only predefined full-strength expert window start{1,3}
and length{1,2}. Baseline noop. 48 frozen development prompts x two fixed seeds
x five routes = 480 images. Four equal families, including OCR+count+color.
Confirmation is not generated in this phase. No training, no agent image reads.

Inherit Phase42 model/CFG4.5/Euler10/512/precision/deterministic protocol exactly.
Each candidate starts at its true shared prefix and recomputes its own suffix.
Save complete trajectories and all six local targets per paired case. First
case independently replays all branches and student/behavior equality without
extra decodes. Expected 3490 behavior + 588 expert + 10 student calls, 8176 CFG
examples, 480 decodes, 96 positive/empty text encodings. Check frozen adapters.

Use unchanged Phase42 A/B evaluators; require exact raw-evidence/label parity on
all eight smoke PNGs first. A uses semantic_accept only on declared clauses,
unknown abstains. B uses original-image native GenEval strict plus full-image
OCR exact, and separate substring OCR reward/CER. A/B share models and are not
human truth. Keep detection boxes, color logits/masks and all OCR evidence.
Errors remain explicit, no dropping or imputing failed rows; any runtime score
error blocks choosing T* and confirmation until resolved in a recorded retry.

Select the highest B joint-success-count nonnoop route as T*. Ties prefer
shorter, then earlier intervention. Choose C_fixed between T* and noop using B
joint counts, noop on a tie. Freeze this result with source hashes. Never select
the window by verifier acceptance, quality metric, human labels or confirmation.
Report fixed and A-gated results for every route; T* gate vs C_fixed is the
descriptive main development comparison, not independent confirmation.

For each route, report A accept/abstain, B repairs/harms and joint wins/losses;
same-B hindsight candidate headroom is diagnostic, not deployable performance
or true visual upper bound. Density-matched random control uses each family’s
gate count but no sample outcomes: expected metric is p*correction+(1-p)*noop.
It is an exposed-cohort mechanism control, not an online efficiency baseline.
Always retain all 96 cases; no score-selected subset. Report per-family and
per-template results and prompt-cluster/family-stratified bootstrap (20k,
seed43043), descriptive 95%, no corrected efficacy or training-repeat claim.

Score all five pinned native quality metrics on all480 outputs (2400 scores),
with original preprocessing/truncation and ImageReward scalar-vs-diagonal parity.
Report native PickScore/26 and raw x26, no composite quality index. No native
DrawBench or original GenEval benchmark claim for these custom prompts.

Budget: one L40 <=60 allocated minutes, reserved from the shared3 GPU-hours.
Prior smoke cost102s included, remaining10698s at submission. Count every failed
attempt; no automatic requeue. Record generation/scoring/quality/peak memory.
After complete development, update real costs before proposing confirmation.
This stage selects T*, measures opportunity and checks evaluation reliability;
it cannot prove learned-student improvements, external human quality or sampler
robustness. Existing completed archives and frozen data are unchanged.

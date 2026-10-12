# Phase23 — teacher capability and one fixed composition

Frozen before new images. We test whether the existing experts can supply useful complete scenes when they control the full trajectory, and whether a simple handoff combines their strengths. This diagnoses the Phase22 candidate-coverage limitation; it is not a training run or a new held-out test.

All24 Phase22 prompt/seed pairs, with original record IDs and cohort provenance retained. Both previously observed cohorts are now mechanism development data. No outcome filtering or prompt rewriting. Same clean SD3.5-M, zero-output rank32/alpha64 LoRA, flow10 shifted sigma grid,512px, fp16 forward/FP32 Euler state, CPU FP32 seeded initial noise and text embeddings. CFG: baseline4.5, GenEval1, OCR4.5, unchanged from Phase22. No optimizer, EMA update, selector, schedule search or reward model. Generic config training/split fields do not override the explicit24-reference plan.

Three fixed routes:

| Route | Indices0–2 | Indices3–9 | Calls per image | CFG-expanded transformer examples |
|---|---|---|---:|---:|
| full_geneval | GenEval | GenEval | 10 | 10 |
| full_ocr | OCR | OCR | 10 | 20 |
| handoff | GenEval | OCR | 10 | 17 |

Use full expert predictions (no alpha interpolation). These are capability references, not mathematical upper bounds or ordinary OPD-trained baselines. All actions are recomputed on actual current states; the handoff reuses no stale teacher actions. Compute each route separately, including the duplicate3 GenEval queries, and count actual work. Assert the first3 actions/first4 states equal the full GenEval prefix. Save each route's11 states,10 actions, route IDs, context and sigmas.

Reuse24 baseline PNGs and frozen baseline labels from Phase22. Before any teacher query, regenerate one complete baseline and require exact10 actions,11 states and PNG hash. All24 initial noises and freshly encoded prompt embeddings must exactly match their saved Phase22 references, and initial adapter hash must match both old cohorts. Inputs/outputs/images/adapter assets and source snapshot are attested. Historical GenEval single-step outputs/labels are included as an explicitly inherited contextual comparator; no new inference or relabeling of these24 images.

Review72 NEW candidate images at native512, with route names hidden among A/B/C, all observations frozen before key join. Baseline and local-control labels are inherited, never rerated silently. Same Phase21/22 complete-scene v2 rubric (copied unchanged): correct requested text, task constraint and full visibility. Preserve unknown. Record aesthetic/geometry defects separately in notes; complete-scene success is not comprehensive visual quality. This is provisional implementing-assistant review, not independent human judgment; previous scenes and some styles are recognizable.

Report all five fixed policies (baseline, inherited local GenEval, full GenEval, full OCR, handoff): complete count, text/task/visibility rates, known harm and unknown. Report paired differences between every new route and baseline/local control, and handoff versus each full teacher. Aggregate24 and describe the two original12-case cohorts separately as development subsets, never as fresh confirmation. Prompt-paired bootstrap intervals are descriptive and omit annotation/model-training uncertainty.

Additional mechanism summaries, fixed in advance: cases where full OCR text and full GenEval task are individually correct; cases where no single full teacher completes the scene but the handoff does; cases with no known-complete candidate. Success of separate clauses in different images is not proof they can be combined. There is no post-hoc best-policy promotion or automatic training gate. If a fixed policy shows credible useful gains, validate it on newly frozen data before training; if experts fail their intended clauses, first audit supported inference settings and task compatibility. Neither negative outcome proves local correction impossible in general.

Budget: one L40 allocation≤15min (≤0.25 GPU-hour including failures), no automatic retry.72 new candidate images+1 replay;24 copied baselines (96 main comparison slots), plus24 inherited local-control slots for context.720 physical teacher queries (GenEval312, OCR408); behavior10, student0, CFG-expanded forward examples1148, decodes73, text_encode_prompts48, backward0. Stop on replay/counter/integrity failure. No other jobs touched.

# Next: verify teacher capability and simple composition before another training run

Phase22 does not support extending the current local interval as the next training target generator. It also does not disprove the original idea: experts can repair individual clauses, but our current one-expert-at-a-time candidates rarely satisfy the whole scene. Oracle selection cannot supply missing actions.

The next bounded diagnostic should answer two separate questions: can the available experts produce the desired scene at all under this inference setup, and does composing their strengths require more than choosing one local correction?

Proposed Phase23 matrix, to freeze before generating its new outputs:

1. Keep clean SD3.5-M, flow10,512px, original per-expert CFG. Use all24 Phase22 prompt/seed pairs as a **mechanism development set**; these images/prompts have been inspected and must not be called a new held-out test. Reuse the attested baseline, with one exact replay gate.
2. Generate full-trajectory GenEval and full-trajectory OCR. These are capability references, neither ordinary OPD-trained students nor a mathematical upper bound on local policies.
3. Add one prespecified simple composition: GenEval on indices0–2, then OCR on3–9, querying each on the real current trajectory; no endpoint selector. This uses the already motivated early-composition/mid-text ordering, fixed before images. No learned router, schedule sweep or mixing-coefficient sweep.
4. Report complete scenes, each clause, known damage, unknowns, visible defects and actual query/CFG cost. Freeze observations before identities; keep all outputs and make a named comparison page. Per-clause improvements matter diagnostically, but do not replace the complete-scene objective.

96 comparison slots including the24 already saved baseline views;72 new candidate images,720 teacher calls plus replay. Tentative budget1L40≤15min,0.25 GPU-hour including failures, no optimizer. This is a plan, not a submitted experiment. Freeze a separate protocol and verify expected counters before allocation.

Decisions after that diagnostic:

- If full experts themselves rarely meet their own intended clauses: audit their supported inference settings, checkpoint/task compatibility and these constructed prompts before adding a more elaborate selector. Do not infer that full-trajectory failure proves every local correction impossible.
- If individual experts repair different clauses and the fixed handoff solves the joint task: retain that simple handoff as a mandatory baseline. Only pursue adaptive verification where it offers extra complete repairs or avoids damage on fresh confirmation data.
- If strong terminal targets are supported: return to the intended short on-policy cycle — freeze behavior, collect fresh states, evaluate local/interval corrections, fit all accepted intervened actions with the original OPD loss, update a few times, refresh behavior. Compare ordinary OPD, fixed/composed teacher targets, density-matched random acceptance and verified acceptance from the same initialization and with total GPU cost recorded.

Do not resume long fitting of historical saved queries, silently remove visibility, change unknown to failure, or present endpoint-label oracle as the deployed method. Independent checks of critical positive/negative examples remain valuable, but no missing human answer should be represented as agreement.

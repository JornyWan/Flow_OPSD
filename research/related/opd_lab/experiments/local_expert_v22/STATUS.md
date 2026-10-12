# Phase22 completed — extending the intervention did not add complete scenes

2026-09-20. Both bounded GPU jobs completed and passed trajectory/image/counter checks. No ongoing Phase22 jobs, optimizer updates or automatic retries.

Open [the offline named comparison](report.html): all24 cases,120 original512px PNGs,26MB, default paired GenEval single/two-step, selectable baseline/OCR/oracles, five-view mode and click-to-enlarge. Nothing selected by outcome; download HTML alone to view images locally. Labels are provisional implementing-assistant observations, frozen with method identity hidden before key joins; no new independent human ratings were received.

| Policy | Development complete /12 | Confirmation complete /12 | Confirmation known harms vs noop |
|---|---:|---:|---:|
| Clean backbone / noop | 1 | 2 | 0 |
| Fixed GenEval single, i1 α0.5 | 2 | 3 | 0 |
| Fixed GenEval two-step, i1–2 α0.5 | 1 | 2 | 1 |
| Fixed OCR single, i3 α1 | 2 | 2 | 0 |
| Fixed OCR two-step, i3–4 α1 | 2 | 2 | 0 |
| Endpoint-label oracle, singles only | 2 | 3 | 0 |
| Endpoint-label oracle, expanded set | 2 | 3 | 0 |

The prespecified confirmation engineering gate failed: 12/12 known baselines, but zero additional complete scenes versus the single-step oracle (required≥2). Expanded oracle also ties the prespecified early GenEval single-step on complete scenes and known harm. This is not a learned verifier/student result. Oracle zero harm is partly built into the rule; do not interpret it as independent safety evidence. Small constructed cohorts and provisional labels do not support a general claim that intervals always fail.

Concrete confirmation cases:

- LOTUS / case8: GenEval single and two-step both change banana bunches into two separate bananas. The single step already captures this complete repair.
- WHEAT / case12: GenEval two-step adds an I (WHEAIT), breaking text that was correct under noop/single; layout remains correct.
- AZURE / case11: two-step GenEval supplies readable text and three vases, but bottoms remain cropped. This is useful partial complementarity, not complete success.
- LEMON / case9: two-step reduces obvious frame truncation, but faint extra fruit-like shapes make count/visibility uncertain; retained as unknown, not counted as a repair.
- PAPAYA / case3 and ORCHID / case7: all five outputs retain spelling errors; SEASHELL / case1 never obtains exactly one book plus one bowl; SUMMIT / case2 retains extra objects.

Failure sensitivity (post-hoc, `failure_audit.json`): development10/12 cases have a known false condition in every one of the five outputs; confirmation8/12 do. One further confirmation case is unresolved due to unknowns. Holding known labels fixed, even favorable resolution of all unknowns adds at most1 complete scene, below the≥2 gate. This bound does not protect against errors in currently known labels. Keep independent human checking separate.

Engineering evidence: 9 relevant CPU tests passed. Single-step branch exactly matches the existing `single_endpoint` kernel; zero-strength two-step matches every baseline suffix state and PNG before any teacher call; later queries use evolving branch states. All48 candidate branches per cohort have exact saved Euler recurrence/action interpolation; paired first transitions agree, second branch states differ from baseline; clean zero-output LoRA and initialization hashes agree across cohorts. All output tensors, original images, inputs and review records are hashed. These checks establish this local experiment's implementation contracts, not reproduction of a complete official training recipe.

Resources: development1059002 COMPLETED0:0,110s; confirmation1059008 COMPLETED0:0,109s;219 allocated L40 GPU-seconds =0.060833 GPU-hour.144 physical teacher queries,120 comparison images+2 noop reference images,0 optimizer updates. Confirmation was briefly pending; Slurm's16:44 estimate was superseded by earlier resource availability. No job settings or frozen matrix changed. Both cohorts cost more teacher queries for two-step than one-step; no compute-matched efficiency claim.

Interpretation: the immediate limitation is useful candidate coverage and some added interference. Repeating offline fitting on the same candidates is not justified by this experiment. Next: establish teacher capability and simple fixed multi-expert composition before another search/selection expansion. See [NEXT.md](NEXT.md) for the bounded diagnostic plan.

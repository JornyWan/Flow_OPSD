# Standard metrics and local-correction evaluation

User request after Phase39: recover the quantitative metric coverage used in the
earlier OPD/OPSD work, without bulk agent image reading. Phase40 begins this on
complete cached cohorts. User observations are separate human evidence, not
automatically converted to per-image labels.

## Keep three questions separate

1. **Did the deployed student improve?** Compare frozen learned checkpoints on
   every fixed prompt/noise pair. Primary mechanism control: verified+retention
   versus random+retention, alongside comparison with the same starting S1.
2. **What did the teacher action change?** Compare cached zero/half/full local
   interventions. These are intervention outcomes, not trained-student results.
3. **Can the training selector recognize a useful action?** Audit acceptance,
   damage, abstention and coverage. This does not replace final-model evaluation;
   selector calibration need not block measuring all existing student outputs.

## Metric contract

| Axis | Metrics | Interpretation |
|---|---|---|
| Text | Original upstream OCR reward; separately named exact-match/character error diagnostics | Preserve normalization, word extraction and extra-text handling. Do not rename a custom exact-match rule as the original OCR benchmark. |
| Composition | Native GenEval strict, split by count/color/position and other available benchmark tags | Missing detections can be scorer errors. Our custom joint prompts are not the standard GenEval benchmark. |
| Whole task | Joint task success, protection violations, unknown/error coverage | Keep the visibility proxy and its shared dependence on detection explicit. Do not silently count unavailable evidence as confirmed damage or success. |
| General quality/matching | PickScore, CLIPScore, HPSv2.1, Aesthetic, ImageReward | Correlated proxies, not independent human truth or exact spelling/counting checks. |
| Efficiency | GPU-hours and separately inference/teacher/reward/optimizer counts | Include failed jobs and evaluation costs; optimizer steps alone are not a fair compute comparison. |

Historical OPD PickScore is divided by26; historical OPSD evaluation used raw
PickScore. Report both with explicit names, but count them as one metric. Do not
sum raw heterogeneous scales. Preserve native preprocessing and truncation.
In particular, this installed ImageReward library truncates at35 tokens; audit
whether requested words remain in its actual input. Fixing/reordering prompts
would define a different evaluation, not an invisible metric repair.

## Execution order

1. Phase40: five general metrics on all Phase33 and Phase34 cached images,
   explicit cached OCR and available native-view object diagnostics. Report
   missing Phase33 automatic object scores as unavailable, not assistant labels.
2. Complete the missing whole-cohort native GenEval/object evaluation for
   Phase33, retaining detector boxes, thresholds, strict/reward distinctions and
   color-classification evidence. Use original full records for metadata; do not
   infer requirements from selected examples. Keep original/native OCR and our
   custom OCR results in separately named columns.
3. Re-evaluate the fixed checkpoints on a fresh, frozen benchmark cohort with
   matched prompt seeds, resolution, CFG, sampler/steps and initialization. Add
   raw backbone and native OPD/OPSD baselines only when their actual checkpoints
   and training/protocol provenance are available. S1 is not a native OPD baseline.
4. Once same-protocol benefit exists, run a fixed sampler/step transfer matrix.
   Scores on old DrawBench Flow40 and our custom Flow10 images cannot be directly
   subtracted. Repeated training seeds are needed for stable-method claims.

Full cohorts and contrasts are frozen before scoring. Keep paired seeds together
and bootstrap prompt clusters within task family. Existing reused development
sets support descriptive analysis; they do not become new independent tests
because a new metric is added. Report all predeclared metrics and failure/unknown
counts, including disagreement with the preferred method. No best-metric or
best-checkpoint selection on the same evaluation outputs.

Default resource policy remains zero agent image reads, local pinned evaluators,
content/prompt/model/code-bound cached predictions, compact aggregate inspection,
and optional user-facing galleries. Old observations and completed archives stay
unchanged. This policy does not require completing exhaustive human annotation
before computing established quantitative metrics.

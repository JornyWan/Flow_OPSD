# Phase27 prospective lexical/noise confirmation — frozen before generation

Fixed short_correction_verifier_v1 from Phase26, unchanged SPEC/source/scorer weights. 60 new target words ×2 new seeds =120 pairs,20 prompts in each color/count/relation family. Familiar Phase24 scene templates/objects/backgrounds are retained; words and random noise are fresh. This is prospective lexical/noise confirmation, not structural or cross-domain generalization. Report exact prompt overlap, word overlap, word-masked template overlap and seed provenance before generation. Do not replace prompts/seeds after viewing any result.

## Models, actions and validation

SD3.5-M plus the same frozen G and O adapters; Flow10 at512px, G CFG1, O CFG4.5, fp16 predictions with FP32 Euler updates. Generate full G×10 and G×3→O×2→G×5 with shared first3G actions. After intervention, recompute each G action at the actual branch state. No training. Existing generic initial student is zero-output LoRA only for the backend bookkeeping, never optimized or used for the route. G remains the behavior reference, not a newly trained student.

Before new images, replay both a prior G trajectory and prior OCR2 trajectory from CPU-seeded noise, checking all states/actions/PNG and context equality. Save all new states/actions/noise hashes and both endpoint PNGs. Release diffusion-model memory before loading scorers. Replay the old first G/O2 pair through the unchanged Phase26 scorer and require exact raw evidence and decision; then score all240 new endpoints. Keep original per-clause outputs, confidence, all extra text rows and object boxes. Reject any unknown, any protected true→false, or no confirmed false→true, including visibility as a clause. Threshold .8 and all other settings unchanged. The worker contains no new reference labels.

Total diffusion counts including replay: G1818, O242, CFG-expanded2302 samples,242 decodes,122 positive/negative text encodings. New-pair counts G1800/O240/2280 forward examples/240 decodes; shared prefix actions are newly computed once per pair. Verifier:240 new endpoint images,480 whole-image OCR views,480 GenEval clause views, data-dependent row rereads; separately include2 replay endpoints/4OCR views/4GenEval views and their rereads. Zero optimizer updates/backward queries. Scorer/teacher assets retain the established immutable provenance checks.

## Observation and analysis

Freeze all scorer predictions by result.json SHA before preparing anonymous A/B review. Inspect every new image at native512px under the unchanged Phase21+ full-scene rubric; no reference labels inherited. Do not inspect method key, auto scores, decisions or aggregate outcomes until all240 new observations are frozen. Familiar model style limits blinding; label source remains implementing-assistant provisional review, not independent human truth. Ambiguity is unknown, never erased or automatically treated as failure. Same image/requirements must have identical semantic labels. Preserve initial draft and any pre-key revisions.

Primary comparison: verified selection versus full G on complete-scene correctness. Report full G, fixed OCR2 and verified selection, with text/task/visibility, known harm, unknown, acceptance, safe precision/recall and all individual outcomes. Denominator120 pairs and60 prompt clusters. Resample prompts within each family, keeping both seeds paired,20,000 draws,seed2401; intervals are95% two-sided. Verified versus fixed OCR2 and fixedOCR2 versus G are secondary descriptive comparisons, not separately selected primary tests.

Unknown sensitivity: for different endpoint image hashes, use selected known complete minus comparator possibly complete; for identical images (including G fallback), difference is exactly0. Preserve the primary known-success metric separately. Report accepted neutral cases in the precision denominator and harm/unknown overlap without collapsing labels.

Predeclared confirmation gate, all required: at least12 accepted, at least10 reference-confirmed safe repairs, safe precision≥80%, zero known harmful accepts, unknown accepts≤10% of accepts, primary complete-scene95% lower bound>0, and paired unknown-adverse95% lower bound>0. Zero accepts fails. Passing warrants a short learning study only, not a claim of student improvement, original OPD superiority or sampler robustness. Failure remains a result; no automatic threshold changes, new candidate search or long training.

## Budget and execution

One L40,4CPUs,96GB host RAM,maximum30min and0.5 allocated GPU-hour including failed attempts. No automatic retry/requeue. Either minilab-gpu or spare-capacity preempt_gpu/low is permitted; record exact allocation/queue and all scheduling changes. Do not touch unrelated user jobs. If interrupted or replay fails, preserve partial artifacts and diagnose before any new attempt within remaining budget.

After scorer output freeze, complete all review, statistics, paired-image report and artifact checks. Archive source/environment/data, outputs, costs, frozen observations and final completion manifest. No model update or new phase is automatically launched.

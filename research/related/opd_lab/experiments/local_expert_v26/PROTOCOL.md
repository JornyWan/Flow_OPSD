# Phase26 frozen protocol: executable verification of short corrections

Freeze before scorer execution. All 72 Phase25 pairs (36 prompts × 2 seeds) are observed development data, not independent confirmation. The only candidate is G×3→O×2→G×5, with G×10 fallback. No case selection, new generation, teacher query, optimizer update, or relabeling. Reference labels are provisional implementing-assistant observations, not independent human truth.

## Inputs and implementation

All 144 endpoint PNGs, shared noise IDs, prompts/seeds and the Phase25 completion manifest are hashed. Worker inputs contain no reference observations. Score all images before joining the old reference labels. Snapshot source, upstream, environment, input images and protocol; fully hash actual PaddleOCR, Mask2Former and OpenCLIP scorer weights. Preserve original raw OCR, object boxes/scores and reasons. Reuse the existing official evaluator through the frozen runtime snapshot; do not change its thresholds or detector.

Primary scorer: short_correction_verifier_v1, based on endpoint_evidence_v2_dev1. OCR uses native and 2× full-image detection, row assembly and both sets of cropped 2× row rereads. Confidence threshold .8; all four readings must agree on exact correctness, otherwise unknown. Empty/low-confidence evidence is unknown. To match the Phase21+ rubric for single-word signs, remove whitespace within each row only after NFKC/casefold. Preserve all other characters and every row; never certify a target substring or discard extra text. This cannot prove absence of text that OCR never detects, and same-row word boundaries may be ambiguous. Retain the old whitespace-preserving text verdict and selection as a predeclared diagnostic, not a second tuned method.

Objects: original and 16px white-border GenEval strict judgments must agree, else unknown. Visibility: the existing required-count/bbox 3px interior heuristic, corrected for the border offset, with consensus across views. It is a proxy, not an actual occlusion/completeness detector. Audit its false certification separately.

Use scene_compare across text, the task clause and full_visibility. Accept OCR2 only if at least one definite false→true, no true→false and no unknown in either endpoint. Visibility repair counts. Otherwise select G. No confidence ranking, special cases, label-dependent threshold or post hoc policy substitution.

## Analysis locked before outputs

Report acceptance/72; safe precision (reference-confirmed safe accepts / all accepts); safe recall (/ all reference safe opportunities); harmful accepts / all accepts and / all known-harm opportunities; unknown accepts and known neutral accepts. Keep harm and unknown flags separate where they overlap. Report all selected endpoint clause/complete-scene counts, pairwise known-complete gains/losses versus G and fixed OCR2, 36-prompt cluster bootstrap differences within the three families (20,000 draws, seed2401), and adverse treatment of unknowns (selected known complete minus comparator possibly complete). All intervals are descriptive.

Show each condition's true/false/unknown confusion, all pair decisions/rejection reasons and both endpoint raw readings. Report both full-cohort counts and family/seed breakdowns, never substitute a favorable subgroup.

Development readiness uses the earlier Phase13 scale: at least 4 known safe repairs retained, safe precision ≥80%, zero known harmful accepts, and unknown accepts ≤10% of accepts. Zero accepts fails readiness. This gate is a development decision only; even a pass requires independent new-data confirmation and does not authorize a claimed OPD gain. Do not automatically train. Failures remain archived; any subsequent scorer changes require a new version.

## Cost and failure handling

One L40, 4 CPUs, at most 30 minutes and 0.5 allocated GPU-hour including failed attempts. Expected: 144 evidence images, 288 full-image OCR views, 288 GenEval clause views, data-dependent row rereads (at most 9216). OCR on CPU, object/CLIP evaluator on GPU. Zero diffusion-model calls, decodes, generated images or optimizer updates. Log actual row counts and scheduler cost, including initialization. No automatic retry. Assert source hashes before scoring; fail rather than silently skip images. Freeze evidence.jsonl by hash in result.json before reference-label analysis. Archive tests, job command, outputs, report and final audit.

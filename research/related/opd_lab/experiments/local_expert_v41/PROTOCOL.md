# Phase41: complete cached Phase33 task characterization

Claim: C2 evidence completion, not a new efficacy confirmation. Only change:
add pinned native task evaluation to ALL 600 existing Phase33 model images.
No image generation, training, agent image viewing, threshold tuning, checkpoint
choice, assistant-label imputation, or alteration of Phase33–40 archives.

Five fixed arms, 60 prompts × two paired seeds × five models. Primary descriptive
comparison: verified_retention minus random_matched_retention on automatic
joint exact OCR AND native GenEval strict. Secondary: verified minus random,
verified_retention minus verified, random_retention minus random, verified and
verified_retention minus S1. S1 is a historical common trained starting model,
not native OPD/backbone. One trained checkpoint per arm; no training-repeat CI.

Native OCR reward preserves upstream substring acceptance / ASCII-space removal
/ lowercasing / clipped edit distance. Raw recognized text, boxes and confidence
are saved. Separately report exact equality with the same normalization and
uncapped character error rate; empty recognized text is exact=false, upstream
errors explicitly recorded. Native reward=1 does not imply exact=true.
Reuse Phase36 original baseline tri-state OCR unchanged, clearly labelled, and
combine it with newly computed native GenEval only as a separate diagnostic.

GenEval preserves upstream native original-image thresholds and both strict and
non-strict paths; capture raw prefilter detections, filtered boxes, color logits,
labels and lossless compressed masks used for classification. Observer hooks
return values unchanged; public API parity on one image per family precedes
scoring, and native OCR arithmetic replay must match every scored image.
Detector failure is not a human finding that an object physically disappeared.

Joint task here means declared OCR/GenEval clauses, not full scene correctness:
fully-visible/separated wording is not completely measured by those metrics.
Visibility proxy and old assistant labels must not be silently folded in.
These are familiar, already exposed custom prompts, NOT the official GenEval
or DrawBench benchmark. Shared model errors persist; no independent human
validation is added. Quality scores copied from Phase40 with exact image hashes.

Report all rows and families, exact/strict/reward/CER, unknowns/errors, and harms
to previously true OCR/object clauses relative to paired S1 with denominators.
Use prompt-cluster, family-stratified bootstrap 20,000 draws, seed41041,
descriptive 95% intervals. Unknowns retain tri-state values, success lower/upper
bounds and paired adverse sensitivity, not fabricated false labels. No
multiplicity-adjusted efficacy or sampler/generalization claims.

Resource: one L40S requested as Slurm gpu:l40:1, <=30 allocated minutes including
failed attempts; no automatic requeue. CPU OCR is within this allocation. Full
600-image scoring needed for completion; no favorable subset. Results decide
whether the existing quality signal also has task support, then inform the
already planned single-expert fixed-versus-verified candidate experiment.

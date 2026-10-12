# Phase5: measurement calibration and small-action re-verification

Frozen before phase5 outcomes. External-expert OPD remains the main method.
No optimizer updates or validation/test evaluation in this phase.

## Fixed cohorts and observations

Development: all 44 phase4-reviewed noop/selected-target images, with the original
frozen assistant labels, including nulls. Diagnostic matrix: all 286 phase4 actions
(the 44 development images are endpoints of this matrix, not additional independent
samples). All old images/scores/labels remain immutable.

Calibration holdout: 16 new base images, four per TRAIN task family. Exclude phase4
accepted prompt IDs and phase3 review IDs, then choose by SHA256 of
['phase5_calibration', id]. Use new CPU noise stable_seed(314159,
'phase5_calibration', id), original phase3 frozen behavior, SD3.5-M/fp16/512/flow10
and the existing CFG. Prompts have been used in earlier training; these are new
noise realizations held out from this measurement-rule check, NOT a new benchmark
or held-out model efficacy set. No target selection on this cohort.

Before reading calibration scores, show its 16 images in hash order with prompts,
hide scores, and record all explicit clauses plus full visibility as true/false/null,
literal visible text and reasons. Freeze observations/time/hash before score reveal.
The observer is the implementing assistant; these provisional labels are not
independent human ground truth. Preserve later caveats separately.

## Predeclared OCR checks (audit only)

Use existing pinned Paddle OCR and the separately pinned Tesseract 4.1.1 full-image
reader, PSM11/OEM1/eng, without prompt hints or character whitelist. Reuse one
Tesseract call per unique PNG hash; report actual calls and elapsed CPU time.

For our single alphabetic target words, compare four fixed predicates:

1. Existing v1: NFKC/casefold/collapse whitespace, exact whole-string match.
2. Paddle compact: same reader, remove whitespace only, exact whole-string match.
3. Tesseract compact: same whitespace-only normalization and exact match.
4. Both compact: conjunction of predicates 2 and 3.

Removing whitespace does not remove letters, digits, punctuation, duplicated words
or registration symbols. This compact rule is restricted to single alphabetic
targets and is not a generic OCR normalization. Do not adjust PSM/thresholds or
select rules after seeing the new calibration images. Record TP/FP/TN/FN, precision,
recall, unknown cases and disagreements on development and calibration separately.
No metric becomes a training reward merely because it wins this small audit.

## Scene trace and coverage

On all 44 development images and 16 calibration images, run the SAME pinned
GenEval detector once more and save all returned boxes/confidences for requested
classes, before confidence filtering. Record counts retained by the upstream
0.9 counting / 0.3 other thresholds, maximum16/class, NMS1.0. Verify those constants
in the loaded scorer closure; fail on unexpected implementation changes. Compare
count-only strict decisions with original scores; report every mismatch.

Record whether a retained box is within one pixel of an image border as a
descriptive clipping flag. This is neither independent detection nor a complete
visibility/occlusion check; absence of the flag is not proof of full visibility.
Color and relation correctness still require their other upstream components.
Use traces to explain errors, never to silently lower thresholds or change v1.

## Existing action matrix under alternate measurement

Read every phase4 image with Tesseract; preserve all original v1 values. Separately
recompute diagnostic utilities with only OCR changed: compact normalized edit
progress for rules2/3, conjunction and min(progress2,progress3) for rule4; keep
GenEval and joint utility weight unchanged. Report all five beta curves and four
actual/projected checkpoint contrasts. For each beta separately, count positive
gains that preserve every clause satisfied by that rule's noop. No new target
selection, no filtering of the 22 prompts, no retraining. These are sensitivity
checks using shared GenEval, not independent quality estimates.

Paired prompt bootstrap2000/seed0, no multiplicity correction, selected TRAIN
states and one training seed. Do not treat 286 images as independent prompts.

## Budget and completion

One L40-class allocation, maximum20 minutes, generating16 images and60 detector
traces. CPU OCR on302 image records (fewer unique hashes) and16 image observations.
Verify all input/output hashes, complete all cohorts/rules, and report actual costs.
If measurement errors remain, do not promote a new reward or expand training.

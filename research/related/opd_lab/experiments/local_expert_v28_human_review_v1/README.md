# Phase28 independent human annotation package

Download review.html and open it directly. All108 images are embedded. Each of36 cases shows three anonymously shuffled methods. No controls have prefilled judgments. You can answer incrementally; download JSON before changing browser or machine. Put the real exported file in human_rating/ with its original unique name.

This covers the complete development cohort, not a favorable subset. Previous result/image exposure is possible. Existing assistant observations and the Phase28 archive remain unchanged. Missing, uncertain and incorrect answers have different meanings. Overall aesthetic preference is optional and separate from constraint satisfaction.

Analyze actual returned answers with scripts/analyze_learning_human_review.py --phase28 (use --phase 28 as separate command arguments), --answers pointing to the export and --output pointing to a new directory. Browser checks generated a synthetic export under runs/le28_human_browser_001; it is not human feedback and must never enter analysis.

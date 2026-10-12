# Phase21 observation provenance

The implementing assistant inspected identity-blind native512px images, arranged
in four sheets per case. Development: all48 sheets/all156 views inspected before
freezing observations and before opening the identity key. Individual native
images were additionally checked for small text/crops/class ambiguity:
case2 E/J; case3 C/E/I; case6 C/H/K; case8 D/F/I; case9 A/I/K;
case10 B/F; case11 B/C/F/J/K/M. Case11 sheet pages1–3 were also re-opened
after a tool output was truncated. A truncated response was not treated as proof
of inspection; the six native views were re-opened in smaller groups.

The explicit drafts, frozen labels, key hashes and rubric hash are retained in
runs/le21_development_review_001. Unclear text, jar/bottle identity, barely visible
objects and ambiguous boundary crops remain null. No identity-based label changes
were made after freezing. These are provisional assistant annotations, not user
ratings or an independent-human measurement.

Execution note: two initial CPU postprocessing invocations (tensor verification
and review preparation) ran before development result.json was available and
returned FileNotFoundError. Neither wrote an output or launched a GPU job. After
waiting for result.json and removal of RUNNING, both completed successfully.
This was an orchestration timing issue; there was no failed GPU attempt or retry.

Confirmation observations and any later independent user review must be recorded
separately; frozen development labels must never be overwritten.

Confirmation is now complete: all48 sheets/all156 views were inspected before
freezing and joining the key. Additional native checks: case1 C/F; case2 C/I;
case3 A/L; case5 B; case6 I/F; case7 A; case10 M; case11 B/H. Both cohorts use
the same frozen rubric. All drafts remain available alongside frozen observations.
There have been no post-key label edits. Independent human observations have not
been received for Phase21.

Confirmation also briefly showed stale result-existence/RUNNING status after
Slurm had completed. Explicitly listing the run directory showed result.json and
absence of RUNNING; then both CPU validation and review preparation completed.
No confirmation verifier/preparation was run prematurely and no GPU was retried.

The targeted human audit selects all confirmation cases with complete-scene
disagreement or known harm among noop, label-oracle, i1_geneval@0.5 and i3_ocr@1.
It contains cases1/2/4/9/12 (20 views), newly randomized. These two fixed modes
were selected after seeing confirmation results; the audit is exploratory and
does not replace development_selection.json or the preregistered comparison.

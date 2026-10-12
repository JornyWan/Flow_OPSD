# Phase4: tuple/list verdict comparison

Job 1048104 completed. Its immutable result reports `same_images=132` and
`same_verdicts=0`. The latter is a comparison-flag bug, not changed verifier values.
`Verdict.as_dict()` preserves tuples for satisfied/progress; JSON references use
lists. The runtime compared them directly: `(True,) != [True]`.

The affected frozen source SHA256 is
`1a7eedf61557e3ffb61594ecea4e6a18d1f5315979d4d99f405aacf4822f4d1d`.
The first CPU report rejected the inconsistent flags. Independent exact comparison
of serialized valid/satisfied/progress/utility/joint_success confirms **132/132**
matches, as do all 132 PNG hashes. All 286 saved action tensors/images and 88
geometry rows also pass recomputation/integrity checks.

Original runtime source, logs, flags and scores remain unchanged. The reporter
permits this false flag only for the audited source hash above, checks other
fields, and lists each correction in `integrity.comparison_flag_errors`.
Unexpected discrepancies fail. Future runtimes use `same_verdict()` to normalize
tuple/list representation while preserving exact values; a regression test rejects
real changes in every field, including progress changed by 1e-8.

No GPU rerun or old-score substitution was needed. The failed CPU report consumed
no GPU. The full 48-test suite passed before the fix; four action-probe tests passed
after it, including the new regression. The corrected report then passed.

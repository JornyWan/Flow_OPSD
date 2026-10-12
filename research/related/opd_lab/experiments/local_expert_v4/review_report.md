# Accepted-target image review

Descriptive implementing-assistant review, not independent human truth. All 22 selected TRAIN pairs, with possible prior phase3 exposure. Known visual disagreements require adjudication; unchanged binary clauses alone do not invalidate progress gains.

Preferences: {'tie': 7, 'target_better': 10, 'noop_better': 2, 'uncertain': 3}

| Case | Target/noop utility delta (v1) | Visual preference | Target known failed clauses | Target full visibility |
|---|---:|---|---|---|
| 01 | 0.562500 | tie | none recorded | True |
| 02 | 0.533333 | target_better | none recorded | False |
| 03 | 0.642857 | target_better | none recorded | False |
| 04 | 0.033333 | tie | none recorded | None |
| 05 | 0.250000 | noop_better | text | None |
| 06 | 0.750000 | target_better | count | False |
| 07 | 0.541667 | tie | text | True |
| 08 | 0.104167 | tie | relation | False |
| 09 | 0.750000 | target_better | none recorded | True |
| 10 | 0.666667 | uncertain | none recorded | False |
| 11 | 0.662500 | target_better | none recorded | False |
| 12 | 0.750000 | target_better | none recorded | False |
| 13 | 0.750000 | tie | count | False |
| 14 | 0.583333 | target_better | none recorded | True |
| 15 | 0.666667 | noop_better | text | False |
| 16 | 0.074074 | tie | none recorded | True |
| 17 | 0.541667 | target_better | none recorded | True |
| 18 | 0.550000 | uncertain | text | True |
| 19 | 0.666667 | target_better | none recorded | False |
| 20 | 0.150000 | uncertain | none recorded | True |
| 21 | 0.600000 | target_better | none recorded | None |
| 22 | 0.625000 | tie | count | False |

An empty failure list may contain unknown clauses; inspect summary.json. Full visibility, text color and sign placement are not fully represented in v1. This review does not alter training selection or labels.

Known visual-clause failure despite target v1 joint success: cases [6, 7, 13, 15, 18, 22].
Visibility failure despite target v1 joint success: cases [2, 3, 6, 10, 11, 12, 13, 15, 19, 22].

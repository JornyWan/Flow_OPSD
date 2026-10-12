# Phase6: independent model audit

Audit only, provisional implementing-assistant references; no human ground truth or policy efficacy claim. Joint encoded excludes style and full visibility.

| Cohort | Reader | Clause | TP | FP | TN | FN | Abstain +/- | Unknown reference | Coverage | Recall incl. abstain |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| development | v1 | text | 23 | 3 | 6 | 11 | 0/0 | 1 | 1.000 | 0.676 |
| development | v1 | color | 12 | 0 | 0 | 2 | 0/0 | 6 | 1.000 | 0.857 |
| development | v1 | joint_encoded | 10 | 6 | 13 | 9 | 0/0 | 6 | 1.000 | 0.526 |
| development | v1 | count | 9 | 3 | 4 | 1 | 0/0 | 3 | 1.000 | 0.900 |
| development | v1 | relation | 5 | 0 | 5 | 2 | 0/0 | 0 | 1.000 | 0.714 |
| development | qwen_exact | text | 33 | 6 | 3 | 1 | 0/0 | 1 | 1.000 | 0.971 |
| development | qwen_exact | color | 12 | 0 | 0 | 0 | 2/0 | 6 | 0.857 | 0.857 |
| development | qwen_exact | joint_encoded | 16 | 7 | 8 | 1 | 2/4 | 6 | 0.842 | 0.842 |
| development | qwen_exact | full_visibility | 9 | 13 | 6 | 0 | 5/5 | 6 | 0.737 | 0.643 |
| development | qwen_exact | count | 6 | 3 | 2 | 0 | 4/2 | 3 | 0.647 | 0.600 |
| development | qwen_exact | relation | 7 | 2 | 3 | 0 | 0/0 | 0 | 1.000 | 1.000 |
| development | qwen_compact_sensitivity | text | 33 | 6 | 3 | 1 | 0/0 | 1 | 1.000 | 0.971 |
| development | qwen_compact_sensitivity | joint_encoded | 16 | 7 | 8 | 1 | 2/4 | 6 | 0.842 | 0.842 |
| exposed_calibration | v1 | text | 6 | 0 | 7 | 2 | 0/0 | 1 | 1.000 | 0.750 |
| exposed_calibration | v1 | count | 5 | 0 | 0 | 1 | 0/0 | 2 | 1.000 | 0.833 |
| exposed_calibration | v1 | color | 8 | 0 | 0 | 0 | 0/0 | 0 | 1.000 | 1.000 |
| exposed_calibration | v1 | joint_encoded | 5 | 0 | 8 | 2 | 0/0 | 1 | 1.000 | 0.714 |
| exposed_calibration | v1 | relation | 0 | 0 | 4 | 0 | 0/0 | 0 | 1.000 | NA |
| exposed_calibration | qwen_exact | text | 7 | 3 | 4 | 1 | 0/0 | 1 | 1.000 | 0.875 |
| exposed_calibration | qwen_exact | count | 6 | 0 | 0 | 0 | 0/0 | 2 | 1.000 | 1.000 |
| exposed_calibration | qwen_exact | color | 8 | 0 | 0 | 0 | 0/0 | 0 | 1.000 | 1.000 |
| exposed_calibration | qwen_exact | joint_encoded | 6 | 1 | 7 | 1 | 0/0 | 1 | 1.000 | 0.857 |
| exposed_calibration | qwen_exact | full_visibility | 4 | 7 | 4 | 0 | 0/1 | 0 | 0.938 | 1.000 |
| exposed_calibration | qwen_exact | relation | 0 | 1 | 3 | 0 | 0/0 | 0 | 1.000 | NA |
| exposed_calibration | qwen_compact_sensitivity | text | 7 | 3 | 4 | 1 | 0/0 | 1 | 1.000 | 0.875 |
| exposed_calibration | qwen_compact_sensitivity | joint_encoded | 6 | 1 | 7 | 1 | 0/0 | 1 | 1.000 | 0.857 |
| new_calibration | v1 | text | 10 | 1 | 4 | 0 | 0/0 | 1 | 1.000 | 1.000 |
| new_calibration | v1 | relation | 0 | 0 | 4 | 0 | 0/0 | 0 | 1.000 | NA |
| new_calibration | v1 | joint_encoded | 4 | 1 | 8 | 1 | 0/0 | 2 | 1.000 | 0.800 |
| new_calibration | v1 | count | 5 | 0 | 2 | 0 | 0/0 | 1 | 1.000 | 1.000 |
| new_calibration | v1 | color | 5 | 0 | 0 | 1 | 0/0 | 2 | 1.000 | 0.833 |
| new_calibration | qwen_exact | text | 10 | 1 | 4 | 0 | 0/0 | 1 | 1.000 | 1.000 |
| new_calibration | qwen_exact | relation | 0 | 3 | 1 | 0 | 0/0 | 0 | 1.000 | NA |
| new_calibration | qwen_exact | joint_encoded | 4 | 3 | 5 | 0 | 1/1 | 2 | 0.857 | 0.800 |
| new_calibration | qwen_exact | full_visibility | 2 | 7 | 2 | 1 | 0/4 | 0 | 0.750 | 0.667 |
| new_calibration | qwen_exact | count | 4 | 0 | 1 | 0 | 1/1 | 1 | 0.714 | 0.800 |
| new_calibration | qwen_exact | color | 5 | 0 | 0 | 0 | 1/0 | 2 | 0.833 | 0.833 |
| new_calibration | qwen_compact_sensitivity | text | 10 | 1 | 4 | 0 | 0/0 | 1 | 1.000 | 1.000 |
| new_calibration | qwen_compact_sensitivity | joint_encoded | 4 | 3 | 5 | 0 | 1/1 | 2 | 0.857 | 0.800 |

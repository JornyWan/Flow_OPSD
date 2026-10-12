# Measurement audit

Fixed measurement checks; provisional implementing-assistant labels; no independent human truth or deployed-policy efficacy claim. Matrix shares GenEval; calibration uses previously trained prompts with fresh noise.

| Cohort | Rule | TP | FP | TN | FN | Unknown | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| development | v1 | 23 | 3 | 6 | 11 | 1 | 0.8846153846153846 | 0.6764705882352942 |
| development | paddle_compact | 25 | 3 | 6 | 9 | 1 | 0.8928571428571429 | 0.7352941176470589 |
| development | tesseract_compact | 8 | 1 | 8 | 26 | 1 | 0.8888888888888888 | 0.23529411764705882 |
| development | both_compact | 7 | 1 | 8 | 27 | 1 | 0.875 | 0.20588235294117646 |
| calibration | v1 | 6 | 0 | 7 | 2 | 1 | 1.0 | 0.75 |
| calibration | paddle_compact | 6 | 1 | 6 | 2 | 1 | 0.8571428571428571 | 0.75 |
| calibration | tesseract_compact | 2 | 0 | 7 | 6 | 1 | 1.0 | 0.25 |
| calibration | both_compact | 2 | 0 | 7 | 6 | 1 | 1.0 | 0.25 |

Unknown labels are not silently counted as correct or incorrect. Rule agreement is not truth. Border flags do not cover occlusion or all clipping. Full counts/traces/intervals are in summary.json.

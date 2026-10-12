# Phase67：固定初始状态的逐段诊断

64训练题，非独立评测，不证明泛化收益。

| 模式 | joint/64 | OCR/64 | 结构/64 |
|---|---:|---:|---:|
| base | 10 | 27 | 16 |
| teacher | 9 | 29 | 19 |
| all_local_action | 12 | 29 | 18 |
| all_updated_suffix | 8 | 23 | 23 |
| all_full_student | 5 | 8 | 37 |
| verified_local_action | 10 | 28 | 18 |
| verified_updated_suffix | 7 | 18 | 19 |
| verified_full_student | 0 | 4 | 10 |

全部五质量、配对区间、状态子集、局部误差和账务见 summary.json。阶段差可能存在交互，不强行归为单一原因。

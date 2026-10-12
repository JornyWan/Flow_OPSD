# 0008 — GuardedOPSD 前置条件检验:teacher-KL 能否作为能力退化的 proxy

Date: 2026-08-29
Status: 必要条件通过;充分性未验证(需新增逐 teacher KL 日志)
Author: Claude

## 背景
0006 §7 把这条列为**最关键的 falsification 条件**:

> 论文必须先证明 teacher-relative KL 是真实旧能力退化的有效 leading indicator。
> 如果相关性弱,应重新设计 constraint proxy,而不是继续优化 QP。

本条目用**现有数据**做第一轮检验,不消耗 GPU。

## 数据可得性(先查清,避免设计错实验)
MOPD 臂(`poe15`)的 wandb 记录里:
- **有** 逐 epoch 的在线能力测量:`monitor_ocr` / `monitor_geneval` / `monitor_pickscore`(各 800 点);
- **只有聚合** `train_reverse_kl`,**没有逐 teacher 的 KL**。

=> 这本身是一个必须先解决的实现前提:**GuardedOPSD 需要 per-teacher KL,而当前代码不记录它。**

## 结果(n=800,poe15 全程)
| 相关性 | Pearson |
|---|---|
| `train_reverse_kl` vs `monitor_ocr` | **−0.701** |
| `train_reverse_kl` vs `monitor_geneval` | **−0.698** |
| `train_reverse_kl` vs `monitor_pickscore` | −0.647 |

KL 越大、能力越低,三轴一致,量级 −0.65~−0.70。
=> **"teacher-KL 与能力退化相关"这一必要条件成立**,proxy 不是无关量。

## 但不能据此认为充分 —— 三个限制
1. **聚合 KL ≠ 逐 teacher KL。** GuardedOPSD 要的是"对 OCR teacher 的 KL 约束 OCR 能力"。
   这里三个能力与**同一个**聚合 KL 都相关约 −0.7,**无法区分**
   "KL 分别预测各自能力" 与 "三者只是随训练进度共同变化"(共同趋势混淆)。
2. **相关 ≠ leading indicator。** 0006 §7 要求 KL 变化能**预测**能力变化,
   应检验**差分序列**的相关性与领先阶数,而非水平序列。
3. **数据来自多 teacher 蒸馏臂**,KL 是其训练损失;
   GuardedOPSD 里 KL 是**约束**而非目标,动力学不同,外推需谨慎。

## 因此下一步的实现顺序(修正 0005 Phase B 的优先级)
**必须先加 per-teacher KL 日志,再做任何约束标定。** 具体:
1. 在 MOPD/OPSD 训练路径记录 `kl_teacher_{ocr,geneval,pickscore}` 逐 epoch 值;
2. 用**差分序列**重做相关性,并检验领先阶数(KL 的变化是否先于能力变化);
3. 只有当"对 OCR teacher 的 KL"与"OCR 能力"的偏相关(控制训练进度后)仍显著,
   才认为 proxy 选对。

在此之前不应进入 QP / primal-dual 的标定 —— 否则是在一个未验证的 proxy 上做精细优化。

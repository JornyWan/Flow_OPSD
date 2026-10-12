# 0009 — 加 per-teacher KL 日志 + 修 0007 的分支步数不对称

Date: 2026-08-29
Status: 已实现;**但发现 RTC 路径不可达,需在 GuardedOPSD 实现时一并处理**
Author: Claude

## 做了什么

### ① per-teacher KL 日志(0008 指出的前置条件)
发现 `per_step_kl_per_teacher_scalar`(形状 `[B, n_teachers]`)**本来就被算出来了**
(`train_sd3_opd.py:2886`),只是在下一行被 `.sum(dim=1)` 折叠、从未单独记录。
新增只**读**该张量、按 teacher 的 `name` 字段(pickscore/ocr/geneval)分别记录:
```python
info[f"kl_teacher_{name}"].append(per_step_kl_per_teacher_scalar[:, i].mean().detach())
```
**不改变任何计算**,备份 `scratchpad/train_sd3_opd.bak_klteacher`。

### ② 分支生效步数(0007 指出的 10x 标定偏差)
新增 `info["branch_steps_direct"]`,使 epoch 级的分支梯度比可**直接读出**,
而不必从"direct 在 1 个 j、retention 在 ~10 个 j"去推。

## ⚠️ 验证时发现:新日志对 RTC 路径**不可达**

| 位置 | 行号 |
|---|---|
| `if rtc:` 分支开始 | 2394 |
| 该分支的 `continue` | **2535** |
| per-teacher KL 日志 | **2900** |

=> **RTC 臂在 L2535 直接 `continue`,永远执行不到 L2900 的日志。**

影响:
- **MOPD/POE 臂**(teacher 蒸馏路径):日志生效 ✓ —— 0008 的相关性分析可以升级为逐 teacher 版本;
- **RTC / OPSD 臂**:日志**不生效** ✗ —— 而 GuardedOPSD 恰恰要在这条路径上加约束。

这是典型的"不会崩、只会静默无效":加了日志、跑完 exit=0、metrics 里就是没有那几个键。

## 对实现顺序的影响

GuardedOPSD 的约束需要在 **OPSD 路径**上计算 per-teacher KL,而该路径当前:
1. 不走 teacher 蒸馏分支,故没有 `per_step_kl_per_teacher_scalar`;
2. 有自己的 `continue`,任何加在其后的日志都不可达。

因此 **per-teacher KL 必须在 GuardedOPSD 的约束计算里自行产生**,
而不是复用 MOPD 路径的量。这与 0005 Phase B 的方向一致
("Do this in the official DiffusionOPSD training path"),但明确了一个此前未点出的约束:
**不能指望复用现有 KL 张量,要新算。**

## 另一个记下来的坑
`read_wandb.py` 有键白名单(默认只显示 `monitor_*` 等)。
读新键必须加 `--keys=kl_teacher_ocr,...` 或 `--all`,
否则**看不到新键会误判为"没记上"**。

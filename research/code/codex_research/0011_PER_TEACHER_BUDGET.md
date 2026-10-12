# 0011 — 逐 teacher 能力预算 —— **作废(前提错误)**

> **结论:本条目的前提不成立,方法未被实现,两个作业已崩溃。保留全文以记录错误链条。**
>
> **真相**:`teacher_prev_sample_means` 的形状是 `[B, T_steps, max_teachers_per_slot, C, H, W]`,
> 第 2 维是**单个 slot 挂了几个 teacher LoRA**,不是任务 teacher 数。
> 正在跑的作业日志给出定论:`OPD: configured 3 teacher slot(s) using 3 unique teacher adapter(s) (K_max=1)`。
> 即 `per_step_kl_per_teacher_scalar` 形状为 **[B, 1]**,`.sum(dim=1)` 只是折叠一个元素 ——
> **0002 的约束本来就是逐路由 teacher 的**,不存在"对三个 teacher 求和"这回事。
>
> **后果**:我的 `gather(1, _own)` 用 0..2 的索引去索引 size=1 的维 ->
> `RuntimeError: CUDA error: device-side assert triggered`,`ptdirect=1035990` / `ptguard=1035991`
> 双双 exit=1(03:33,启动后约 75 秒)。我在崩溃前的那次汇报说"跑起来了",依据是前置检查通过 ——
> 那正是"把'没崩'当成'算对了'"。
>
> **方法论教训**:`code/0011_test_per_teacher.py` 全绿,但它测的是**我脑中的模型**(假定 T=3),
> 不是真实代码。建立在错误前提上的测试会通过并给出虚假信心。
> 正确顺序是**先确认张量真实形状**(一行 `grep K_max` 就够),再写测试。
>
> 下面保留原始(错误的)分析。

日期: 2026-08-29 03:32 | 作业: `ptdirect=1035990` / `ptguard=1035991`(ai-gpu14, 各 1×H100)

## 1. 起因:0010 的 benchmark 暴露出一个可诊断的失败

0010 跑到 epoch 16 的实测(见 `0010_GUARDED_PAIRED_RUN.md` 第 9 节):

- **OCR 被守住了**:无约束臂 0.942 → 0.888 → 0.913 → **0.800**;约束臂 0.924 → 0.947 → **0.966**。
- **但 GenEval 反而变差**(−0.04),尽管它与 OCR 同在 retention 槽 `(1, 2)` 里。
- **PickScore 跌破初始值**:0.922(init)→ 0.876,比无约束臂的 0.930 低 0.054 = **36σ**。
  约束不是"守住 reward 收益",而是把它反噬掉了。
- w=3 与 w=10 的 benchmark 几乎相同 ⇒ 代价是**结构性**的,不是调权重能解决。

## 2. 真因

读 0002 的实现:

```python
per_step_kl_scalar = per_step_kl_per_teacher_scalar.sum(dim=1)   # 对**所有** teacher 求和
_ref_kl = _ref_kl.mean(...).sum(dim=1)                            # 参考同样求和
_violation = per_step_kl_scalar - (1 + rel_tol) * _ref_kl - abs_tol
```

于是约束变成"到**全部三个** teacher 的总 KL 不得超预算"。后果:

1. 总量里含到 **PickScore teacher** 的 KL,而 reward 优化正是要抬高它
   → 预算被 reward 自己吃掉 → hinge 在 retention 样本上激活 → 把共享参数拽回去
   → **反噬掉 reward 收益**;
2. 聚合口径下谁在总和里占比大谁被保护 → OCR 守住、GenEval 得不到保护。

**这与 codex 0006 的设计不符。** 0006 明写:

> OCR prompts:只评估 OCR capability constraint;GenEval prompts:只评估 GenEval capability constraint。

即原本就要求**逐 teacher**。0002 的聚合是实现与设计的偏离。

## 3. 改动(唯一一处,与 0002 可直接归因)

每个样本只用**它自己路由到的**那个 teacher 的 KL,与同一 teacher 的冻结参考比较:

```python
_ref_kl_pt = _ref_kl.mean(dim=tuple(range(2, _ref_kl.ndim)))      # [B, n_teachers],不再 .sum(1)
_own = _tidx.clamp(min=0).unsqueeze(1)
_ref_kl = _ref_kl_pt.gather(1, _own).squeeze(1)
_cur_kl = per_step_kl_per_teacher_scalar.gather(1, _own).squeeze(1)
_violation = _cur_kl - ((1.0 + cr_rel_tol) * _ref_kl + cr_abs_tol)
```

日志量 `cr_current_teacher_kl` 同步改为 `_cur_kl`,否则日志与约束不是同一个量。

**前置确认**:`retain_w` 默认 0.0、`snr_weight` 默认 False,`codex_cr_smoke` 两者都没设,
故 cr 块处 `per_step_kl_scalar` 就是未经修改的逐 teacher 求和 —— 逐 teacher gather 量纲一致。

## 4. 验证(提交前全部实跑通过)

- `code/0011_test_per_teacher.py`:构造"OCR/GenEval 的 KL 完全没变、只有 PickScore 涨 5 倍"的情形。
  **聚合口径(0002)把 100% 的 retention 样本判为违约,逐 teacher 口径判 0%** ——
  量化证实 0002 的违约可以完全由非 retention teacher 造成。
- 补丁链 `pin base → 0002 → 0003 → 0011` 四步 hash 全部匹配
  (0002 后 `8d8e7552…`,0011 后 `775d5952…`,均已写死进 slurm 校验)。
- `py_compile` 通过;`0002_test_constraint_math.py` 仍 PASS(未破坏原不变量)。
- config 不变量 PASS,`bs=3 nproc=1 nbpe=24 gas=24 samples/epoch=72`。
- 运行期确认 `[CODEX-CR] direct_slot=0 retention_slots=(1,2) weights=(1.0,0.0)` / `(1.0,10.0)`。

## 5. 实验设计

| 臂 | `CODEX_CR_RETENTION_WEIGHT` |
|---|---|
| `ptdirect` | 0.0 |
| `ptguard` | 10.0 |

**为什么另跑一个 w=0 控制而不复用 0010 的 `gdirect`**:0010 三臂在 ai-gpu18 的 H200 上,
而 03:30 时 H200/A100 全满,只剩 ai-gpu14 的 2 张 H100。跨型号 GPU 比较会引入未控变量。
两臂都放 H100 使 0011 的结论**自足**,不依赖跨硬件比较。
(w=0 时 retention 项乘 0,故该控制臂与 0010 的 `gdirect` 数学上等价,只是硬件不同。)

## 6. 判读标准

0011 若成立,应当同时看到:

1. `cr_active_frac` 仍非零(约束仍在做功),但**违约不再由 PickScore 的漂移驱动**;
2. **PickScore 不再跌破初始值** —— 这是 0002 最刺眼的病征;
3. **GenEval 与 OCR 一起被守住**,而不是只守住 OCR;
4. 代价(aes/hps)显著小于 0010。

若 2/3 仍不成立,则说明问题不在预算口径,而在"用 teacher KL 代理能力"这个前提本身
(0008 只验了聚合相关 r≈−0.70 的**必要**条件,充分性从未验证)。

## 7. 状态

训练中。

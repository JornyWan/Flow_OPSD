# 0039 更正：主方向应从 base 开始，continuation 降为第二阶段

**日期：2026-08-30**  
**更正：代码底座仍选 Self-OPD，但主实验初始化改为 SD3.5-M base；DiffusionOPD Student continuation 不再是默认主线。**  
**状态：研究问题优先级修正；本条未提交训练任务。**

## 1. 为什么此前一直做 continuation

这是历史路径依赖，不是方法的必要条件：

1. 项目最初的问题是能否在 released DiffusionOPD multi-teacher Student 上继续加入 Pick/preference，
   同时不忘 OCR/GenEval；
2. 该 Student、teacher LoRA、评估和复现都已经齐全，能低成本研究 continual capability retention；
3. 相比重新做一篇 from-base reward optimization，continuation 曾经看起来是更独特的论文切口。

这些理由解释了我们为什么这样开始，但不足以证明它仍应是最有希望的主线。

## 2. 为什么现在不应把 continuation 放在第一优先级

### 2.1 两篇新方法的 headline success 都主要来自 base 起点

- DiffusionOPSD 的 public SD3.5 presets 默认 `train.lora_path=None`，从 SD3.5-M base 初始化 LoRA；
- Self-OPD 的公开默认配置同样从 SD3.5-M base 开始，论文另讨论 specialist warm-start；
- DiffusionOPSD 的 two-stage OPD 对比也明确把 shared student 从 SD3.5-M base 初始化。

因此它们证明的是：从一个尚有明显提升空间的 base，通过 local improvement operator 学出更好的
policy；它们没有证明一个已经 multi-teacher distill 过的 strong Student 仍有足够大的局部 Pareto
改进集合。

### 2.2 我们的结果已经显示 strong-Student neighborhood 很窄

从 DiffusionOPD Student continuation：

- direct：Pick 强升，但 OCR/GE/CLIP/HPS 明显下降；
- hinge：OCR/GE 可保留，但 HPS/CLIP 仍下降；
- MOPD continue：基本不动；
- naive KL：也没有提供安全方向。

这更像“Student 已处在某个局部 capability frontier”，而不只是 loss 没调好。在这种起点上加
Pareto hard gate 很可能得到大量 no-op：selector 很可靠，但没有可学的 branch。

### 2.3 continuation 让论文对比变得不自然

若我们从 DiffusionOPD Student 开始，而 DiffusionOPSD/Self-OPD 的 headline 表从 SD3.5 base 开始，
则无法直接公平比较。必须重跑两个 `@Student` 版本，最后回答的是一个更窄的问题：

> strong multi-teacher OPD student 能否被再次 post-train？

这个问题有价值，但成功概率和 audience 都小于：

> trajectory-level multi-reward OPSD 能否比 scalar reward fusion 更可靠地到达 joint Pareto region？

## 3. 修正后的主线

### Track A（主线，优先）：SD3.5-M base → Pareto Self-OPD/RPI

共同初始化：SD3.5-M base + fresh LoRA。

主表：

| 方法 | 起点 | 作用 |
|---|---|---|
| SD3.5-M base | base | 零训练对照 |
| DiffusionOPSD-Open3 | base | reward-gradient + scalar mixed reward baseline |
| Self-OPD-mixed | base | trajectory branch + scalar reward-level fusion baseline |
| Pareto Self-OPD / RPI | base | vector feasibility + reliable update gate（ours） |

目标不是保留一个旧 Student，而是让同一模型在 OCR、GenEval、Pick、HPS、CLIP 上形成更好的 joint
trade-off。这里 candidate space 较大，也最贴近 Self-OPD 对 reward-level fusion 的核心主张。

### Track B（第二阶段）：DiffusionOPD Student → continuation stress test

只有 Track A 先证明 selector/update gate 有用，才把完全相同的方法从 released DiffusionOPD Student
warm-start，比较：

```text
Frozen Student
DiffusionOPSD-cont@Student
Self-OPD-cont@Student
RPI-cont@Student
```

它在论文中可作为：

- strong-initialization robustness；
- continual capability acquisition；
- local Pareto frontier / no-op behavior 分析。

Track B 即使失败，也不会否定 Track A 的主方法；反而能形成“方法何时有可行 improvement”的分析。

## 4. from-base 下 RPI 应怎样改写

不再把 OCR/GE 叫作 frozen old capabilities，而是所有 reward 都对 branch 共同评分：

```text
A_k = [ΔOCR, ΔGE, ΔPick, ΔHPS, ΔCLIP]
```

候选 branch 必须：

1. 至少一个主目标有显著正 improvement；
2. 其他目标不能超过动态 tolerance/LCB 下界；
3. 换 suffix seed 后 improvement 大体可复现；
4. optimizer update 后 held-out mini-gate 不发生明显 Pareto regression。

Self-OPD scalar weighted sum 是正式 baseline；ours 的贡献是把 scalar ranking 改为 vector reliable
improvement。DiffusionOPSD gradient proposal 可后加，不应成为 V1 的必要部分。

## 5. 新的执行顺序

1. 保留并完成当前正在跑的 official DiffusionOPSD Open3/单 reward jobs；它们本来就是 Track A baseline。
2. pin Self-OPD 官方 commit，先复现其公开 OCR smoke。
3. 接入相同 reward/evaluation stack，跑零训练 K-branch feasibility probe，分别在 base 和 Student 上测：
   - feasible branch rate；
   - Pareto coverage；
   - suffix-seed stability。
4. 若 base feasibility 明显高于 Student，直接验证 Track A 20-update Pareto Self-OPD。
5. Track A 成功后才跑 continuation Track B。

## 6. 最终判断

**继续把 continuation 当主线，没有充分理由。** 它是由旧项目问题和已有基础设施造成的路径依赖，
而现有负结果已经说明 strong Student continuation 是更难、可能局部不可行的设定。

更合理的论文策略是：

> 先从 SD3.5-M base 与官方 Self-OPD、DiffusionOPSD 做同起点对比，证明 reliable Pareto branch
> selection 能 work；再把 DiffusionOPD Student continuation 作为高难度 stress test，而不是把整篇
> paper 的成败押在 continuation 上。

Primary sources：

- DiffusionOPSD：<https://arxiv.org/abs/2608.24646>，<https://github.com/worldbench/DiffusionOPSD>
- Self-OPD：<https://arxiv.org/abs/2608.26872>，<https://github.com/Shiy-Zhang/Self-OPD>


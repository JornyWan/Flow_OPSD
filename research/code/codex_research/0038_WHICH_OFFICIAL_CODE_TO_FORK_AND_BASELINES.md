# 0038 应基于哪套官方代码开发，以及如何定义 baseline

**日期：2026-08-30**  
**决定：主方法 fork Self-OPD 官方代码；DiffusionOPSD 官方仓库保持独立、只做 baseline 与模块复用；DiffusionOPD 不作为主开发底座。**  
**状态：代码架构与实验口径决定；本条未提交训练任务。**

> **后续更正（0039）：** Self-OPD 仍是主开发底座，但实验优先级改为先做 SD3.5-M from-base；
> continuation 降为第二阶段 stress test。0038 中以 continuation 为“主表”的安排已被 0039 修订。

## 1. 最短答案

RPI-OPSD 的核心创新发生在：

```text
K 个 local branches 已产生
        ↓
从 scalar reward ranking 改成 vector-valued Pareto feasibility
        ↓
只蒸馏可靠的 positive branch
        ↓
actual optimizer update 后再做 exact commit/reject
```

这与 Self-OPD 官方代码的原生抽象完全重合。因此主开发应基于：

```text
https://github.com/Shiy-Zhang/Self-OPD
```

而不是在 DiffusionOPSD 或 DiffusionOPD 训练循环上重新造 branch sampler。

## 2. 为什么不是直接改 DiffusionOPSD

DiffusionOPSD 官方代码已经成熟支持：

- SD3.5-M、LoRA、EMA、checkpoint/resume；
- PickScore、CLIPScore、HPSv2、Aesthetic 等 reward；
- single/mixed reward；
- official cross-evaluation；
- 我们已经在本地成功运行和核验。

但它的原生 improvement operator 是：

```text
image-space reward gradient → bounded positive/negative clean target
```

它没有 Self-OPD 所需的 `K` 个 SDE next-state branches、每个 branch 的 ODE suffix rollout、
per-branch score tensor 及 branch velocity target。把这些移入 DiffusionOPSD 会重写训练主循环，既容易
messy，也会使“相对哪个 baseline 改了什么”不清楚。

因此 DiffusionOPSD 应保持为独立 upstream baseline；只把它的 differentiable gradient proposal 和
reward/evaluation adapters 以窄接口接进 Self-OPD fork。

## 3. 为什么 Self-OPD 是最小、最清晰的改动面

官方 Self-OPD 当前已有：

- `SELF_OPD_WARM_LORA` / `config.train.lora_path`，可直接从 released DiffusionOPD Student LoRA
  warm-start；
- `sample_branches_and_score`：生成 K branches、ODE rollout、保留每个 scorer 的分数；
- `select_all_branches`：当前 scalar weighted-sum、advantage、direction-aware pull-push 集中在一个函数；
- 训练和评估均使用 SD3.5-M + PEFT LoRA，与我们当前 backbone 对齐。

所以 V1 的核心修改可以局限为：

```text
self_opd/pareto_selector.py        # 新增，不污染官方 selector
self_opd/reward_adapters/*.py      # 接入已验证的 Pick/HPS/CLIP
self_opd/update_gate.py            # post-fit evaluate/backtrack/identity
configs/rpi_opd.py                 # 新配置
scripts/probe_pareto_feasibility.py# Gate A，不训练
```

并保留原 `select_all_branches` 不改，从同一 fork 直接运行官方 Self-OPD baseline。

## 4. 一个必须明确的官方代码缺口

截至本次检查，Self-OPD 官方仓库虽然提供通用 reward registry 和 weighted multi-reward 接口，但实际
内置的训练 reward **只有 OCR**；README 也说明额外 reward configs/checkpoints 将逐步发布。

因此：

- 可以把其 branch generation 和 OCR run 称为官方代码路径；
- 不能把我们自行加入 Pick/HPS/CLIP 后的 mixed run 不加限定地称作“官方 Self-OPD 复现”；
- 应命名为 `Self-OPD-cont (official objective, our reward adapters)`；
- 等官方发布完整 mixed config 后，再补一个 commit-pinned exact reproduction。

另外，官方仓库当前只有一个公开 commit，因此必须记录 commit SHA，不依赖浮动的 `main`。

## 5. 公平 baseline 矩阵

### 5.1 主表：回答 continuation 是否有效

所有方法必须使用完全相同的：

```text
initialization = released DiffusionOPD Student LoRA
backbone       = SD3.5-M
train prompts  = 同一 hash-pinned manifests
rewards        = 同一 scorer checkpoints / preprocessing
sampling CFG   = 同一 protocol
update budget  = 同一 optimizer updates，同时报告 GPU-hours
evaluation     = 同一 seeds + own-run frozen Student remeasurement
```

主表至少包含：

| Run | 作用 | 实现来源 |
|---|---|---|
| Frozen DiffusionOPD Student | continuation 的真实起点/零训练对照 | released checkpoint |
| DiffusionOPSD-cont | reward-gradient OPSD baseline | untouched official DiffusionOPSD objective，warm-start Student |
| Self-OPD-cont | branch-search OPSD baseline | official Self-OPD objective，warm-start Student；新增 reward adapter 要注明 |
| RPI-OPSD | 我们的方法 | Self-OPD fork + Pareto selector + exact update gate |

已有 direct、hinge、naive-KL、MOPD-continue 作为内部 diagnostic/ablation，不代替上述两条正式 OPSD
baseline。

### 5.2 次表：回答与论文原始 setting 的关系

从 SD3.5-M base 开始，分别按照两篇论文的原始配置运行：

- official DiffusionOPSD from base；
- official Self-OPD from base；
- RPI-OPSD from base（只有 continuation 主表成功后才值得做）。

这张表不能与 continuation 主表混在一起。`from base` 和 `from DiffusionOPD Student` 是两个不同问题。

## 6. 最干净的 ablation 顺序

在同一个 Self-OPD fork 中逐项打开：

```text
A0  official all-branch scalar pull-push
A1  positive-only（检验负 repulsion 是否有害）
A2  A1 + per-metric Pareto veto
A3  A2 + suffix-seed lower-confidence check
A4  A3 + post-fit exact commit/backtrack
A5  A4 + DiffusionOPSD gradient proposal（最后加入）
```

如果直接比较 `Self-OPD` 与完整 `RPI-OPSD`，即使成功也不知道收益来自 selector、去掉 negative push，
还是 optimizer gate。以上顺序能形成清楚的 paper ablation。

## 7. 实施原则

1. 两个官方 upstream 分目录、commit-pinned、只读保存；我们的 fork 单独放置。
2. 不把 DiffusionOPSD 源文件复制进 Self-OPD 主循环；通过 adapter 接口复用 reward 和 gradient proposal。
3. baseline objective 不改；初始化/数据/评估适配单独列 patch，并在 run manifest 记录。
4. 先跑官方 Self-OPD OCR smoke，确认 branch/advantage/loss 与 upstream 一致。
5. 再做 Gate A feasibility probe；不过 gate 不提交长训练。

## 8. 最终判断

**代码底座选 Self-OPD，正式 baseline 同时保留 Self-OPD-cont 与 DiffusionOPSD-cont。**

这样方法故事最清晰：

> DiffusionOPSD 提供 gradient proposal；Self-OPD 提供 trajectory proposal；RPI-OPSD 的贡献是只接受
> 可重复、Pareto-feasible 且在实际 optimizer update 后仍成立的局部 improvement。

Primary sources：

- Self-OPD paper/code：<https://arxiv.org/abs/2608.26872>，<https://github.com/Shiy-Zhang/Self-OPD>
- DiffusionOPSD paper/code：<https://arxiv.org/abs/2608.24646>，<https://github.com/worldbench/DiffusionOPSD>

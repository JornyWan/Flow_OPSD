# 0028 方法谱系、原论文 baseline 与本项目共同起点

**日期：2026-08-30**  
**目的：消除 DiffusionOPD / DiffusionOPSD / Self-OPD 名称和初始化协议的混淆。**

## 0. 最重要的澄清

`DiffusionOPSD` **不是**在 `DiffusionOPD` checkpoint 上继续训练得到的方法。两者名字接近，
但原论文主实验是独立的 post-training 路线。

在 SD3.5-M 实验中：

```text
DiffusionOPSD paper: stable-diffusion-3.5-medium base → fresh LoRA → DiffusionOPSD
Self-OPD paper:      stable-diffusion-3.5-medium base → fresh LoRA → Self-OPD
```

而本项目研究的是 continual alignment，因此人为统一成：

```text
released DiffusionOPD multi-capability Student
    ├─ no training                    = Student-0 anchor
    ├─ continue with DiffusionOPSD    = primary direct baseline
    ├─ continue with Self-OPD         = latest nearest-neighbor baseline
    ├─ continue original MOPD         = teacher-based replay baseline
    ├─ fixed hinge / naive KL         = retention baselines
    └─ RG-OPSD                        = ours
```

这不是复刻两篇论文的原始初始化，而是为了让所有 continuation 方法回答同一个问题：
**从同一个已经具备 OCR/GenEval/preference 能力的 Student 出发，加入新目标时谁能最好地保持旧能力。**

---

## 1. DiffusionOPSD 原论文到底从哪里开始

论文 SD3.5-M protocol 明确写明：所有可训练 SD3.5-M rows 都从
`stabilityai/stable-diffusion-3.5-medium` 开始，训练 rank-32、alpha-64 LoRA。

```text
SD3.5-M base
   ├─ FlowGRPO
   ├─ ReFL
   ├─ DiffusionNFT
   └─ DiffusionOPSD
```

因此 DiffusionOPSD 最直接的 main baselines 是：

- unadapted SD3.5-M base；
- FlowGRPO；
- ReFL；
- DiffusionNFT。

它还在 Z-Image-Turbo 上重复相似比较：从 released `Tongyi-MAI/Z-Image-Turbo` 开始继续适配，
对比 FlowGRPO、ReFL、DiffusionNFT。Z-Image-Turbo 自身虽然是 distilled checkpoint，但
DiffusionOPSD 训练不访问其原始 teacher/distillation components。

### 为什么论文表里又出现 DiffusionOPD

那是单独的 **two-stage OPD baseline block**，不是 DiffusionOPSD 的初始化：

1. 从 SD3.5-M base 分别训练 Pick/CLIP/HPS 三个 DiffusionOPSD specialists；
2. 冻结三个 specialists 作为 teachers；
3. 再从 SD3.5-M base 初始化一个 shared student；
4. 用 DanceOPD / DiffusionOPD / FlowOPD 分别做 300-step teacher distillation。

所以正确关系是：

```text
base → DiffusionOPSD specialists → frozen teachers
base → shared student + frozen teachers → DiffusionOPD/FlowOPD/DanceOPD baseline
```

不是：

```text
DiffusionOPD → DiffusionOPSD
```

Primary sources：<https://arxiv.org/abs/2608.24646>，
<https://github.com/worldbench/DiffusionOPSD>。

---

## 2. Self-OPD 原论文到底从哪里开始

论文 implementation details 与公开 config 均给出：

```text
base model = stabilityai/stable-diffusion-3.5-medium
resolution = 512×512
trainable  = transformer LoRA
K branches = 8
optimizer  = AdamW, lr=3e-4
```

### single-reward

GenEval、OCR、PickScore、HPSv2 各自的 Self-OPD specialist 都从 SD3.5-M base/fresh LoRA 开始。

### mixed-reward

论文展示两种训练路径：

```text
from scratch:
    SD3.5-M base → one tri-reward Self-OPD run

warm-start:
    SD3.5-M base → single-reward specialist LoRAs
                  → use a specialist LoRA as warm start for mixed Self-OPD
```

warm-start 仍然叫 teacher-free，是因为 mixed stage 不冻结/查询 specialist teacher；它只把已训练
LoRA 当初始化。公开 config 也通过 `SELF_OPD_WARM_LORA` 支持这种初始化。

但是当前 v1 论文的 Table 2 只写 `Self-OPD (Ours)`，没有在表格行上明确标注最终点来自
from-scratch 还是 warm-start curve；公开 mixed config/checkpoint 又尚不完整。因此复现主表时不能
擅自假设，必须等待作者 release 或分别报告两个版本。

Self-OPD 的原论文 baselines 分三组：

- base：unadapted SD3.5-Medium；
- teacher-free RL：Flow-GRPO、GRPO-Guard、DiffusionNFT；
- teacher-based OPD：Flow-OPD、DiffusionOPD。

Primary sources：<https://arxiv.org/abs/2608.26872>，
<https://github.com/Shiy-Zhang/Self-OPD>。

---

## 3. DiffusionOPD 本身是什么角色

DiffusionOPD 是 teacher-based two-stage OPD：

```text
base → train task specialists/teachers
base → initialize shared student
student follows its on-policy trajectory and matches frozen teacher transitions
    → multi-capability DiffusionOPD Student
```

我们现在使用的 released DiffusionOPD Student 已经不是原始 SD3.5-M base，而是完成过多 teacher
融合、已经具备多能力的模型。

因此：

- 和 `Student-0` 比较，回答是否总体超过已有模型；
- 从 Student 跑 `DiffusionOPSD-direct`，回答我们的 guard/selector 是否优于同 target constructor；
- 从 Student 跑 `Self-OPD`，回答 branch-based self-distillation 在同一 continual 起点是否更强；
- 从 Student 继续 `MOPD`，回答保留旧 teachers/replay 是否已经足够。

---

## 4. 我们最终应如何命名，避免审稿时混淆

主表每一行都把 **initialization → method** 写全：

| 建议名称 | 初始化 | continuation |
|---|---|---|
| `Student-0` | released DiffusionOPD Student | none |
| `Student→MOPD-continue` | same Student | original multi-teacher OPD |
| `Student→DiffusionOPSD-direct` | same Student | canonical DiffusionOPSD objective |
| `Student→Self-OPD-Pick` | same Student | Self-OPD single reward |
| `Student→Self-OPD-mixed` | same Student | Self-OPD reward-level fusion |
| `Student→KL` | same Student | naive KL retention |
| `Student→fixed-w10` | same Student | OPSD + fixed hinge |
| `Student→RG-OPSD` | same Student | post-fit candidate selection |

另外单列 `Reported original-paper results`，只用于背景：

```text
base→DiffusionOPSD (reported)
base→Self-OPD (reported)
```

这些官方点估计不能与我们从 DiffusionOPD Student 开始的结果放在同一公平 ranking 中，因为
初始化、已有能力、训练目标和 protocol 都不同。

## 5. 一句话版本

> **原论文里，DiffusionOPSD 和 Self-OPD 都主要从 SD3.5-M base 开始训练；DiffusionOPD 是它们
> 拿来比较的 teacher-based OPD baseline，而不是 DiffusionOPSD/Self-OPD 的必然起点。我们为了
> continual alignment，才把 released DiffusionOPD Student 设为所有方法的共同新起点。**

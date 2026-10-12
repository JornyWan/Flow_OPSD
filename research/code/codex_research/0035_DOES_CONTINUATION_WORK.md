# 0035 continuation 训练到底有没有效果？

**日期：2026-08-30**  
**结论：MOPD-continue 基本无效果；reward-directed continuation 效果很强但高度偏科；截至目前没有一条 continuation 全轴优于 Student。**

## 1. 必须区分三种 continuation

### A. `Student→MOPD-continue`

六点相对 Student、单位为各轴重测 `2σ`：

```text
Pick −0.03, OCR +0.17, GenEval −0.15,
CLIP +0.24, AES −0.25, HPS −0.32
```

全部在噪声内，所以它确实几乎没有学习出任何新东西。这并不奇怪：released Student 本来就是
这组 multi-teacher distillation 的产物，继续用相同 teachers/objective 蒸馏，Student 已接近固定点。

这条臂的价值不是提高指标，而是证明：**同样再训练 200 epoch 本身不会自动损坏模型。**

### B. `Student→OPSD-style direct reward continuation`

```text
Pick +7.39, OCR −3.39, GenEval −4.07,
CLIP −5.20, AES +3.01, HPS −11.41
```

所以它不是没效果，而是效果非常强、非常偏向被优化的 PickScore。它显著推动新 reward，同时
破坏 OCR、GenEval、CLIP 和 HPS。用“没效果”描述会掩盖真正问题：**reward-only update 发生了
严重 capability trade-off / reward hacking。**

### C. `Student→one-sided hinge w=10`

```text
Pick +2.55, OCR +0.39, GenEval −0.23,
CLIP −2.80, AES −1.77, HPS −11.20
```

它得到一个窄而真实的正结果：OCR/GenEval 与 Student 基本持平，同时 Pick 明显上升；而且六轴
严格支配 naive KL。但是 held-out HPS/CLIP 仍显著下降，所以不是全面胜出 Student。

## 2. 最准确的一句话

> **目前不是 continuation 没有效果，而是现有 objective 只能在“几乎不动”（MOPD）和“沿新
> reward 强烈移动但忘掉其他能力”（direct）之间选择；hinge 找到了中间的窄改进，却仍未解决
> broad general quality retention。**

## 3. 对项目方向的含义

- 如果论文目标是“所有指标全面超过 Student”，目前答案是 **没有做到**；
- 如果目标是“在保住 OCR/GenEval 的条件下继续提高 Pick”，hinge 已经给出 **初步正结果**；
- 真正剩余的 blocker 是 HPS/CLIP 分布外退化，不是 OCR/GenEval constraint；
- from-base 的官方 OPSD Pick specialist 也出现 OCR/strict GenEval 代价，说明这不是单纯因为我们
  从 Student continuation，而是 reward-directed post-training 的共同 failure mode；
- 下一步 RG-OPSD 的价值在于逐步选择 direct/guarded/no-op，并加入独立 general-prompt HPS gate，
  目标是把“窄正结果”变成真正的 Pareto 改善。

## 4. 当前结论边界

- 六个 checkpoint 属于各自的一条训练轨迹，不是六个独立 seeds；
- 数值足以判断机制、筛候选和画 trajectory Pareto，不足以给 paper-level confidence interval；
- Self-OPD continuation 尚未 own-run；官方 mixed release 不完整；
- 因此当前能说“没有全轴赢家”，不能说“所有可能的 continuation 都无效”。

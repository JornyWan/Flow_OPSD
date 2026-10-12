---
name: opsd-open3-baseline-structure
description: DiffusionOPSD 论文有两套互不相同的结果(reward-specific 主表 vs Open3 对比);DiffusionOPD 是 Open3 对比里的一条 baseline;Open3 无 published 数值
metadata: 
  node_type: memory
  type: project
  originSessionId: 5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe
  modified: 2026-08-30T16:33:10.149Z
---

**DiffusionOPSD 论文的结果分两套,口径完全不同 —— 混用会造出假的"复现"结论。**

1. **Main Results 表**(README L133-146):原文写明
   "Reward-specific checkpoints are evaluated on their **matched** held-out objective"
   ⇒ **每一列都来自一个单独 reward 专训的 checkpoint**(Pick 24.94 / CLIP 0.340 /
   HPS 0.390 / Aes 12.08 / ImgR 1.76)。拿一个臂去比五列是错的。

2. **Open3 comparison**(见仓库 `opd/README.md`):**一个 checkpoint 打全部三列**,
   joint 目标 `config/public.py:sd35_open3`(pickscore+clipscore+hpsv2 各 1.0,
   β=0.1,300 updates;其余 reward-specific 臂 β=1.0、100 updates)。
   **⚠️ 公开版只有图 `assets/robustness.png`,没有数值表** ⇒
   **跑完 Open3 不能说"复现了论文",只能说"按论文给定的 joint 目标训了这条臂"。**

**关键结构认识:DiffusionOPD 不是"另一篇论文",它是 OPSD 的 Open3 对比里的一条 baseline。**
该对比放三条 teacher-distillation baseline(DanceOPD / **DiffusionOPD** / FlowOPD),
三条都把**三个冻结的 OPSD 专家**(pickscore/clipscore/hpsv2 的 **checkpoint-100** LoRA)
按 1:1:1 蒸馏进同一个 student。我们一直当起点用的发布版 Student 就是这条臂的产物。

**因此缺的东西是同一件事的两面**:我们只训了 `pickscore` 专家。
补 `clipscore` + `hpsv2` 两个专家可同时 (a) 让 OPSD 主表复现从 1/5 列 → 3/5 列,
(b) 凑齐 Open3 对比所需的三个教师,使 DanceOPD/FlowOPD 变得可训。
(Aes/ImgR 两列只服务主表,不进 Open3 对比,优先级低。)

**踩过的坑:** OPSD 仓库**根本没有 `reward_ckpts/` 目录** —— 之前只跑 pickscore(走 HF hub)
所以从未触发;open3/hpsv2 需要本地 `open_clip_pytorch_model.bin`(ViT-H-14, 3.94G)与
`HPS_v2.1_compressed.pt`。已软链到 DiffusionOPD 侧同名目录。
另:`src/reward_ckpt_path.py` 的回退路径经 shim 后**仍落在仓库内**(OS 先跟随符号链接再解析 `..`),
旧注释说"解析到垫片上级"是误诊,真因只是目录不存在。

相关: [[opd-caopd-project]] [[opd-need-positive-results]] [[cluster-usage-conventions]]

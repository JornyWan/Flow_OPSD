# 迁移复现验证结论 (2026-08-23)

在本集群(minilab SLURM)从零重建的 `diffusionopd` 环境上,重跑原服务器(colligo/Pluto)已有的
全部评测,逐项对比。**结论:管线数值可信,不存在系统性偏差。**

## 方法

- 16 个目标 × 3 数据集 × 6 指标 = **96 项对比**
- 协议与原记录对齐:n 各自匹配(base/teacher/Student=512,各 idea arm=256——256 是 512 的前缀子集,
  混用会比错);student gs=4.5,GenEval-teacher gs=1.0;40 步 ODE;`EVAL_REWARDS` 四项
  (去掉需要 sglang 服务的 unifiedreward)
- **判据是"落在采样噪声内",不是数字相同**。两次评测是同分布的独立采样:不同 GPU 的 fp16 kernel
  差异在 40 步采样中会放大成完全不同的图像,不存在逐位复现
  - `geneval`/`ocr` 是逐 prompt 的规则判定 → 二项标准误 `sqrt(p(1-p)/n)`
  - 模型打分类(pickscore/clipscore/aesthetic/hpsv2)是连续量,方差小得多

## 结果

**92 / 96 项在噪声内 (96%)**

z 分数分布(`z = (新-原)/1σ`,模型打分类以 0.5% 为等效 1σ):

| 指标 | n | z 均值 | z 标准差 | \|z\|max |
|---|---|---|---|---|
| geneval | 16 | −0.33 | 0.97 | 2.53 |
| ocr | 16 | +0.06 | 0.83 | 1.77 |
| pickscore | 16 | −0.04 | 0.20 | 0.50 |
| clipscore | 16 | −0.56 | 0.99 | 2.89 |
| aesthetic | 16 | +0.01 | 0.87 | 2.28 |
| hpsv2 | 16 | −0.04 | 1.10 | 3.71 |
| **全部** | **96** | **−0.15** | **0.90** | 3.71 |

整体 z 均值 −0.15 ≈ 0、标准差 0.90 ≈ 1 → **无系统性偏差,且噪声模型合适**。

### 最硬的单点证据:官方 Student LoRA

外部固定产物(非我们训练的),6 项最大偏差 **0.22%**:

| 指标 | 原 | 本集群 | 相对差 |
|---|---|---|---|
| GenEval | 0.9604 | 0.9615 | 0.11% |
| OCR | 0.9252 | 0.9273 | 0.22% |
| PickScore | 0.9219 | 0.9222 | 0.03% |
| ClipScore | 0.2957 | 0.2958 | 0.04% |
| Aesthetic | 6.1657 | 6.1594 | 0.10% |
| HPSv2.1 | 0.3426 | 0.3427 | 0.02% |

这一行同时验证了最难重建的两条链路:**GenEval**(自编 CUDA 算子的 mmcv 1.7.2 + mmdet 2.28.2
Mask2Former + open_clip ViT-L-14)和 **OCR**(paddlepaddle-gpu 2.6.2 + PaddleOCR)。

## 4 项超出阈值的说明

| 项 | 差 | 说明 |
|---|---|---|
| AesTeacher / geneval | −0.056 (−2.5σ) | GenEval 在 p≈0.5 处二项方差最大;AesTeacher 不为 GenEval 优化,输出大量落在判定边界。96 次比较里出现一个 2.5σ 属预期内 |
| mtnft690 / clipscore, aesthetic, hpsv2 | ≤1.9% 相对 | 该 run 本身是**训练崩溃**(geneval 0.054 / ocr 0.010),输出退化、方差远大于正常模型,1% 阈值对它过紧。绝对差 ≤0.048 |

clipscore 整体 z 均值 −0.56(约 2.2 个均值标准误)提示可能有极小系统性负偏,绝对量级
约 −0.0008 / 0.29 ≈ **0.3%**,低于任何实用显著性。

## 过程中修掉的 3 个自身问题

1. **评测脚本缺 GPU 数守卫** → SLURM 把 4 卡降级成 3 卡 → 512 样本被分布式采样器补齐成 513
   (1 个重复 prompt,影响 ~0.2%)。已加守卫;已完成结果影响小于噪声带,未重跑
2. **`--checkpoint_path $CKPT` 未加引号** → 路径中的空格被 word-split → 8 个目标全部退出码 2。
   改用数组传参
3. **`glob.glob` 未转义** → `run_name` 里的 `[...]` 被当成字符类 → auto-resume 扫不到 checkpoint、
   旧 train_state 不被清理。改用 `glob.escape`

第 2、3 条同源:`config.save_dir` 形如 `logs/[MOPD-CAOPD + SD3.5-M + PickScore+OCR+GenEval]`,
**方括号在 glob 里是字符类、空格在 shell 里是分隔符**。

## 复现方式

```bash
sbatch opd_related/slurm/eval_all.slurm      # 16 目标 × 3 数据集,A40 四路并行
python3 opd_related/compare_eval.py          # 带噪声带的逐项对比
```

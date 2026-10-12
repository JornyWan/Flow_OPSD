#!/usr/bin/env python
"""DiffusionOPD 论文 8 列复现表(本集群),每列带重测误差棒。

前 6 列的 1σ 来自"原服务器 vs 本集群"的跨机配对重测;
第 7/8 列(ImageReward/UnifiedReward)原服务器从未测过,1σ 来自同一 checkpoint 的重复评测
(nbIR/nbUNI 各 4 次)以及 IR/UNI 两个评测目录对同一模型的重复测量。
PickScore 与 UnifiedReward 代码里存的是归一化值,论文报原始值(分别 x26 和 x5)。
"""
import importlib, math, sys

O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
src = open(f"{O}/overall_score.py").read().split('agg(TRAINED,')[0]
M = type(importlib)("M"); exec(compile(src, "M", "exec"), M.__dict__)
ours, sigma = M.ours, M.sigma

COLS = [("geneval", "GenEval", 1.0), ("ocr", "OCR", 1.0),
        ("pickscore", "PickSc", 26.0), ("clipscore", "ClipSc", 1.0),
        ("aesthetic", "Aes", 1.0), ("hpsv2", "HPS", 1.0),
        ("imagereward", "ImgRwd", 1.0), ("unifiedreward", "UniRwd", 5.0)]
ROWS = ["Student", "mopd931", "caopd781", "regret721", "robust721",
        "poe15_331", "AesTeacher", "OCRTeacher", "GenEvalTeacher", "base_gs4.5"]

print("行 = 模型, 列 = 论文 8 项指标(已换算成论文口径)\n")
print(f"{'模型':<16}" + "".join(f"{n:>10s}" for _, n, _ in COLS))
print(f"{'重测 1σ':<16}" + "".join(
    f"{sigma[k]*s:>10.4f}" if k in sigma else f"{'—':>10s}" for k, _, s in COLS))
print("-" * (16 + 10 * len(COLS)))
for r in ROWS:
    if r not in ours:
        continue
    print(f"{r:<16}" + "".join(
        f"{ours[r][k]*s:>10.4f}" if k in ours[r] else f"{'—':>10s}" for k, _, s in COLS))

print("\n注: 判定两个模型是否有差别,用 2σ(表中 1σ 的两倍)。")
print("    论文正文数字对应 Student 行 —— 该行是官方发布的 Student LoRA 在本集群的重评。")

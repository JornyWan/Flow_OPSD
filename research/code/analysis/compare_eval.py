#!/usr/bin/env python
"""对比"原服务器记录"与"本集群重跑"的评测结果。

判据是【是否落在采样噪声内】而不是数字相同:两次评测是同分布的独立采样
(不同 GPU 的 fp16 kernel 差异在 40 步采样中会放大成完全不同的图)。
geneval/ocr 是逐 prompt 的规则判定 -> 用二项标准误 sqrt(p(1-p)/n);
模型打分类(pickscore/clip/aesthetic/hps)是连续量、方差小得多,用 1% 相对阈值。
"""
import json, os, math, sys
O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
R, N = f"{O}/CA-OPD/results/eval", f"{O}/outputs/eval"
KEYS = ["geneval", "ocr", "pickscore", "clipscore", "aesthetic", "hpsv2"]
BINOM = {"geneval", "ocr"}

def load(base, name):
    out = {}
    for ds in ("geneval", "ocr", "drawbench"):
        f = os.path.join(base, f"{name}_{ds}", "average_scores.json")
        if os.path.exists(f):
            try: out.update({k: v for k, v in json.load(open(f)).items() if k in KEYS})
            except Exception: pass
    return out

def n_of(name):
    f = os.path.join(R, f"{name}_geneval", "evaluation_results.jsonl")
    return sum(1 for _ in open(f)) if os.path.exists(f) else 512

targets = [l.split("\t")[0] for l in open("/midtier/minilab/scratch/chw4026/tmp/eval_targets.tsv") if l.strip()]
if "Student" not in targets: targets.insert(0, "Student")

hdr = f"{'目标':<15}{'指标':<11}{'原':>9}{'本集群':>9}{'差':>9}{'噪声(1σ)':>10}  判定"
print(hdr); print("-" * len(hdr))
tot = ok = pend = 0
for t in targets:
    old, new = load(R, t), load(N, t)
    if not old: continue
    n = n_of(t)
    first = True
    for k in KEYS:
        o, v = old.get(k), new.get(k)
        if o is None: continue
        if v is None: pend += 1; continue
        tot += 1
        d = v - o
        if k in BINOM:
            sigma = math.sqrt(max(o * (1 - o), 1e-9) / n)
            good = abs(d) <= 2 * sigma
            sig = f"{sigma:.4f}"
        else:
            sigma = 0.01 * abs(o)
            good = abs(d) <= sigma
            sig = f"{sigma:.4f}"
        ok += good
        print(f"{(t if first else ''):<15}{k:<11}{o:>9.4f}{v:>9.4f}{d:>+9.4f}{sig:>10}  {'✅' if good else '⚠️'}")
        first = False
    print()
print("-" * len(hdr))
print(f"已完成 {tot} 项对比,{ok} 项在噪声内 ({ok/tot*100:.0f}%)" if tot else "尚无可对比结果")
if pend: print(f"另有 {pend} 项还在跑")
print("判据: geneval/ocr 用 2σ 二项噪声带; 模型打分类用 1% 相对阈值")

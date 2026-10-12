#!/usr/bin/env python
"""两个 arm 在标准 benchmark 上的同步号配对检验。

用法: compare_arms.py <tagA> <tagB> [step...]
配对而非独立检验: 同一 step 的两臂共享训练进度这个主要方差来源,配对能把它消掉。
噪声带用实测的评测重测 RMS(见 09:52 记录: 现用 1σ 低估 1.1~2.5 倍),故这里按 2x 保守判定。
"""
import json, os, sys, statistics as st
E = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related/outputs/eval"
# 口径必须与 overall_score.py 一致(codex 0023 §3.2):
# geneval 目录的 JSON 里**同时有** `geneval` 与 `accuracy` 两个字段,取值差别很大
# (Student: geneval=0.9615 / accuracy=0.9376)。论文主表口径是 `geneval`
# —— Student 的 0.9615 与原论文 ~0.96 吻合。此前本脚本读 `accuracy` 却把行名
# 写成 "geneval",而 overall_score.py 读的是真正的 `geneval`,两边表格不是同一个指标。
# 现在:主行用 `geneval`,`accuracy` 保留为**另行命名**的诊断行,不再混用。
AXES = [("geneval", "geneval"), ("ocr", "ocr"), ("pickscore", "pick"),
        ("clipscore", "clip"), ("aesthetic", "aes"), ("hpsv2", "hps"),
        ("accuracy", "gen_acc*")]
# 实测评测重测 1σ(09:52),已按 2x 保守放宽
SIG = {"accuracy": 0.0297, "ocr": 0.0200, "pickscore": 0.00148,
       "clipscore": 0.00143, "aesthetic": 0.0179, "hpsv2": 0.00110,
       # ⚠️ `geneval` 字段的重测 sigma **未独立测过**,暂借用 accuracy 的 0.0297。
       # 两者量纲相近但不是同一个量;涉及 geneval 的显著性判定要标注这一点。
       "geneval": 0.0297}

DSETS = ("geneval", "ocr", "drawbench")

def load(tag):
    """返回 (指标, 实际读到的 benchmark 集合)。
    覆盖度必须回传:评测还没跑完时部分目录缺失,旧版静默 o.update 会拿残缺数据算配对检验而不报警。
    (潜在缺陷,非实测故障;2026-08-27 排查 horizon2 时顺带发现并封堵。)"""
    o, got = {}, set()
    for ds in DSETS:
        f = os.path.join(E, f"{tag}_{ds}", "average_scores.json")
        if os.path.exists(f):
            try:
                o.update(json.load(open(f))); got.add(ds)
            except Exception: pass
    return o, got

A, B = sys.argv[1], sys.argv[2]

def avail(tag):
    """该 arm 实际有评测的 step。硬编码 step 列表会静默漏掉大半可用点
    (实测 horizon2/tau10 硬编码只命中 n=2,真实可用 n=8),而 n 变小本身就是选择效应。"""
    import re, glob
    out = set()
    for d in glob.glob(os.path.join(E, f"{tag}_*")):
        m = re.match(rf"{re.escape(tag)}_(\d+)_", os.path.basename(d))
        if m: out.add(int(m.group(1)))
    return out

steps = [int(x) for x in sys.argv[3:]] or sorted(avail(A) & avail(B))
rows, partial = [], []
for s in steps:
    (a, ga), (b, gb) = load(f"{A}_{s}"), load(f"{B}_{s}")
    if not (a and b): continue
    miss = (set(DSETS) - ga) | (set(DSETS) - gb)
    if miss: partial.append((s, sorted(miss))); continue   # 残缺 step 一律剔除,不参与检验
    rows.append((s, a, b))
if partial:
    print("[warn] 以下 step 的 benchmark 未跑全,已剔除(等评测完成后重跑):")
    for s, m in partial: print(f"        step {s}: 缺 {m}")
    print()
if not rows:
    print("没有共同 step 的完整评测数据"); sys.exit(1)
print(f"共同 step: {[r[0] for r in rows]}  (n={len(rows)})\n")
print(f"{'轴':>8s} {A[:10]:>11s} {B[:10]:>11s} {'差(A-B)':>10s} {'se':>9s} {'t':>7s}  判定")
for k, nm in AXES:
    d = [r[1][k] - r[2][k] for r in rows if k in r[1] and k in r[2]]
    if len(d) < 2: continue
    ma = st.mean([r[1][k] for r in rows if k in r[1]])
    mb = st.mean([r[2][k] for r in rows if k in r[2]])
    m = st.mean(d); se = st.stdev(d) / len(d) ** 0.5
    t = m / se if se > 0 else float("inf")
    # 同时按噪声带判: |差| 是否超过 2σ
    big = abs(m) > 2 * SIG.get(k, 0)
    v = "显著" if (abs(t) > 2.5 and big) else ("t 显著但幅度在噪声带内" if abs(t) > 2.5 else "打平")
    print(f"{nm:>8s} {ma:11.4f} {mb:11.4f} {m:+10.4f} {se:9.4f} {t:+7.1f}  {v}")
print("\n注: 判定需同时满足 |t|>2.5 与 |差|>2σ(σ 为实测评测重测噪声,已按 2x 保守取值)。")

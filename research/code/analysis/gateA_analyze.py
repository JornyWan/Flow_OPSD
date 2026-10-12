#!/usr/bin/env python
"""Gate A 分析:按 0040 预注册的判据计算向量可行率 VFR。

判据(先于数据写定,不得事后调整):
    VFR >= 0.90  -> A1 无立足点,杀掉 A1
    VFR <= 0.60  -> A1 成立
    0.60~0.90    -> 不明确,报告逐轴统计再决定(不默认放行)

用法: gateA_analyze.py <gateA_probe.jsonl> [--by-epoch]
"""
import json, sys, os, math
from collections import defaultdict

def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    by_epoch = "--by-epoch" in sys.argv
    if not args:
        print("用法: gateA_analyze.py <gateA_probe.jsonl>"); return 2
    path = args[0]
    os.listdir(os.path.dirname(path) or ".")           # 刷 NFS 属性缓存
    if not os.path.exists(path):
        print(f"❌ {path} 不存在"); return 2

    rows = []
    for line in open(path):
        line = line.strip()
        if line:
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
    if not rows:
        print("❌ 探针文件为空 —— 探针未真正产出(不要当成'没崩就是对了')"); return 2

    keys = sorted(rows[0]["r0"])
    def stats(sub):
        n = len(sub)
        d = {k: [r["rp"][k] - r["r0"][k] for r in sub] for k in keys}
        all_pos = sum(1 for i in range(n) if all(d[k][i] > 0 for k in keys))
        return n, d, all_pos / n

    n, d, vfr = stats(rows)
    print(f"探针记录 {n} 条 state,reward 轴: {', '.join(keys)}")
    print(f"\n主判据  向量可行率 VFR = {vfr:.4f}   ({int(vfr*n)}/{n} 个 state 三轴同时改善)")
    if   vfr >= 0.90: verdict = "≥0.90  ⇒ **A1 无立足点,按预注册杀掉 A1**(只保留 A2)"
    elif vfr <= 0.60: verdict = "≤0.60  ⇒ **A1 成立**,进入实现"
    else:             verdict = "0.60~0.90 ⇒ **不明确**,看辅助统计再决定,不默认放行"
    print(f"        判定: {verdict}")

    print(f"\n辅助统计(预注册,不作主判据)")
    print(f"  {'轴':<12s}{'mean Δ':>12s}{'P(Δ<0)':>10s}{'P(Δ>0)':>10s}")
    for k in keys:
        v = d[k]; mean = sum(v)/n
        neg = sum(1 for x in v if x < 0)/n
        print(f"  {k:<12s}{mean:>+12.5f}{neg:>10.3f}{1-neg:>10.3f}")

    print(f"\n  轴间相关(是否存在结构性对立):")
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            a, b = d[keys[i]], d[keys[j]]
            ma, mb = sum(a)/n, sum(b)/n
            va = math.sqrt(sum((x-ma)**2 for x in a)); vb = math.sqrt(sum((x-mb)**2 for x in b))
            r = sum((a[t]-ma)*(b[t]-mb) for t in range(n))/(va*vb) if va*vb > 0 else float('nan')
            print(f"    corr({keys[i]:<10s}, {keys[j]:<10s}) = {r:+.4f}")

    if by_epoch:
        print(f"\n  按 epoch 分组的 VFR(看是否随训练变化):")
        g = defaultdict(list)
        for r in rows: g[r.get("epoch", -1)].append(r)
        for e in sorted(g):
            _n, _d, _v = stats(g[e])
            print(f"    epoch {e:<3d} n={_n:<6d} VFR={_v:.4f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())

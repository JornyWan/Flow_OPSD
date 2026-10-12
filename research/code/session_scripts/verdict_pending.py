#!/usr/bin/env python
"""待判实验的判定器。用法: verdict_pending.py {lownfe|latewindow|qfamily|hwoffset}

设计要点(都是踩过的坑):
  - 每个输入文件的臂/ck 一律由 JSON 的 `ckpt` 字段**反查**,不从文件名推断
    (`inner2100` 是 u100@ck100 = 外层 ep50,不是 ep100;`_a100` 是**评测**硬件后缀)。
  - 训练硬件由 `arm_provenance.json` 的 host 决定,打印出来供核对。
  - 判据阈值硬编码自登记节,**不接受命令行覆盖** —— 防止执行时改判据。
"""
import json, math, os, re, sys
O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
GPU = {**{f"ai-gpu{n:02d}": "A40"  for n in (1,2,3,4,5)},
       **{f"ai-gpu{n:02d}": "A100" for n in (6,7,8,9,10,11)},
       **{f"ai-gpu{n:02d}": "H100" for n in (12,13,14,16)},
       **{f"ai-gpu{n:02d}": "H200" for n in (17,18)}}
MDE = 0.0033
O3 = lambda r: r["pickscore"]/26.0 + r["clipscore"] + r["hpsv2"]

def load(fname):
    """返回 (逐图 dict, 元信息 dict) 或 (None, 原因)。"""
    p = os.path.join(O, "outputs", fname + ".json")
    if not os.path.exists(p): return None, f"缺文件 {fname}"
    d = json.load(open(p)); pi = d.get("per_image")
    if not isinstance(pi, list) or not pi: return None, f"{fname} 无 per_image"
    bad = sum(1 for r in pi if any(k not in r for k in ("pickscore","clipscore","hpsv2")))
    if bad: return None, f"{fname} 有 {bad} 条缺字段"
    m = re.search(r"(ab_open3_[^/]+)/checkpoints/checkpoint-(\d+)", d.get("ckpt",""))
    arm, ck = (m.group(1), int(m.group(2))) if m else ("(未训练底模)", 0)
    host = "—"
    try: host = json.load(open(os.path.join(O,"outputs",arm,"arm_provenance.json"))).get("host","—")
    except Exception: pass
    # CK_MULT = 各拟合端旋钮的**乘积**(inner<N> × gacc<M>)。
    # 原来是写死的 if/elif 链,没有 inner8 这一格 ⇒ inner8 静默落到 1,
    # ck800 会被报成 outer=800(真值 100)。改成从臂名解析,新旋钮不会再静默漏。
    mult = 1
    for _k in re.findall(r"(?:inner|gacc)(\d+)", arm): mult *= int(_k)
    if mult == 1 and re.search(r"(?:inner|gacc)(?![0-9])", arm):
        print(f"  ⚠️ {arm}: 臂名含 inner/gacc 但解析不出倍数,CK_MULT 按 1 处理 —— 请人工核对")
    meta = dict(arm=arm, ck=ck, mult=mult, outer=ck//mult if mult else ck,
                host=host, hw=GPU.get(host,"?"), n=len(pi),
                sampler=f"{d.get('sampler')}-{d.get('num_steps')}", adapter=d.get("eval_protocol",""))
    return {r["index"]: O3(r) for r in pi}, meta

def paired(a, b):
    ks = sorted(set(a) & set(b)); n = len(ks); d = [a[k]-b[k] for k in ks]
    m = sum(d)/n; sd = math.sqrt(sum((x-m)**2 for x in d)/(n-1)); se = sd/math.sqrt(n)
    if se == 0:   # 两份评测逐位相同 ⇒ 同一组权重(多半是 resume),不是独立 run
        return n, m, 0.0, float("nan"), sum(1 for x in d if x>0)/n*100
    return n, m, se, m/se, sum(1 for x in d if x>0)/n*100

def show(tag, meta):
    print(f"    {tag:<26} {meta['arm']:<40} ck={meta['ck']:<4} ×{meta['mult']} ⇒ 外层ep{meta['outer']:<4}"
          f" {meta['hw']:<5} n={meta['n']} {meta['sampler']}")

def lownfe():
    print("§4.525 低 NFE 探针 —— 判据见 §4.525(阈值硬编码,不可覆盖)")
    cells = []
    files = {"base4": "matched_dflt_basenp3_default_drawbench_flow4",
             "base10": "matched_dflt_basenp3_default_drawbench_flow10",
             "q4":    "matched_dflt_qcore_default_drawbench_flow4",
             "bm4":   "matched_dflt_basemodel_none_drawbench_flow4",
             "bm10":  "matched_dflt_basemodel_none_drawbench_flow10"}
    P = {}; ok = True
    print("  载入:")
    for k, f in files.items():
        v, meta = load(f)
        if v is None: print(f"    {k:<8} ✗ {meta}"); ok = False
        else: P[k] = v; show(k, meta)
    # 逐轴独立判:某一轴的输入齐了就算,不因其它轴缺件而整体放弃。
    # ⚠️ 阈值全部来自 §4.525 登记节,下面一个字不改。
    if not ({"base4","base10"} <= set(P)):
        print("\n  轴 L  ✗ 缺 base4/base10,不可判")
    else:
        n,m,se,t,w = paired(P["base4"], P["base10"])
        print(f"\n  轴 L  D(4) = basenp3@flow-4 − @flow-10:  n={n} d={m:+.5f} ±{se:.5f} t={t:+.2f} 胜率={w:.1f}% = {m/MDE:+.2f}×MDE")
        L = "L1(≥5×MDE 缺陷,Q 式修复有市场)" if m <= -0.0165 else \
            "L2(1–5×MDE,市场窄,不派训练活)" if m <= -0.0033 else \
            "L3(无缺陷 ⇒ 修复无市场,Q 转机制观察)" if m < 0.0033 else \
            "L4(更少步反而更好 ⇒ 先查实现)"
        print(f"        ⇒ 格 {L}")
    if not ({"q4","base4"} <= set(P)):
        print("  轴 M  ✗ 缺 qcore@flow-4,不可判(lownfe_qcore 未完成)")
        M=None
    else:
     n,m,se,t,w = paired(P["q4"], P["base4"])
     print(f"  轴 M  qcore@flow-4 − basenp3@flow-4:      n={n} d={m:+.5f} ±{se:.5f} t={t:+.2f} 胜率={w:.1f}% = {m/MDE:+.2f}×MDE")
     M = "M1(Q 首次占优)" if m >= 0.0033 else "M2(打平)" if m > -0.0033 else "M3(Q 仍落后)"
     print(f"        ⇒ 格 {M}")
    if {"bm4","bm10"} <= set(P):
     n,m,se,t,w = paired(P["bm4"], P["bm10"])
     print(f"  副轴  basemodel@flow-4 − @flow-10:        d={m:+.5f} t={t:+.2f} = {m/MDE:+.2f}×MDE(缺陷是通用还是模型特异)")

def latewindow():
    print("§4.459 晚窗族二分重做 —— 门槛 −0.0079,四格 A/B/C/D(硬编码)")
    F = {"b1":"matched_pi_base200_100","b2":"matched_pi_base200_200",
         "l1":"matched_pi_lr2x200_100","l2":"matched_pi_lr2x200_200",
         "i1":"matched_pi_inner2H100_200","i2":"matched_pi_inner2H100_400"}
    P={}; meta={}; missing=[]
    print("  载入:")
    for k,f in F.items():
        v,mt = load(f+"_default_drawbench_flow40")
        if v is None: print(f"    {k:<8} ✗ {mt}"); missing.append(k)
        else: P[k]=v; meta[k]=mt; show(k, mt)
    def dd(a2,b2,a1,b1):
        ks=sorted(set(a2)&set(b2)&set(a1)&set(b1)); n=len(ks)
        d=[(a2[k]-b2[k])-(a1[k]-b1[k]) for k in ks]
        m=sum(d)/n; sd=math.sqrt(sum((x-m)**2 for x in d)/(n-1)); se=sd/math.sqrt(n)
        return n,m,se,m/se,sum(1 for x in d if x>0)/n*100
    if not {"l1","l2","b1","b2"} & set(missing):
        n,m,se,t,w = dd(P["l2"],P["b2"],P["l1"],P["b1"])
        print(f"\n  d_lr2x   = {m:+.5f} ±{se:.5f} t={t:+.2f} 胜率={w:.1f}%  {'≤' if m<=-0.0079 else '>'} −0.0079")
        dl = m
    else: dl = None; print("\n  d_lr2x   ✗ 输入不全")
    if not {"i1","i2","b1","b2"} & set(missing):
        n,m,se,t,w = dd(P["i2"],P["b2"],P["i1"],P["b1"])
        print(f"  d_inner2 = {m:+.5f} ±{se:.5f} t={t:+.2f} 胜率={w:.1f}%  {'≤' if m<=-0.0079 else '>'} −0.0079")
        di = m
    else: di = None; print(f"  d_inner2 ✗ 输入不全(缺 {[k for k in ('i1','i2') if k in missing]})")
    if dl is not None and di is not None:
        c = "A(二分持续)" if di<=-0.0079 and dl>-0.0079 else \
            "B(早期现象)" if di>-0.0079 and dl>-0.0079 else \
            "C(族不可分)" if di<=-0.0079 and dl<=-0.0079 else "D(反转)"
        print(f"\n  ⇒ 格 {c}")
        hw = {meta[k]['hw'] for k in ('i1','i2','l1','l2','b1','b2')}
        print(f"  训练硬件集合 = {hw}  {'✅ 一致' if len(hw)==1 else '⚠️ 混杂,判定前须核 §4.537 的量化'}")
    else:
        print("\n  ⇒ 尚不可判(等 inner2H100 的 ck200/ck400 评测)")

def qfamily():
    """§4.523 的 Q 家族表升级为逐图配对(等 mev_pi_{qopsd,qbt}_100 补测落地)。
    全部 H200 训练,对 H200 basenp3;qcore 只有 L40 评测,单独标注。"""
    print("§4.523 Q 家族 · 官方口径(flow-40, default/EMA, drawbench 1000)")
    B, mb = load("matched_pi_basenp3_default_drawbench_flow40")
    if B is None: print(f"  ✗ 基线缺失: {mb}"); return
    show("basenp3(对照)", mb)
    ROWS = [("qcorelr", "matched_pi_qcorelr_100_default_drawbench_flow40", ""),
            ("qcore",   "matched_pi_qcore_100_l40_default_drawbench_flow40", "⚠️L40 评测,≤0.7×MDE 漂移"),
            ("qopsd",   "matched_pi_qopsd_100_default_drawbench_flow40", ""),
            ("qbt",     "matched_pi_qbt_100_default_drawbench_flow40", "")]
    print(f"\n  {'臂':<12}{'Δ vs basenp3':>14}{'±SE':>10}{'t':>8}{'胜率':>8}{'×MDE':>9}  备注")
    print("  "+"-"*74)
    for lbl, f, note in ROWS:
        A, ma = load(f)
        if A is None:
            print(f"  {lbl:<12}{'✗ ' + str(ma):>14}   ← 补测未到")
            continue
        n, m, se, t, w = paired(A, B)
        print(f"  {lbl:<12}{m:>+14.5f}{se:>10.5f}{t:>8.2f}{w:>7.1f}%{m/MDE:>9.2f}  {note}")
        if ma["hw"] != mb["hw"]:
            print(f"               ⚠️ 训练硬件 {ma['hw']} ≠ 基线 {mb['hw']}")
    print("\n  说明:§4.523 原表里 qopsd/qbt 是**均值差**(raw JSON 无 per_image)。")
    print("       补测落地后本命令给出配对 t 与胜率;配对只会缩小 SE ⇒ 符号不会翻。")


def hwoffset():
    """§4.548:直接测 A100↔H100 训练硬件偏移(等 inner2H100@ck200 评测落地)。"""
    print("§4.548 A100↔H100 硬件偏移 —— 判据 H1/H2/H3(阈值 ±0.0033,硬编码)")
    P={}
    for k,f in [("h100","matched_pi_inner2H100_200_default_drawbench_flow40"),
                ("a100_u100","matched_pi_inner2_default_drawbench_flow40"),
                ("a100_u200","matched_pi_inner2b_200_default_drawbench_flow40")]:
        v,m = load(f)
        if v is None: print(f"  {k:<10} ✗ {m}")
        else: P[k]=v; show(k,m)
    if {"a100_u200","a100_u100"} <= set(P):
        n,m,se,t,w = paired(P["a100_u200"],P["a100_u100"])
        print(f"\n  噪声地板 Δ_R(同 A100 两次 run) = {m:+.5f} ±{se:.5f} t={t:+.2f} = {m/MDE:+.2f}×MDE")
    if {"h100","a100_u100"} <= set(P):
        n,m,se,t,w = paired(P["h100"],P["a100_u100"])
        print(f"  主轴   Δ_H(H100 − A100)      = {m:+.5f} ±{se:.5f} t={t:+.2f} 胜率={w:.1f}% = {m/MDE:+.2f}×MDE")
        c = "H1(A100 显著高,§4.462 前提成立)" if m<=-0.0033 else \
            "H2(无显著硬件差 ⇒ §4.462 的作废理由须换)" if m<0.0033 else \
            "H3(反号 ⇒ 偏移方向在 inner 臂上不成立)"
        print(f"  ⇒ 格 {c}")
    else:
        print("\n  ⇒ 尚不可判(等 inner2H100 的 ck200 评测)")


if __name__ == "__main__":
    {"lownfe": lownfe, "latewindow": latewindow, "qfamily": qfamily, "hwoffset": hwoffset}.get(
        sys.argv[1] if len(sys.argv)>1 else "", lambda: print(__doc__))()

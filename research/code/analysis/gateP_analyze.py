#!/usr/bin/env python
"""Gate P(LSR-OPSD 前置闸门)分析:判据 P1-P4 见 0051 §③,阈值 launch 前已写定。

用法:
  gateP_analyze.py [--probe PATH]      正常分析
  gateP_analyze.py --selftest          用已知答案的合成数据自检(不读真实数据)
"""
import argparse, json, os, statistics as st, sys

O = "/midtier/minilab/scratch/chw4026/projects/side_project/generative/opd_related"
K3 = ("pickscore", "clipscore", "hpsv2")
U0 = "u0_official"
# 0051 §③ 预注册阈值 —— 不得在看到数据后修改
TH = {"P1_min_nonU0": 0.20, "P2_min_gain": 0.15, "P3_max_drop": 0.05}
EPS = 1e-12


def select(row, layer="roll"):
    """返回 (j*, retention_by_cand, worst_by_cand, scalar_by_cand, delta)。

    retention_{j,k} = ΔR_{j,k} / (H_k + eps),H_k = max_j ΔR_{j,k}(该 state 的局部 headroom)。
    j* = argmax_j min_k retention_{j,k}(max-min,保住最弱轴)。

    layer='roll'  : ΔR 测在**rollout 终点**(v2,主判据)。0055 证实 clean 层与"最终图更好"
                    在训练后期严重脱钩(98.6% vs 52%),故 v2 改用终点量。
    layer='clean' : ΔR 测在构造目标 y_j 上(v1 口径),仅作对照。
    兼容 v1 旧格式(cand[j] 直接是 {轴:分数})。
    """
    cand = row["cand"]
    _first = next(iter(cand.values()))
    if isinstance(_first, dict) and "roll" in _first:          # v2 嵌套格式
        r0 = row["r0_roll"] if layer == "roll" else row["r0"]
        d = {j: {k: cand[j][layer][k] - r0[k] for k in K3} for j in cand}
    else:                                                       # v1 旧格式
        r0 = row["r0"]
        d = {j: {k: cand[j][k] - r0[k] for k in K3} for j in cand}
    H = {k: max(d[j][k] for j in d) for k in K3}
    ret = {j: {k: d[j][k] / (H[k] + EPS) for k in K3} for j in d}
    worst = {j: min(ret[j][k] for k in K3) for j in d}
    scalar = {j: sum(d[j][k] for k in K3) for j in d}
    # 平手时优先 u0(保守:不因浮点抖动把选择判成"改变了")
    best = max(worst.values())
    jstar = U0 if worst.get(U0, -9e9) >= best - 1e-12 else max(worst, key=lambda j: worst[j])
    return jstar, ret, worst, scalar, d


def analyze(rows, label="", layer="roll"):
    n = len(rows)
    non_u0 = 0
    w_lsr, w_u0, s_lsr, s_u0 = [], [], [], []
    rows_changed = []
    neg_lsr = neg_u0 = 0
    picks = {}
    for row in rows:
        j, ret, worst, scalar, d = select(row, layer)
        picks[j] = picks.get(j, 0) + 1
        rows_changed.append(j != U0)
        if j != U0:
            non_u0 += 1
        w_lsr.append(worst[j]); w_u0.append(worst[U0])
        s_lsr.append(scalar[j]); s_u0.append(scalar[U0])
        neg_lsr += any(d[j][k] < 0 for k in K3)
        neg_u0 += any(d[U0][k] < 0 for k in K3)

    p1 = non_u0 / n
    # ⚠️ P2 必须只在**选择真的改变了的 state** 上算。
    # max-min 规则下,凡 u0 已是 argmax 的 state,retention_LSR 与 retention_u0 **逐位相同**,
    # 于是这些 state 对差值贡献恒为 0。若对全体取中位数,只要多数 state 选中 u0,
    # 中位数就被钉死在 0 —— 合成自检里 LSR 明明救回了 40/100 条,P2 仍读作 +0.00%。
    # 这是判据本身的**假阴性缺陷**,已于见到真实数据前修正(见 0051 §③ 修订记录)。
    chg = [(a, b) for a, b, r in zip(w_lsr, w_u0, rows_changed) if r]
    mw_l = st.median([a for a, _ in chg]) if chg else float("nan")
    mw_0 = st.median([b for _, b in chg]) if chg else float("nan")
    p2 = (mw_l - mw_0) / (abs(mw_0) + EPS) if chg else 0.0
    # 全体上的平均提升(信息性,不作判据)
    p2_all = st.mean(a - b for a, b in zip(w_lsr, w_u0))
    ms_l, ms_0 = st.mean(s_lsr), st.mean(s_u0)
    p3 = (ms_0 - ms_l) / (abs(ms_0) + EPS)          # 正数 = 总量下降
    p4l, p4u = neg_lsr / n, neg_u0 / n

    ok1, ok2 = p1 >= TH["P1_min_nonU0"], p2 >= TH["P2_min_gain"]
    ok3, ok4 = p3 <= TH["P3_max_drop"], p4l <= p4u + 1e-12
    print(f"{label}  n={n} 条 state  [层面={layer}]")
    print(f"  P1 选中非 u0 占比        = {p1:7.4f}   (阈值 ≥ {TH['P1_min_nonU0']})  {'✅' if ok1 else '❌'}")
    _f = (lambda v: "  —   " if v != v else f"{v:.4f}")   # nan(无改选 state)显示为 —
    print(f"  P2 worst-axis retention  = {_f(mw_l)} vs u0 {_f(mw_0)} => 提升 {p2:+7.2%}  (≥ {TH['P2_min_gain']:.0%})  {'✅' if ok2 else '❌'}")
    print(f"     ↑ 仅在选择改变的 {sum(rows_changed)}/{n} 条上算(见代码注释);全体平均提升 {p2_all:+.4f}(信息性)")
    print(f"  P3 scalar target gain    = {ms_l:.5f} vs u0 {ms_0:.5f} => 下降 {p3:+7.2%}  (≤ {TH['P3_max_drop']:.0%})  {'✅' if ok3 else '❌'}")
    print(f"  P4 三轴出现负 gain 的比率 = {p4l:7.4f} vs u0 {p4u:7.4f}            {'✅' if ok4 else '❌'}")
    print(f"  候选被选中次数: " + ", ".join(f"{k}={v}" for k, v in sorted(picks.items(), key=lambda x: -x[1])))
    allok = ok1 and ok2 and ok3 and ok4
    print(f"\n  ⇒ P1–P4 {'全部通过 —— 可按 0051 §③ 跑 P5(换 seed 复跑)' if allok else '未全过 ⇒ 按 0051 §④ 终止 LSR 线,不调阈值重看'}")
    return allok, dict(P1=p1, P2=p2, P3=p3, P4l=p4l, P4u=p4u)


def selftest():
    """已知答案的合成数据:构造 100 条 state,其中 40 条 u0 明显不是 max-min 最优。"""
    rows = []
    for i in range(100):
        r0 = {k: 0.0 for k in K3}
        if i < 40:
            # u0 抬高 pick 但压低 hps;单轴 hps 候选能把最弱轴救回来
            cand = {U0: {"pickscore": 1.0, "clipscore": 0.5, "hpsv2": 0.05},
                    "u_hpsv2": {"pickscore": 0.30, "clipscore": 0.30, "hpsv2": 1.0}}
        else:
            cand = {U0: {"pickscore": 1.0, "clipscore": 1.0, "hpsv2": 1.0},
                    "u_hpsv2": {"pickscore": 0.1, "clipscore": 0.1, "hpsv2": 1.0}}
        rows.append({"r0": r0, "cand": cand})
    # 负例:u0 在每条 state 上都是三轴的 headroom 上界 => 应当一条都不改选,闸门必须判"不过"
    neg_rows = [{"r0": {k: 0.0 for k in K3},
                 "cand": {U0: {"pickscore": 1.0, "clipscore": 1.0, "hpsv2": 1.0},
                          "u_hpsv2": {"pickscore": 0.2, "clipscore": 0.2, "hpsv2": 0.9}}}
                for _ in range(100)]
    print("=== 自检(负例):u0 恒为最优,闸门必须判不过 ===")
    ok_neg, mneg = analyze(neg_rows, "  [合成-负]", "clean")
    print(f"  期望 P1=0 且整体不过;实测 P1={mneg['P1']:.4f} 整体{'不过 ✅' if not ok_neg else '通过 ❌'}\n")

    print("=== 自检(正例):构造时已知 40/100 条应选非 u0 ===")
    _, m = analyze(rows, "  [合成-正]", "clean")
    exp = 0.40
    print(f"\n  构造真值 P1 = {exp:.4f};脚本算得 {m['P1']:.4f}  "
          f"{'✅ 一致' if abs(m['P1']-exp) < 1e-9 else '❌ 不一致 —— 脚本有 bug'}")
    # 第 40..99 条 u0 三轴都是 headroom 上界 => worst_u0 = 1.0;前 40 条 u0 的 hps retention=0.05
    print(f"  另:u0 的 median worst-axis 应为 1.0(后 60 条占多数)  实测 {st.median([select(r, 'clean')[2][U0] for r in rows]):.4f}")
    return abs(m["P1"] - exp) < 1e-9 and (not ok_neg) and abs(mneg["P1"]) < 1e-9


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default=f"{O}/outputs/gateP_lsr_probe/gateP_probe.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    os.listdir(os.path.dirname(a.probe)) if os.path.isdir(os.path.dirname(a.probe)) else None  # 刷 NFS
    if not os.path.exists(a.probe):
        print(f"⏳ 探针文件尚未产出: {a.probe}"); sys.exit(2)
    rows = [json.loads(l) for l in open(a.probe) if l.strip()]
    if not rows:
        print("⏳ 探针文件为空"); sys.exit(2)
    print("=" * 74); print("Gate P v2 分析(阈值见 0051 §③,launch 前写定)"); print("=" * 74)
    _f = next(iter(rows[0]["cand"].values()))
    if isinstance(_f, dict) and "roll" in _f:
        print("\n### 主判据:rollout 终点层(v2)")
        analyze(rows, "", "roll")
        print("\n### 对照:clean 目标层(v1 口径,0055 已证实是弱代理)")
        analyze(rows, "", "clean")
    else:
        analyze(rows, "", "clean")

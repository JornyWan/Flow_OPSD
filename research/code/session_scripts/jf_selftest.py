#!/usr/bin/env python
"""§4.875 判定器(数据前写定)。MSP 后期不稳的修复臂 mkpos300(ADVPOS:额外态 r1 ≥ 0.5)。每个 seed 单独判。
显著 = ≥2.5×MDE 且 |t|>2(单 seed);无效应 = |Δ|<1.0×MDE。
(1) 对 MSP(同 seed,ck250+ck300 × 两集,flow-40):M+ ≥3/4 格显著正且无显著负 | M− 任一格显著负 | M0 四格都无效应 | M? 其余。
(2) 对 canonical(同 seed,ck100–300 × 两集 = 10 格;flow-40 主、flow-4 副,各自分格):
    W 10/10 显著正 | W− 无显著负且 ≥5 格显著正 | T 无显著负且 <5 格显著正 | L 任一格显著负。
(3) 稳定性(离线 adam_global.py,ck150–300 每 10 步的 rms(m̂/√v̂) 最大值):由 §4.875 文字给出,不在本脚本内。"""
import sys, os, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py"); J = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv = _argv
CKS = (100, 150, 200, 250, 300); SETS = ("drawbench", "unseen1k")
SEEDS = {"seed1": dict(fix="mkema300", canon="o3canon300", msp="o3msp300", ema="mkema300"),
         "seed2": dict(fix="mkpos300s2", canon="canon300s2", msp="msp300s2", ema="mkema300s2")}
def d(a, b, ck, p, n=40):
    fa = f"matched_pi_{a}_{ck}_default_{p}_flow{n}"; fb = f"matched_pi_{b}_{ck}_default_{p}_flow{n}"
    if not (os.path.exists(f"outputs/{fa}.json") and os.path.exists(f"outputs/{fb}.json")): return None
    r = J.paired(J.load(fa, "open3"), J.load(fb, "open3")); return r[1] / J.MDE, r[3]
f = lambda v: "%+6.2f(%+5.1f)" % v if v else "   ⏳      "
sp = lambda v: v[0] >= 2.5 and v[1] > 2
sn = lambda v: v[0] <= -2.5 and v[1] < -2
for s, T in SEEDS.items():
    print(f"\n===== {s}")
    for n in (40, 4):
        print(f" flow-{n}:  ck | fix−canon (dr / un) | fix−msp | fix−mkema(只报)")
        for ck in CKS:
            row = [" / ".join(f(d(T['fix'], T[k], ck, p, n)) for p in SETS) for k in ("canon", "msp", "ema")]
            print(f"         {ck} | {row[0]} | {row[1]} | {row[2]}")
    M = [d(T["fix"], T["msp"], ck, p) for ck in (250, 300) for p in SETS]
    if any(v is None for v in M): vm = "⏳ 缺格"
    elif any(sn(v) for v in M): vm = "M− 修复臂更差"
    elif sum(sp(v) for v in M) >= 3: vm = "M+ 修好了后期"
    elif all(abs(v[0]) < 1.0 for v in M): vm = "M0 与 MSP 无差"
    else: vm = "M? 未判"
    out = []
    for n in (40, 4):
        C = [d(T["fix"], T["canon"], ck, p, n) for ck in CKS for p in SETS]
        if any(v is None for v in C): out.append(f"flow-{n} ⏳ 缺格"); continue
        k = sum(sp(v) for v in C)
        if any(sn(v) for v in C): vc = "L 有输"
        elif k == 10: vc = "W 全胜"
        elif k >= 5: vc = "W− 不输且多胜"
        else: vc = "T 持平"
        out.append(f"flow-{n} {vc}({k}/10 显著正)")
    print(f" {s} 判定(§4.875):对 MSP **{vm}** | 对 canonical " + " ; ".join(f"**{o}**" for o in out))

#!/usr/bin/env python
"""§4.860 判定器(数据前写定)。每个 seed 单独判;flow-40 主轴、两集;≥+2.5 且 t>2 为正,|·|<2.5 为无效应。
D = mkema − canonical(连续)| E = mkema − canonema(只有 seed1)| M = mkema − MSP(连续)。
E1 五个 ck 上 D(及 seed1 的 E)都 ≥+2.5(两集)| E2 M(300) 或 M(250) ≥+2.5(两集)且某 ck 的 D < +2.5
E3 M(250)、M(300) 都 |·|<2.5(两集)| E4 M(250) 或 M(300) ≤ −2.5(两集)| E2 与 E4 同时成立或其余 ⇒ E5。"""
import sys, os, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py"); J = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv = _argv
CKS = (100, 150, 200, 250, 300); SETS = ("drawbench", "unseen1k")
SEEDS = {"seed1": dict(ema="mkema300", canon="o3canon300", msp="o3msp300", cema="canonema300"),
         "seed2": dict(ema="mkema300s2", canon="canon300s2", msp="msp300s2", cema=None)}
def d(a, b, ck, p, n=40):
    if b is None: return None
    fa = f"matched_pi_{a}_{ck}_default_{p}_flow{n}"; fb = f"matched_pi_{b}_{ck}_default_{p}_flow{n}"
    if not (os.path.exists(f"outputs/{fa}.json") and os.path.exists(f"outputs/{fb}.json")): return None
    r = J.paired(J.load(fa, "open3"), J.load(fb, "open3")); return r[1] / J.MDE, r[3]
f = lambda v: "%+6.2f(%+5.1f)" % v if v else "   ⏳      "
pos = lambda L: all(v and v[0] >= 2.5 and v[1] > 2 for v in L)
neg = lambda L: all(v and v[0] <= -2.5 and v[1] < -2 for v in L)
nul = lambda L: all(v and abs(v[0]) < 2.5 for v in L)
for s, T in SEEDS.items():
    print(f"\n===== {s}(flow-40;flow-4 只报)")
    for n in (40, 4):
        print(f" flow-{n}:  ck | D=ema−canon (dr / un) | E=ema−canonema | M=ema−msp | canonema−canon")
        for ck in CKS:
            row = [" / ".join(f(d(T['ema'], T[k], ck, p, n)) for p in SETS) for k in ("canon", "cema", "msp")]
            ce = " / ".join(f(d(T["cema"], T["canon"], ck, p, n)) for p in SETS) if T["cema"] else "—"
            print(f"         {ck} | {row[0]} | {row[1]} | {row[2]} | {ce}")
    Dall = [d(T["ema"], T["canon"], ck, p) for ck in CKS for p in SETS]
    Eall = [d(T["ema"], T["cema"], ck, p) for ck in CKS for p in SETS] if T["cema"] else []
    M250 = [d(T["ema"], T["msp"], 250, p) for p in SETS]; M300 = [d(T["ema"], T["msp"], 300, p) for p in SETS]
    if any(v is None for v in Dall + Eall + M250 + M300):
        print(f" {s} 主判定:⏳ 缺格"); continue
    e2 = (pos(M300) or pos(M250)) and not pos(Dall); e4 = neg(M250) or neg(M300)
    if pos(Dall) and (not Eall or pos(Eall)): v = "E1 稳健胜出" + ("" if Eall else "(seed2 无 canonema,只按 D)")
    elif e2 and e4: v = "E5 未判(E2 与 E4 同时成立)"
    elif e2: v = "E2 修好崩塌,增益仍小"
    elif e4: v = "E4 重启有害"
    elif nul(M250) and nul(M300): v = "E3 重启无用"
    else: v = "E5 未判"
    print(f" {s} 主判定(§4.860):**{v}**  —— 底模门另查")

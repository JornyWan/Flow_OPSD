#!/usr/bin/env python
"""§4.888 判定器(数据前写定):MSP + 额外态独立 Adam(multikF4sa,λ=0.5)的 ck50/100 筛查。首要目标 = 40 步每个 ckpt 都胜 canonical。
  s0830 H100 np3:sah3s30 vs canonical baseH100(主)/ baserepH100(副),vs MSP mkF4hr2
  s0905 A100 np3:saa3s05 vs canonical bases2,vs MSP mkF4seed2
每 seed、每 ck 的 flow-40 两集(drawbench / unseen1k)对 canonical:
  N  任一格显著负 | P 两格都显著正 | P− 一格显著正、无显著负 | 0 无显著。显著 = ≥2.5×MDE 且 |t|>2。
筛查结论(两 seed 的 ck100 flow-40 对 canonical):
  晋级   = 两 seed 都 P / P−,且至少一个 P,**且两 seed 的 ck100 flow-40 对 MSP 都无显著负**(MSP 自己在这两组对照上已是 P / P−,
           所以"胜 canonical"不够;split 的预期收益在后期,ck≤100 只能查"不付早期代价")⇒ 上 H200×4 跑 300 步,排在 mkema300s2 之前
  持平   = 无 N,但不满足晋级 ⇒ 让本 run 跑到 ck150/200 再看(H100 那条有 g3h_canon 对照)
  负     = 任一 seed 的 ck100 flow-40 为 N ⇒ 先归因(实现 / 口径 / 机制),不直接判负
副读数:对 MSP 的同样格(split 是否不比 MSP 差)、flow-4、IR(held-out)同号数;同协议噪声 baserepH100 − baseH100。"""
import sys, os, json, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py"); J = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv = _argv
SETS = ("drawbench", "unseen1k")
def d(a, b, ck, p, n):
    fa = f"matched_pi_{a}_{ck}_default_{p}_flow{n}"; fb = f"matched_pi_{b}_{ck}_default_{p}_flow{n}"
    if not (os.path.exists(f"outputs/{fa}.json") and os.path.exists(f"outputs/{fb}.json")): return None
    r = J.paired(J.load(fa, "open3"), J.load(fb, "open3"))
    A = json.load(open(f"outputs/{fa}.json"))["per_image"]; B = json.load(open(f"outputs/{fb}.json"))["per_image"]
    return r[1] / J.MDE, r[3], sum(x["imagereward"] - y["imagereward"] for x, y in zip(A, B)) / len(A)
f = lambda v: "%+6.2f(t%+5.1f)" % v[:2] if v else "   ⏳        "
sp = lambda v: v[0] >= 2.5 and v[1] > 2
sn = lambda v: v[0] <= -2.5 and v[1] < -2
def cell(L):
    if any(v is None for v in L): return "⏳"
    if any(sn(v) for v in L): return "N"
    if all(sp(v) for v in L): return "P"
    if any(sp(v) for v in L): return "P−"
    return "0"
SEEDS = {"s0830 H100 np3": ("sah3s30", "baseH100", "mkF4hr2"), "s0905 A100 np3": ("saa3s05", "bases2", "mkF4seed2")}
print("同协议噪声 baserepH100 − baseH100:", "  ".join(f"ck{ck} f{n} " + "/".join(f(d('baserepH100', 'baseH100', ck, p, n)) for p in SETS) for ck in (50, 100) for n in (40, 4)))
final, msp_neg = {}, {}
for s, (a, c, m) in SEEDS.items():
    print(f"\n===== {s}:{a}(canonical {c},MSP {m})")
    for ck in (50, 100):
        for n in (40, 4):
            Lc = [d(a, c, ck, p, n) for p in SETS]; Lm = [d(a, m, ck, p, n) for p in SETS]
            ir = [v[2] for v in Lc if v]
            print(f"  ck{ck} f{n:<2d} vs canon: {f(Lc[0])} / {f(Lc[1])} ⇒ {cell(Lc):2s} | vs MSP: {f(Lm[0])} / {f(Lm[1])} ⇒ {cell(Lm):2s} | ΔIR vs canon {' / '.join('%+.3f' % x for x in ir) or '⏳'}")
    final[s] = cell([d(a, c, 100, p, 40) for p in SETS])
    vm = [d(a, m, 100, p, 40) for p in SETS]
    msp_neg[s] = None if any(x is None for x in vm) else any(sn(x) for x in vm)
    print(f"  ck100 flow-40 对 canonical:**{final[s]}** | 对 MSP 显著负:{msp_neg[s]}")
# 副(不进晋级规则):λ=1.0 在 H200 np3 上对 canonical basenp3(该硬件无 MSP 对照;basenp3 的 ck50 评测已补投)
print("\n===== 副:λ=1.0,s0830 H200 np3:sa1h2n3s30 vs canonical basenp3")
for ck in (50, 100):
    for n in (40, 4):
        Lc = [d("sa1h2n3s30", "basenp3", ck, p, n) for p in SETS]
        print(f"  ck{ck} f{n:<2d} vs canon: {f(Lc[0])} / {f(Lc[1])} ⇒ {cell(Lc):2s}")
# 副 2(§4.888 补 10:20,数据前):λ=0.5 同 seed 同硬件(H200 np3)⇒ 第三个 λ=0.5 点 + λ 剂量的同硬件对比
#   剂量读法(ck100 flow-40,λ=1.0 − λ=0.5):P/P− ⇒ 晋级时改用 λ=1.0;N ⇒ 维持 λ=0.5;0 ⇒ 维持登记的 λ=0.5
print("\n===== 副 2:λ=0.5,s0830 H200 np3:sah2n3s30 vs canonical basenp3;剂量 sa1h2n3s30 − sah2n3s30")
for ck in (50, 100):
    for n in (40, 4):
        Lc = [d("sah2n3s30", "basenp3", ck, p, n) for p in SETS]; Ld = [d("sa1h2n3s30", "sah2n3s30", ck, p, n) for p in SETS]
        print(f"  ck{ck} f{n:<2d} λ0.5 vs canon: {f(Lc[0])} / {f(Lc[1])} ⇒ {cell(Lc):2s} | λ1.0−λ0.5: {f(Ld[0])} / {f(Ld[1])} ⇒ {cell(Ld):2s}")
dose = cell([d("sa1h2n3s30", "sah2n3s30", 100, p, 40) for p in SETS])
# §4.891 修订(14:10,λ=0.5 的 H200 数据与 λ=1.0 ck100 之前写定):λ=1.0 ck50 的领先(+9.3 dr f40)与 canonical lr×2 一样大(lr2x50 +8.76,差 +0.54 不显著)
#   ⇒ λ=1.0 的收益可能只是总步长变大。步长对照 = lr2xnp3(canonical lr×2,H200 np3,s0830,同期 basenp3 的姊妹臂)。
#   改为:只有剂量格 P/P− **且** λ=1.0 − lr2xnp3 在 ck100 flow-40 也是 P/P− 才改用 λ=1.0;否则维持 λ=0.5。
print("\n===== 步长对照(§4.891):split − canonical lr×2(lr2xnp3,H200 np3 s0830)")
for ck in (50, 100):
    for n in (40, 4):
        L1 = [d("sa1h2n3s30", "lr2xnp3", ck, p, n) for p in SETS]; L5 = [d("sah2n3s30", "lr2xnp3", ck, p, n) for p in SETS]
        print(f"  ck{ck} f{n:<2d} λ1.0−lr2x: {f(L1[0])} / {f(L1[1])} ⇒ {cell(L1):2s} | λ0.5−lr2x(只报): {f(L5[0])} / {f(L5[1])} ⇒ {cell(L5):2s}")
stepc = cell([d("sa1h2n3s30", "lr2xnp3", 100, p, 40) for p in SETS])
use1 = "⏳" if "⏳" in (dose, stepc) else ("晋级时用 λ=1.0" if dose in ("P", "P−") and stepc in ("P", "P−") else "维持 λ=0.5")
print(f"  剂量结论(ck100 flow-40):剂量格 {dose} / 步长对照 {stepc} ⇒ {use1}")
v = list(final.values())
if "⏳" in v or None in msp_neg.values(): concl = "⏳ 未齐"
elif "N" in v: concl = "负 ⇒ 先归因"
elif all(x in ("P", "P−") for x in v) and "P" in v and not any(msp_neg.values()): concl = "晋级 ⇒ H200×4 300 步"
else: concl = "持平 ⇒ 跑到 ck150/200 再看"
print(f"\n筛查结论(§4.888):**{concl}**  {final}")

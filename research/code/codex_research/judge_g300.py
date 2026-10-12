#!/usr/bin/env python
"""§4.886 判定器(数据前写定)。用户首要目标(2026-10-11):**40 步在每个 ckpt 都最好**。
每个 seed 单独、同硬件配对(s0830 = H100 np3,s0905 = A100×4 scu):gate = MSP+门控(τ=0.75),msp = MSP,canon = canonical。
主判(flow-40,ck100/150/200/250/300 × 两集 = 10 格,对 canon):
  W 10/10 显著正 | W− 无显著负且 ≥6 格显著正 | T 无显著负且 <6 | L 任一格显著负。显著 = ≥2.5×MDE 且 t>2。
副:同样的格对 msp(门控是否比不门控好);flow-4 只报;IR(held-out)逐格同号数。"""
import sys, os, json, importlib.util
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py"); J = importlib.util.module_from_spec(spec)
_argv = sys.argv; sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass
sys.argv = _argv
CKS = (100, 150, 200, 250, 300); SETS = ("drawbench", "unseen1k")
SEEDS = {"s0830(H200 np4)": ("h2gate30", "o3msp300", "o3canon300"), "s0905(H200 np4)": ("h2gate05", "msp300s2", "canon300s2"),  # §4.888:主配对,对照是现成的 H200×4 300 步
         "s0830(H100 np3)": ("g3hgate", "g3hmsp", "g3hcanon"), "s0905(A100)": ("g300gates05", "g300msps05", "g300canons05")}
def d(a, b, ck, p, n):
    fa = f"matched_pi_{a}_{ck}_default_{p}_flow{n}"; fb = f"matched_pi_{b}_{ck}_default_{p}_flow{n}"
    if not (os.path.exists(f"outputs/{fa}.json") and os.path.exists(f"outputs/{fb}.json")): return None
    r = J.paired(J.load(fa, "open3"), J.load(fb, "open3"))
    A = json.load(open(f"outputs/{fa}.json"))["per_image"]; B = json.load(open(f"outputs/{fb}.json"))["per_image"]
    ir = sum(x["imagereward"] - y["imagereward"] for x, y in zip(A, B)) / len(A)
    return r[1] / J.MDE, r[3], ir
f = lambda v: ("%+6.2f(t%+5.1f)%s" % (v[0], v[1], "*" if v[0] >= 2.5 and v[1] > 2 else ("-" if v[0] <= -2.5 and v[1] < -2 else " "))) if v else "   ⏳         "
sp = lambda v: v[0] >= 2.5 and v[1] > 2
sn = lambda v: v[0] <= -2.5 and v[1] < -2
def verdict(L):
    if any(v is None for v in L): return "⏳ 缺格"
    k = sum(sp(v) for v in L)
    if any(sn(v) for v in L): return f"L 有输({k}/10 显著正)"
    if k == 10: return "W 全胜"
    if k >= 6: return f"W− 不输且多胜({k}/10)"
    return f"T 持平({k}/10)"
for s, (g, m, c) in SEEDS.items():
    print(f"\n===== {s}")
    for n in (40, 4):
        print(f" flow-{n}: ck | gate−canon (dr / un) | gate−msp | msp−canon")
        for ck in CKS:
            print(f"   {ck} | " + " | ".join(" / ".join(f(d(a, b, ck, p, n)) for p in SETS) for a, b in ((g, c), (g, m), (m, c))))
    for a, b, nm in ((g, c, "gate vs canon"), (g, m, "gate vs msp"), (m, c, "msp vs canon")):
        L = [d(a, b, ck, p, 40) for ck in CKS for p in SETS]
        irs = [v[2] for v in L if v]
        print(f" 主判定 flow-40 {nm}:**{verdict(L)}** | IR 为正 {sum(x > 0 for x in irs)}/{len(irs)}")
# §4.892 补(2026-10-11 15:50,门控 300 步零数据时写定,只报不决策):等效提前步数 Δ(c) —— 在 canonical 轨迹上线性插值找 c' 使 canon(c') = arm(c),Δ = c' − c
#   (* = 超出 ck100–300 按端段斜率外推)。读法:Δ 随 c 增长 = 速率增益(40 步每 ck 都显著所需);Δ 恒定 = 一次性提前量(MSP 自身的 40 步形状)。
def shift(a, c, p, n):
    if any(d(a, c, k, p, n) is None for k in CKS) or any(d(c, c, k, p, n) is None for k in CKS): return "⏳"
    lc = [0.0] + [J.paired(J.load(f"matched_pi_{c}_{k}_default_{p}_flow{n}", "open3"), J.load(f"matched_pi_{c}_100_default_{p}_flow{n}", "open3"))[1] / J.MDE for k in CKS[1:]]
    out = []
    for k in CKS:
        y = d(a, c, k, p, n)[0] + (lc[CKS.index(k)]); s = None; t = ""
        for j in range(4):
            if lc[j] <= y <= lc[j + 1]: s = CKS[j] + 50 * (y - lc[j]) / (lc[j + 1] - lc[j]); break
        if s is None:
            t = "*"; s = (300 + 50 * (y - lc[-1]) / (lc[-1] - lc[-2])) if y > lc[-1] else (100 + 50 * (y - lc[0]) / (lc[1] - lc[0]))
        out.append("%+.0f%s" % (s - k, t))
    return "/".join(out)
print("\n===== §4.892 副读数:等效提前步数 Δ(ck100/150/200/250/300)")
for s, (g, m, c) in SEEDS.items():
    for n in (40, 4):
        print(f" {s} f{n}: gate " + " | ".join(f"{p[:2]} {shift(g, c, p, n)}" for p in SETS) + " ‖ msp " + " | ".join(f"{p[:2]} {shift(m, c, p, n)}" for p in SETS))

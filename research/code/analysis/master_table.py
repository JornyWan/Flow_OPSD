#!/usr/bin/env python3
"""全臂绝对值大表。

口径要点(踩过的坑,别再犯):
  - 分数必须读 d["scores"],不能递归查找:顶层 train_reward_fn 里也有
    clipscore/hpsv2/pickscore 三个键(是权重 1.0),会把真分数遮蔽掉。
  - 每个臂只能和**同 nproc** 的 baseline 比:a1/n1/n2/a3 → np4;qopsd/qbt/b4x → np3。
  - b4x 是算力配平臂(B_eff 4×),它的 ck25 才对应 baseline 的 ck100。
  - 目标合计 = pickscore/26 + clipscore + hpsv2(与 scorer 里 PickScore 的 /26 归一一致)。
"""
import json, os, math

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
AXES = [("pickscore","Pick"),("clipscore","CLIP"),("hpsv2","HPSv2.1"),
        ("aesthetic","Aes"),("imagereward","ImgR")]
OBJ  = [("pickscore",1/26.0),("clipscore",1.0),("hpsv2",1.0)]

# (显示名, 文件名, 匹配的 baseline key, 备注)
ROWS = [
    ("base SD3.5-M",   "opsd_eval_base_ckbase_raw.json", None,   "未训练起点"),
    ("baseline np4",   "opsd_eval_ab_base_ck100_raw.json",    None, "官方复现 ck100"),
    ("baseline np3",   "opsd_eval_ab_basenp3_ck100_raw.json", None, "nproc 对照 ck100"),
    ("a1",     "opsd_eval_ab_a1_ck100_raw.json",    "np4", "ck100"),
    ("n1",     "opsd_eval_ab_n1_ck100_raw.json",    "np4", "ck100"),
    ("n2",     "opsd_eval_ab_n2_ck100_raw.json",    "np4", "ck100"),
    ("a3",     "opsd_eval_ab_a3_ck100_raw.json",    "np4", "ck100"),
    ("qopsd",  "opsd_eval_ab_qopsd_ck100_raw.json", "np3", "ck100"),
    ("qbt",    "opsd_eval_ab_qbt_ck100_raw.json",   "np3", "ck100"),
    ("b4x",    "opsd_eval_ab_b4x_ck25_raw.json",    "np3", "ck25 = 算力配平于 ck100"),
]
BASE_FILE = {"np4":"opsd_eval_ab_base_ck100_raw.json",
             "np3":"opsd_eval_ab_basenp3_ck100_raw.json"}

def load(fn):
    p = os.path.join(OUT, fn)
    if not os.path.exists(p): return None
    d = json.load(open(p))
    return {"s": d["scores"], "st": d.get("score_stats", {})}

def se(st, k):
    """逐样本 SE。score_stats 里存的是 std 与 n。"""
    e = st.get(k) or {}
    n = e.get("n") or e.get("count")
    sd = e.get("std")
    if not n or sd is None: return None
    return sd / math.sqrt(n)

D = {name: load(fn) for name, fn, _, _ in ROWS}
B = {k: load(v) for k, v in BASE_FILE.items()}

W = 16
print("="*118)
print("① 绝对分数大表(全部 5 轴 + Open3 目标合计)   —— 每格是 held-out prompt 集上的均值")
print("="*118)
hdr = f"{'臂':<{W}}" + "".join(f"{lab:>10}" for _, lab in AXES) + f"{'目标合计':>12}   备注"
print(hdr); print("-"*118)
AGG = {}
for name, fn, bkey, note in ROWS:
    d = D[name]
    if d is None:
        print(f"{name:<{W}}  ⏳ 缺 {fn}"); continue
    cells = "".join(f"{d['s'].get(k, float('nan')):>10.4f}" for k, _ in AXES)
    agg = sum(w * d["s"][k] for k, w in OBJ); AGG[name] = agg
    print(f"{name:<{W}}{cells}{agg:>12.4f}   {note}")

print()
print("="*118)
print("② 相对各自匹配 baseline 的差(Δ),以及逐轴显著性  判据:|Δ|>2σ 且 |t|>2.5")
print("="*118)
print(f"{'臂':<{W}}" + "".join(f"{lab:>10}" for _, lab in AXES) + f"{'Δ目标合计':>12}   判定")
print("-"*118)
for name, fn, bkey, note in ROWS:
    if bkey is None or D[name] is None: continue
    a, b = D[name], B[bkey]
    cells, nbet, nwor = "", 0, 0
    for k, _ in AXES:
        dv = a["s"][k] - b["s"][k]
        sa, sb = se(a["st"], k), se(b["st"], k)
        mark = " "
        if sa and sb:
            sd = math.sqrt(sa*sa + sb*sb); t = dv/sd
            if abs(dv) > 2*max(sa, sb) and abs(t) > 2.5:
                mark = "+" if dv > 0 else "-"
                nbet += dv > 0; nwor += dv < 0
        cells += f"{dv:>+9.4f}{mark}"
    dagg = AGG[name] - sum(w*b["s"][k] for k, w in OBJ)
    verdict = f"好 {nbet}/5, 差 {nwor}/5"
    print(f"{name:<{W}}{cells}{dagg:>+12.4f}   vs {bkey}: {verdict}")
print()
print("  标记 +/- = 该轴显著;空白 = 打平。目标合计的 MDE(可检出下限)= 0.0101,")
print("  等于训练自身总增益 +0.39567 的 2.6% ⇒ |Δ目标合计| < 0.0101 才叫'打平'。")

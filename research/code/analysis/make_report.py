#!/usr/bin/env python3
"""汇总当前所有臂的状态与已出的五轴表 -> RESULTS.md(每轮巡检刷新)。"""
import io, json, os, re, subprocess, glob, statistics as st
O = os.path.dirname(os.path.abspath(__file__))
os.listdir(f"{O}/outputs")                      # 刷 NFS

ARMS = [  # (显示名, 目录, 对照, 干预说明)
 ("baseline(np4)", "ab_open3_baseline_s20260830_u100", "—", "官方 Open3 目标"),
 ("basenp3", "ab_open3_baseline_np3_s20260830_u100", "np4", "同 baseline,nproc=3(量评测口径的 nproc artifact)"),
 ("a1",   "ab_open3_a1_s20260830_u100",   "np4", "向量可行性 veto(绝对判据,**只触发 1.39%,等于空转**)"),
 ("n1",   "ab_open3_n1_s20260830_u100",   "np4", "尺度均衡加权"),
 ("n2",   "ab_open3_n2_s20260830_u100",   "np4", "冻结 \|r\| 归一化分母"),
 ("a3",   "ab_open3_a3_s20260830_u100",   "np4", "opa_cert=1 rollout 认证门控(触发约 10%)"),
 ("qopsd","ab_open3_qopsd_s20260830_u100","np3", "**Q-OPSD**:d_Q 方向(穿冻结后缀反传)+ 在线回溯"),
 ("qbt",  "ab_open3_qbt_s20260830_u100",  "np3", "消融:**同框架仅换回 canonical 方向**"),
 ("a1v2", "ab_open3_a1v2_s20260830_u100", "np3", "A1-v2:相对改善 veto(预计触发约 23%)"),
 ("b4x",  "ab_open3_b4x_s20260830_u25",   "np3", "4× batch(192 prompt × 24 图,**算力配平 25 upd**)"),
]

def last_epoch(d):
    p = f"{O}/outputs/{d}/metrics.jsonl"
    if not os.path.exists(p): return None
    m = -1
    for l in io.open(p):
        try: r = json.loads(l)
        except Exception: continue
        if "epoch" in r and "reward_avg" in r: m = max(m, r["epoch"])
    return m

def running():
    try:
        out = subprocess.run(["squeue","-u","chw4026","-h","-o","%j|%T|%M"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception: return {}
    d = {}
    for l in out.strip().split("\n"):
        if "|" in l:
            n,s,t = l.split("|")[:3]; d[n.strip()] = (s.strip(), t.strip())
    return d

def verdict(path):
    """从 TABLE_*.txt 抽出判定行与逐轴差。"""
    if not os.path.exists(path): return None
    txt = io.open(path).read()
    res = re.search(r"结果:(.+)", txt)
    rows = re.findall(r"^\s{2}(\S+)\s+([-\d.]+)\s+([-\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+).*?(打平|显著\S*)\(t=([-+][\d.]+)\)", txt, re.M)
    return (res.group(1).strip() if res else "?"), rows

L = []
L.append("# CA-OPD 结果汇总(自动生成)\n")
L.append("> 判据:`|差| > 2σ`(实测重测噪声)**且** `|t| > 2.5`;两者都过才算显著。\n")
L.append("> 训练期 `reward_avg` **不是**评测口径,判定一律以五轴表为准。\n")

# ── 设计分辨率:MDE(决定"打平"到底是什么意思)──────────────────
import math as _m0
def _J0(t):
    p=f"{O}/outputs/opsd_eval_{t}_ck100_raw.json"
    return json.load(io.open(p)) if os.path.exists(p) else None
_bb=_J0("ab_base")
if _bb:
    _s,_e=_bb["scores"],_bb["score_stats"]
    _vb=_s["pickscore"]/26+_s["clipscore"]+_s["hpsv2"]
    _eb=_m0.sqrt((_e["pickscore"]["se"]/26)**2+_e["clipscore"]["se"]**2+_e["hpsv2"]["se"]**2)
    _base=20.5868/26+0.2408+0.2070          # base SD3.5-M(见 TABLE 的 ③ 段)
    _gain=_vb-_base; _mde=2.5*_m0.hypot(_eb,_eb)
    L.append("\n## ⓪ 本设计的分辨率(决定『打平』是什么意思)\n")
    L.append(f"- 训练本身带来的目标增益(base → baseline@100upd):**{_gain:+.5f}**")
    L.append(f"- 两臂配对 SE(n=1000 held-out prompt):{_m0.hypot(_eb,_eb):.5f}")
    L.append(f"- **最小可检测效应 MDE(|t|>2.5):{_mde:.5f} = 训练总增益的 {_mde/_gain:.1%}**")
    L.append(f"\n> **所有『打平』的正确表述是 `|Δ| < {_mde:.4f}`,不是『等于零』。**")
    L.append(f"> 改进若小于训练总增益的 {_mde/_gain:.1%},本设计测不出来。\n")

L.append("\n## ① 已出的五轴表\n")
tabs = sorted(glob.glob(f"{O}/outputs/TABLE_*.txt"))
if not tabs: L.append("(暂无)\n")
for t in tabs:
    name = os.path.basename(t).replace("TABLE_","").replace("_ck100.txt","").replace("_ck25.txt","")
    v = verdict(t)
    if not v: continue
    concl, rows = v
    L.append(f"\n### {name}\n")
    L.append(f"**{concl}**\n")
    if rows:
        L.append("\n| 轴 | 对照 | 臂 | 差 | /2σ | t | 判定 |")
        L.append("|---|---|---|---|---|---|---|")
        for ax,b,a,d,s,j,tt in rows:
            L.append(f"| {ax} | {b} | {a} | {d} | {s} | {tt} | {j} |")
        L.append("")

# ── Open3 目标合计:方法真正优化的那个量(pick/26 + clip + hps),带合成 SE ──
import math
def _J(tag):
    p = f"{O}/outputs/opsd_eval_{tag}_ck100_raw.json"
    return json.load(io.open(p)) if os.path.exists(p) else None
def _agg(d):
    # ⚠️ 必须显式读 scores.*:JSON 里 train_reward_fn.{clipscore,hpsv2,pickscore}=1.0
    #    会遮蔽同名键,通用递归查找会取到 1.0,差值恒为 0(2026-09-01 踩过)。
    sc, se = d["scores"], d["score_stats"]
    v = sc["pickscore"]/26 + sc["clipscore"] + sc["hpsv2"]
    e = math.sqrt((se["pickscore"]["se"]/26)**2 + se["clipscore"]["se"]**2 + se["hpsv2"]["se"]**2)
    return v, e
_TAG = {"a1":"ab_a1","n1":"ab_n1","n2":"ab_n2","a3":"ab_a3","basenp3":"ab_basenp3",
        "qopsd":"ab_qopsd","qbt":"ab_qbt","a1v2":"ab_a1v2","b4x":"ab_b4x"}
_b = _J("ab_base")
if _b:
    _vb, _eb = _agg(_b)
    L.append("\n## ② Open3 **目标合计**(方法真正优化的量:pick/26 + clip + hps)\n")
    L.append("> 五轴分别判会掩盖净效果 —— 例如 n1 是『2 胜 2 负』,但按目标合计**显著更差**。\n")
    L.append("\n| 臂 | 合计差 | SE | t | 判定 |")
    L.append("|---|---|---|---|---|")
    for _lab in ("basenp3","a1","n1","n2","a3","qopsd","qbt","a1v2","b4x"):
        _d = _J(_TAG.get(_lab,""))
        if not _d: continue
        _va,_ea = _agg(_d); _dv = _va-_vb; _se = math.hypot(_ea,_eb); _t = _dv/_se
        _j = "**显著更差**" if _t<-2.5 else ("**显著更好**" if _t>2.5 else "打平")
        _n = _lab + ("(噪声基线)" if _lab=="basenp3" else "")
        L.append(f"| {_n} | {_dv:+.5f} | {_se:.5f} | {_t:+.2f} | {_j} |")
    L.append("")

L.append("\n## ② 各臂状态\n")
L.append("| 臂 | 干预 | 对照 | 进度 | 表 |")
L.append("|---|---|---|---|---|")
run = running()
for lab, d, base, desc in ARMS:
    ep = last_epoch(d)
    tot = 25 if "_u25" in d else 100
    jobname = {"qopsd":"qopsd2_g17","qbt":"qbt_np3","a1v2":"a1v2_np3","b4x":"b4x_np3",
                "basenp3":"bl_np3_full"}.get(lab)
    st_ = run.get(jobname, ("",""))[0] if jobname else ""
    prog = "—" if ep is None else (f"ep{ep+1}/{tot}" + (f" 🟢{st_}" if st_=="RUNNING" else (" ✅完成" if ep+1>=tot else " ⏸")))
    tf = f"{O}/outputs/TABLE_{lab}_vs_{base}_ck{tot}.txt"
    has = "✅" if os.path.exists(tf) else ("⏳" if base!="—" else "—")
    L.append(f"| **{lab}** | {desc} | {base} | {prog} | {has} |")

io.open(f"{O}/RESULTS.md","w").write("\n".join(L)+"\n")
print(f"✅ 已写 {O}/RESULTS.md ({len(L)} 行)")

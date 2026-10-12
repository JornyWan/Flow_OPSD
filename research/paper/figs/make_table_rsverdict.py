#!/usr/bin/env python
"""reward-specific 的对称呈现表(§4.750):每个 reward 一行,
在**它自己的训练 reward** 上的 4 步/40 步配对 t + 判定。阈值取自 §4.709(数据前登记)。"""
import sys, os, importlib.util, statistics as st
SP = "/midtier/minilab/scratch/chw4026/tmp/claude-1236520/-midtier-minilab-scratch-chw4026-projects-side-project-generative/5f4dc277-8a37-4d1d-8659-0c9b3c27bdfe/scratchpad"
spec = importlib.util.spec_from_file_location("judge", SP + "/judge.py")
J = importlib.util.module_from_spec(spec); sys.argv = [sys.argv[0]]
try: spec.loader.exec_module(J)
except SystemExit: pass

REWARDS = [("HPSv2.1", "hpsv2", "hpsv2"), ("PickScore", "pickscore", "pickscore"),
           ("CLIPScore", "clipscore", "clipscore"),
           ("ImageReward", "imagereward", "imagereward"), ("Aesthetic", "aesthetic", "aesthetic")]

def cell(rw, p, n, ax):
    fa = "matched_pi_rs%smsp_100_default_%s_flow%d" % (rw, p, n)
    fb = "matched_pi_rs%scanon_100_default_%s_flow%d" % (rw, p, n)
    if not (os.path.exists("outputs/%s.json" % fa) and os.path.exists("outputs/%s.json" % fb)): return None
    return J.paired(J.load(fa, ax), J.load(fb, ax))[3]

def basegate(rw, p, n):
    """§4.776 前置口径门:两臂的 open3 复合分都必须高于**未训练底模**(同 set 同 NFE)。
    任一臂低于底模 ⇒ 该配置没在学"质量",R 轴的问题不成立 ⇒ 该格口径否决。
    返回 True=通过,False=否决,None=缺底模格(无法判门)。"""
    fb = "matched_dflt_basemodel_none_%s_flow%d" % (p, n)
    if not os.path.exists("outputs/%s.json" % fb): return None
    base = st.mean(J.load(fb, "open3").values())
    for t in ("rs%scanon_100" % rw, "rs%smsp_100" % rw):
        f = "matched_pi_%s_default_%s_flow%d" % (t, p, n)
        if not os.path.exists("outputs/%s.json" % f): return None
        if st.mean(J.load(f, "open3").values()) < base: return False
    return True

def verdict(ts):
    """§4.709:主轴 flow-4,两集。R1 t>7.5 / R2 |t|<5.5 / R3 t<-7.5 / R4 5.5-7.5。"""
    if any(t is None for t in ts): return "--"
    def one(t): return "+" if t > 7.5 else ("$-$" if t < -7.5 else ("$0$" if abs(t) < 5.5 else "?"))
    vs = [one(t) for t in ts]
    return vs[0] if vs[0] == vs[1] else "/".join(vs)

out = ["% 自动生成,勿手改。源:paper_figs/make_table_rsverdict.py",
       "\\begin{center}", "\\begin{tabular}{lcccccc}", "\\toprule",
       "& \\multicolumn{2}{c}{NFE$=4$ (paired $t$)} & \\multicolumn{2}{c}{NFE$=40$} & &\\\\",
       "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
       "\\textbf{training reward} & DrawB. & Uns.-1k & DrawB. & Uns.-1k & deficit ratio & verdict\\\\",
       "\\midrule"]
print("  %-12s %9s %9s %9s %9s %8s %8s" % ("reward", "t@4 db", "t@4 u1k", "t@40 db", "t@40 u1k", "ratio", "verdict"))
def defi(tag, p, ax):
    a4 = "matched_pi_%s_default_%s_flow4" % (tag, p); a40 = "matched_pi_%s_default_%s_flow40" % (tag, p)
    if not (os.path.exists("outputs/%s.json" % a4) and os.path.exists("outputs/%s.json" % a40)): return None
    x, y = J.load(a4, ax), J.load(a40, ax); ks = sorted(set(x) & set(y))
    return st.mean(y[k] - x[k] for k in ks)
for lab, rw, ax in REWARDS:
    t4 = [cell(rw, p, 4, ax) for p in ("drawbench", "unseen1k")]
    t40 = [cell(rw, p, 40, ax) for p in ("drawbench", "unseen1k")]
    rs = []
    for p in ("drawbench", "unseen1k"):
        a, b = defi("rs%scanon_100" % rw, p, ax), defi("rs%smsp_100" % rw, p, ax)
        rs.append(None if (a is None or b is None or a == 0) else b / a)
    f = lambda t: "--" if t is None else "$%+.2f$" % t
    rr = "--" if any(r is None for r in rs) else "$%.2f$\\,/\\,$%.2f$" % tuple(rs)
    # §4.776:先过底模门。任一格否决 ⇒ 判定列写 n/a,**数字照留**(不隐去不利结果)。
    gates = [basegate(rw, p, n) for p in ("drawbench", "unseen1k") for n in (4, 40)]
    v = "n/a$^{\\dagger}$" if (False in gates) else verdict(t4)
    out.append("%s & %s & %s & %s & %s & %s & %s\\\\" % (lab, f(t4[0]), f(t4[1]), f(t40[0]), f(t40[1]), rr, v))
    print("  %-12s %9s %9s %9s %9s %8s %8s" % (lab,
          "--" if t4[0] is None else "%+.2f" % t4[0], "--" if t4[1] is None else "%+.2f" % t4[1],
          "--" if t40[0] is None else "%+.2f" % t40[0], "--" if t40[1] is None else "%+.2f" % t40[1],
          rr.replace("$", "").replace("\\,", ""), v.replace("$", "").replace("^{\\dagger}", "†")))
out += ["\\bottomrule", "\\end{tabular}",
        "\\\\[2pt]{\\footnotesize $^{\\dagger}$No verdict: both arms of this pair score \\emph{below the untrained}",
        "base model on the reported composite in all four cells, so the comparison is between two",
        "collapsed models and does not measure few-step quality. Numbers are shown unchanged.}",
        "\\end{center}"]
open("paper_figs/table_rsverdict.tex", "w").write("\n".join(out) + "\n")
print("\n  ✅ paper_figs/table_rsverdict.tex   (verdict: + 正向 / 0 无效 / - 负向 / ? 不判)")
